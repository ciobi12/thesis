import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatchClassificationEnv(gym.Env):
    """
    Patch-wise environment: bottom-left to top-right traversal.
    One step = one patch (N x N). The agent acts with a per-pixel MultiBinary (N*N) decision.

    Reward per pixel combines:
    - Base accuracy:     -|action - gt|
    - Optional continuity: -continuity_coef * |action - prev_pred|
    - Optional neighbour smoothness (within patch, 4-neighbours on image intensities):
        -neighbor_coef * sum_{(dx,dy) in {(1,0),(-1,0),(0,1),(0,-1)}} |action(i,j) - image(i+dy, j+dx)|
    """
    metadata = {"render_modes": []}

    def __init__(self, image, mask, patch_size=16, continuity_coef=0.0, neighbor_coef=0.1, start_from_bottom_left=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        self.image = img if img.ndim == 3 else img[..., None]  # (H, W, C)
        self.mask = (mask > 0).astype(np.float32)  # (H, W)

        self.H, self.W = self.mask.shape
        self.C = self.image.shape[2]
        self.N = int(patch_size)
        self.continuity_coef = float(continuity_coef)
        self.neighbor_coef = float(neighbor_coef)
        self.start_from_bottom_left = bool(start_from_bottom_left)

        # Pad image and mask to multiples of N for fixed-size patches
        H_pad = (self.N - (self.H % self.N)) % self.N
        W_pad = (self.N - (self.W % self.N)) % self.N
        self.pad = (H_pad, W_pad)
        if H_pad or W_pad:
            self.image = np.pad(self.image, ((0, H_pad), (0, W_pad), (0, 0)), mode="constant", constant_values=0)
            self.mask = np.pad(self.mask, ((0, H_pad), (0, W_pad)), mode="constant", constant_values=0)
        self.Hp, self.Wp = self.mask.shape

        self.patch_rows = self.Hp // self.N
        self.patch_cols = self.Wp // self.N

        # Build traversal order: bottom row to top row; within row left->right
        order = []
        row_range = range(self.patch_rows-1, -1, -1) if self.start_from_bottom_left else range(self.patch_rows)
        for pr in row_range:
            for pc in range(self.patch_cols):
                y0 = pr * self.N
                x0 = pc * self.N
                order.append((y0, x0))
        self._order = order
        self._idx = 0

        # Spaces: action is MultiBinary over N*N pixels in the patch
        self.action_space = spaces.MultiBinary(self.N * self.N)
        self.observation_space = spaces.Dict({
            "patch_pixels": spaces.Box(0.0, 1.0, shape=(self.N, self.N, self.C), dtype=np.float32),
            "prev_pred": spaces.Box(0.0, 1.0, shape=(self.N, self.N), dtype=np.float32),
            "patch_coords": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),  # (row_norm, col_norm)
        })

        self.prev_pred = np.zeros((self.N, self.N), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self.prev_pred.fill(0.0)
        return self._get_obs(), {}

    def _get_obs(self):
        y0, x0 = self._order[self._idx]
        patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N, N, C)
        pr = y0 // self.N
        pc = x0 // self.N
        coords = np.array([(pr + 1) / self.patch_rows, (pc + 1) / self.patch_cols], dtype=np.float32)
        return {
            "patch_pixels": patch.astype(np.float32),
            "prev_pred": self.prev_pred.astype(np.float32),
            "patch_coords": coords,
        }

    def step(self, action):
        # action: (N*N,) or (N,N)
        if action.ndim == 1:
            action = action.reshape(self.N, self.N)
        action = action.astype(np.float32)
        y0, x0 = self._order[self._idx]

        gt_patch = self.mask[y0:y0+self.N, x0:x0+self.N]  # (N,N)
        img_patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N,N,C)

        # Base reward against GT
        base_rewards = -np.abs(action - gt_patch)

        # Continuity with previous patch prediction
        continuity_rewards = -self.continuity_coef * np.abs(action - self.prev_pred)

        # Neighbour smoothness within patch based on image intensities (use mean channel)
        row_vals = img_patch.mean(axis=2)  # (N, N)
        neighbour_sum = np.zeros((self.N, self.N), dtype=np.float32)
        # 4-neighbour shifts: up, down, left, right
        # up
        neighbour_sum[1:, :] += np.abs(action[1:, :] - row_vals[:-1, :])
        # down
        neighbour_sum[:-1, :] += np.abs(action[:-1, :] - row_vals[1:, :])
        # left
        neighbour_sum[:, 1:] += np.abs(action[:, 1:] - row_vals[:, :-1])
        # right
        neighbour_sum[:, :-1] += np.abs(action[:, :-1] - row_vals[:, 1:])
        neighbour_rewards = -self.neighbor_coef * neighbour_sum

        pixel_rewards = base_rewards + continuity_rewards + neighbour_rewards
        reward = float(pixel_rewards.sum())

        # Update state and advance
        self.prev_pred = action.copy()
        self._idx += 1
        terminated = self._idx >= len(self._order)

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "neighbour_rewards": neighbour_rewards.astype(np.float32),
            "patch_origin": (int(y0), int(x0)),
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "patch_pixels": np.zeros((self.N, self.N, self.C), dtype=np.float32),
            "prev_pred": np.zeros((self.N, self.N), dtype=np.float32),
            "patch_coords": np.array([1.0, 1.0], dtype=np.float32),
        }

    def reconstruct_blank(self):
        """Helper to allocate a padded-size prediction canvas."""
        return np.zeros((self.Hp, self.Wp), dtype=np.uint8)

    def crop_to_original(self, arr2d):
        H_pad, W_pad = self.pad
        if H_pad or W_pad:
            return arr2d[: self.H, : self.W]
        return arr2d

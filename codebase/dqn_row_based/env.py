import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PathReconstructionEnv(gym.Env):
    """Row-wise environment: bottom to top. One step = one row.
    
    Reward components per pixel (summed across the row):
    - Base accuracy:     -|action - gt|
    - Row continuity:    -continuity_coef * |action - prev_row_action|
    - Neighbour smooth.: -neighbor_coef   * sum_{k in {±1,±2}} |action(j) - image(row, j+k)|
        (Only in-bounds neighbours are included.)
    """
    metadata = {"render_modes": []}

    def __init__(self, image, mask, continuity_coef=0.1, neighbor_coef=0.1, start_from_bottom=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        self.image = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        # print(np.unique(self.mask))
        self.H, self.W = self.mask.shape
        self.C = 1 if self.image.ndim == 2 else self.image.shape[2]
        self.continuity_coef = float(continuity_coef)
        self.neighbor_coef = float(neighbor_coef)
        self.start_from_bottom = bool(start_from_bottom)

        self.action_space = spaces.MultiBinary(self.W)
        self.observation_space = spaces.Dict({
            "row_pixels": spaces.Box(0.0, 1.0, shape=(self.W, self.C), dtype=np.float32),
            "prev_pred": spaces.Box(0.0, 1.0, shape=(self.W,), dtype=np.float32),
            "row_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._row_order = None
        self.current_row_idx = None
        self.prev_pred = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._row_order = np.arange(self.H-1, -1, -1) if self.start_from_bottom else np.arange(0, self.H)
        self.current_row_idx = 0
        self.prev_pred = np.zeros((self.W,), dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self._row_order[self.current_row_idx]
        return {
            "row_pixels": self.image[row][:, None] if self.C == 1 else self.image[row],  # shape (W, C)
            "prev_pred": self.prev_pred,
            "row_index": np.array([(row + 1) / self.H], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).clip(0, 1)
        row = self._row_order[self.current_row_idx]
        print(row)
        gt = self.mask[row]  # binary ground truth

        # Base pixel-wise accuracy term against ground-truth mask
        base_rewards = -np.abs(action - gt)

        # Temporal/row continuity term against previous row prediction
        continuity_rewards = -self.continuity_coef * np.abs(action - self.prev_pred)

        # Neighbour similarity term within the same row.
        # For each pixel j, compare action[j] with image(row, j±1, j±2) if in-bounds.
        if self.C == 1 and row != 0:
            row_vals = self.image[row - 1]  # shape (W,)
        else:
            # If multi-channel, use mean intensity
            row_vals = self.image[row].mean(axis=1)

        neighbour_sum = np.zeros((self.W,), dtype=np.float32)
        for d in (1, 2):
            left = np.roll(row_vals, +d)
            right = np.roll(row_vals, -d)
            valid_left = np.zeros(self.W, dtype=bool); valid_left[d:] = True
            valid_right = np.zeros(self.W, dtype=bool); valid_right[:-d] = True
            # Accumulate absolute differences only where neighbours are valid
            neighbour_sum[valid_left] += np.abs(action[valid_left] - left[valid_left])
            neighbour_sum[valid_right] += np.abs(action[valid_right] - right[valid_right])

        neighbour_rewards = -self.neighbor_coef * neighbour_sum

        # Total per-pixel rewards
        pixel_rewards = base_rewards + continuity_rewards + neighbour_rewards
        reward = float(pixel_rewards.sum())

        self.prev_pred = action.copy()
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "neighbour_rewards": neighbour_rewards.astype(np.float32),
            "row_index": row,
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last row, return a zeroed observation (won't be used)
        return {
            "row_pixels": np.zeros((self.W, self.C), dtype=np.float32),
            "prev_pred": self.prev_pred.astype(np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
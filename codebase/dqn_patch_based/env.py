import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class PatchClassificationEnv(gym.Env):
    """
    Patch-wise environment: bottom-left to top-right traversal.
    One step = one patch (N x N). The agent acts with a per-pixel MultiBinary (N*N) decision.

    Reward per pixel combines:
    - Base accuracy:     -base_coef * |action - gt|
    - Continuity: penalty for isolated active pixels not connected to previous prediction
    - Gradient-based: rewards smooth intensity transitions between connected patches
    - Optional neighbour smoothness (within patch, 4-neighbours on image intensities):
        -neighbor_coef * sum_{(dx,dy) in {(1,0),(-1,0),(0,1),(0,-1)}} |action(i,j) - image(i+dy, j+dx)|
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 image, 
                 mask, 
                 patch_size=16, 
                 base_coef=1.0,
                 continuity_coef=0.1, 
                 gradient_coef=0.0,
                 neighbor_coef=0.1, 
                 history_len=3,
                 future_len=3,
                 start_from_bottom_left=True):
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
        self.base_coef = float(base_coef)
        self.continuity_coef = float(continuity_coef)
        self.gradient_coef = float(gradient_coef)
        self.neighbor_coef = float(neighbor_coef)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
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
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.N, self.N), dtype=np.float32),
            "prev_patches": spaces.Box(0.0, 1.0, shape=(self.history_len, self.N, self.N, self.C), dtype=np.float32),
            "future_patches": spaces.Box(0.0, 1.0, shape=(self.future_len, self.N, self.N, self.C), dtype=np.float32),
            "patch_coords": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),  # (row_norm, col_norm)
        })

        # History buffers
        self.prev_preds_buffer = None
        self.prev_patches_buffer = None
        
        # Episode-level accumulators for metrics
        self.episode_predictions = None
        self.episode_ground_truth = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        
        # Initialize history buffers
        self.prev_preds_buffer = deque(
            [np.zeros((self.N, self.N), dtype=np.float32) for _ in range(self.history_len)], 
            maxlen=self.history_len
        )
        self.prev_patches_buffer = deque(
            [np.zeros((self.N, self.N, self.C), dtype=np.float32) for _ in range(self.history_len)], 
            maxlen=self.history_len
        )
        
        # Initialize episode accumulators
        self.episode_predictions = np.zeros((self.Hp, self.Wp), dtype=np.float32)
        self.episode_ground_truth = self.mask.copy()
        
        return self._get_obs(), {}

    def _get_obs(self):
        y0, x0 = self._order[self._idx]
        patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N, N, C)
        pr = y0 // self.N
        pc = x0 // self.N
        coords = np.array([(pr + 1) / self.patch_rows, (pc + 1) / self.patch_cols], dtype=np.float32)
        
        # Get future patches (lookahead)
        future_patches = []
        for i in range(1, self.future_len + 1):
            future_idx = self._idx + i
            if future_idx < len(self._order):
                fy0, fx0 = self._order[future_idx]
                future_patches.append(self.image[fy0:fy0+self.N, fx0:fx0+self.N, :])
            else:
                # Pad with zeros if we're near the end
                future_patches.append(np.zeros((self.N, self.N, self.C), dtype=np.float32))
        
        return {
            "patch_pixels": patch.astype(np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),  # (history_len, N, N)
            "prev_patches": np.array(self.prev_patches_buffer, dtype=np.float32),  # (history_len, N, N, C)
            "future_patches": np.array(future_patches, dtype=np.float32),  # (future_len, N, N, C)
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

        # ========== ENHANCED REWARD FOR SEGMENTATION ==========
        
        # 1. BASE REWARD: Accuracy against ground truth
        base_rewards = -self.base_coef * np.abs(action - gt_patch)
        
        # Track per-patch stats
        true_positives = np.sum((action == 1) & (gt_patch == 1))
        false_positives = np.sum((action == 1) & (gt_patch == 0))
        false_negatives = np.sum((action == 0) & (gt_patch == 1))
        
        # Store prediction for this patch
        self.episode_predictions[y0:y0+self.N, x0:x0+self.N] = (action > 0.5).astype(np.float32)

        # 2. CONTINUITY REWARD: Penalty for isolated active pixels
        continuity_rewards = np.zeros((self.N, self.N), dtype=np.float32)
        
        if len(self.prev_preds_buffer) >= 1:
            prev_pred = list(self.prev_preds_buffer)[-1]  # (N, N)
            
            # For each active pixel, check if it connects to previous prediction
            curr_active = np.where(action == 1)
            for i, j in zip(curr_active[0], curr_active[1]):
                # Check 5-pixel neighborhood in previous prediction
                i_start = max(0, i - 2)
                i_end = min(self.N, i + 3)
                j_start = max(0, j - 2)
                j_end = min(self.N, j + 3)
                
                # If connected to previous prediction, don't penalize
                if np.any(prev_pred[i_start:i_end, j_start:j_end] == 1):
                    continuity_rewards[i, j] = 0
                else:
                    # Penalty for isolated pixels
                    continuity_rewards[i, j] = -1.0
            
            # Don't penalize correctly predicting background
            inactive_with_inactive_prev = ((action == 0) & (prev_pred == 0))
            continuity_rewards[inactive_with_inactive_prev] = 0.0
        
        continuity_rewards *= self.continuity_coef

        # 3. GRADIENT-BASED REWARD: Encourage smooth intensity transitions
        gradient_rewards = np.zeros((self.N, self.N), dtype=np.float32)
        
        if len(self.prev_patches_buffer) >= 1 and self.gradient_coef > 0:
            prev_patch_img = list(self.prev_patches_buffer)[-1]  # (N, N, C)
            prev_pred = list(self.prev_preds_buffer)[-1]
            
            prev_active = np.where(prev_pred == 1)
            curr_active = np.where(action == 1)
            
            # Current patch intensity (mean channel)
            curr_intensity = img_patch.mean(axis=2)  # (N, N)
            prev_intensity = prev_patch_img.mean(axis=2)  # (N, N)
            
            if len(prev_active[0]) > 0 and len(curr_active[0]) > 0:
                for i, j in zip(curr_active[0], curr_active[1]):
                    # Find closest active pixel in previous prediction
                    distances = np.sqrt((prev_active[0] - i)**2 + (prev_active[1] - j)**2)
                    min_idx = np.argmin(distances)
                    closest_i, closest_j = prev_active[0][min_idx], prev_active[1][min_idx]
                    
                    if distances[min_idx] > 5:
                        continue  # Skip if too far
                    
                    # Calculate intensity difference
                    intensity_diff = np.abs(curr_intensity[i, j] - prev_intensity[closest_i, closest_j])
                    
                    # Reward small intensity differences (smooth transitions)
                    gradient_rewards[i, j] = -intensity_diff
        
        gradient_rewards *= self.gradient_coef

        # 4. NEIGHBOUR SMOOTHNESS: within patch based on image intensities
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

        # Combine all rewards
        pixel_rewards = base_rewards + continuity_rewards + gradient_rewards + neighbour_rewards
        reward = float(pixel_rewards.sum())

        # Update history buffers
        self.prev_preds_buffer.append(action.copy())
        self.prev_patches_buffer.append(img_patch.copy())
        
        self._idx += 1
        terminated = self._idx >= len(self._order)

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "gradient_rewards": gradient_rewards.astype(np.float32),
            "neighbour_rewards": neighbour_rewards.astype(np.float32),
            "patch_origin": (int(y0), int(x0)),
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "patch_pixels": np.zeros((self.N, self.N, self.C), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_patches": np.array(self.prev_patches_buffer, dtype=np.float32),
            "future_patches": np.zeros((self.future_len, self.N, self.N, self.C), dtype=np.float32),
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

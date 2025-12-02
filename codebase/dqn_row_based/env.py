import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class PathReconstructionEnv(gym.Env):
    """Row-wise environment: bottom to top. One step = one row.
    
    Reward components per pixel (summed across the row):
    - Base accuracy:     -|action - gt|
    - Row continuity:    -continuity_coef * |action - prev_rows_action|
        (Only in-bounds neighbours are included.)
    """
    metadata = {"render_modes": []}

    def __init__(self, image, 
                 mask, 
                 continuity_coef=0.1, 
                 continuity_decay_factor=0.7,
                 history_len=3, 
                 start_from_bottom=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        self.image = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        # print(np.unique(self.mask))
        self.H, self.W = self.mask.shape
        self.C = 1 if self.image.ndim == 2 else self.image.shape[2]
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)

        self.action_space = spaces.MultiBinary(self.W)
        self.observation_space = spaces.Dict({
            "row_pixels": spaces.Box(0.0, 1.0, shape=(self.W, self.C), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.W), dtype=np.float32),
            "row_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._row_order = None
        self.current_row_idx = None
        self.prev_preds_buffer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._row_order = np.arange(self.H-1, -1, -1) if self.start_from_bottom else np.arange(0, self.H)
        self.current_row_idx = 0
        self.prev_preds_buffer = deque([np.zeros((self.W,), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self._row_order[self.current_row_idx]
        return {
            "row_pixels": self.image[row][:, None] if self.C == 1 else self.image[row],  # shape (W, C)
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),  # (history_len, W)
            "row_index": np.array([(row + 1) / self.H], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).clip(0, 1)
        row = self._row_order[self.current_row_idx]
        gt = self.mask[row]  # binary ground truth

        # Base pixel-wise accuracy term against ground-truth mask
        base_rewards = -np.abs(action - gt)

        # ========== MULTI-ROW CONTINUITY STRATEGIES ==========
        # Choose ONE strategy by uncommenting its block
        
        # # STRATEGY 1: Exponential Decay (CURRENTLY ACTIVE)
        # # Links to all previous rows with decaying weights (recent rows weighted more)
        # continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        # decay_factor = self.continuity_decay_factor  # tune: 0.5-0.9 (higher = longer memory)
        # for i, prev_row in enumerate(reversed(list(self.prev_preds_buffer))):
        #     weight = (decay_factor ** i)  # 1.0, 0.7, 0.49, ...
        #     continuity_rewards += -weight * np.abs(action - prev_row)
        # continuity_rewards *= self.continuity_coef
        
        ## STRATEGY 2: Vertical Alignment (UNCOMMENT TO USE)
        # Rewards when action aligns with any previous row within tolerance
        # continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        # tolerance = 2
        # for j in range(self.W):
        #     if action[j] > 0.5:
        #         aligned = any(
        #             np.any(prev_row[max(0, j-tolerance):min(self.W, j+tolerance+1)] > 0.5)
        #             for prev_row in self.prev_preds_buffer
        #         )
        #         continuity_rewards[j] = 0.5 if aligned else -1.0
        # continuity_rewards *= self.continuity_coef
        
        # STRATEGY 3: Hybrid Decay + Alignment (UNCOMMENT TO USE)
        # Combines smoothness with connectivity checking
        smooth_rewards = np.zeros((self.W,), dtype=np.float32)
        decay_factor = 0.7
        for i, prev_row in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            smooth_rewards += -weight * np.abs(action - prev_row)
        alignment_rewards = np.zeros((self.W,), dtype=np.float32)
        tolerance = 2
        for j in range(self.W):
            if action[j] > 0.5:
                aligned = any(
                    np.any(prev_row[max(0, j-tolerance):min(self.W, j+tolerance+1)] > 0.5)
                    for prev_row in self.prev_preds_buffer
                )
                alignment_rewards[j] = 0.5 if aligned else -1.0
        continuity_rewards = (smooth_rewards + 0.5 * alignment_rewards) * self.continuity_coef
       
        # Total per-pixel rewards
        pixel_rewards = base_rewards + continuity_rewards
                       
        reward = float(pixel_rewards.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "row_index": row,
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last row, return a zeroed observation (won't be used)
        return {
            "row_pixels": np.zeros((self.W, self.C), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
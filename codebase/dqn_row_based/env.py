import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PathReconstructionEnv(gym.Env):
    """Row-wise environment: bottom to top. One step = one row."""
    metadata = {"render_modes": []}

    def __init__(self, image_mask, continuity_coef=0.1, start_from_bottom=True):
        super().__init__()
        assert image_mask.ndim == 2 and image_mask.dtype in (np.uint8, np.int32, np.int64, np.bool_, np.float32)
        self.image = (image_mask.astype(np.float32) > 0).astype(np.float32)  # binary (H, W)
        self.H, self.W = self.image.shape
        self.continuity_coef = float(continuity_coef)
        self.start_from_bottom = bool(start_from_bottom)

        # Actions: per-pixel binary (0=not path, 1=path)
        self.action_space = spaces.MultiBinary(self.W)

        # Observation: current row pixels, previous row prediction, normalized row index
        self.observation_space = spaces.Dict({
            "row_pixels": spaces.Box(0.0, 1.0, shape=(self.W,), dtype=np.float32),
            "prev_pred": spaces.Box(0.0, 1.0, shape=(self.W,), dtype=np.float32),
            "row_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._row_order = None
        self.current_row_idx = None
        self.prev_pred = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Bottom-to-top order
        self._row_order = np.arange(self.H-1, -1, -1) if self.start_from_bottom else np.arange(0, self.H)
        self.current_row_idx = 0
        self.prev_pred = np.zeros((self.W,), dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self._row_order[self.current_row_idx]
        return {
            "row_pixels": self.image[row].astype(np.float32),
            "prev_pred": self.prev_pred.astype(np.float32),
            "row_index": np.array([(row + 1) / self.H], dtype=np.float32),  # 1..H normalized (avoid 0)
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).clip(0, 1)
        row = self._row_order[self.current_row_idx]
        gt = self.image[row]  # shape (W,)

        # Per-pixel accuracy reward: -|a - y|
        base_rewards = -np.abs(action - gt)
        # Per-pixel continuity reward: penalize difference from previous row prediction
        continuity_rewards = -self.continuity_coef * np.abs(action - self.prev_pred)
        pixel_rewards = base_rewards + continuity_rewards

        # Total row reward
        reward = float(pixel_rewards.sum())

        # Advance
        self.prev_pred = action.copy()
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H

        # Info returns pixel rewards for per-pixel training targets
        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "row_index": row,
        }
        # Truncated is always False here
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last row, return a zeroed observation (won't be used)
        return {
            "row_pixels": np.zeros((self.W,), dtype=np.float32),
            "prev_pred": self.prev_pred.astype(np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
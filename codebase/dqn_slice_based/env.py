import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage

class PathReconstructionEnv(gym.Env):
    """Slice-wise environment: bottom to top. One step = one slice.
    
    Reward components per pixel (summed across the slice):
    - Base accuracy:     -|action - gt|
    - Slice continuity:  -continuity_coef * |action - prev_slices_action|
        (Only in-bounds neighbours are included.)
    """
    metadata = {"render_modes": []}

    def __init__(self, volume, 
                 mask, 
                 continuity_coef=0.1, 
                 continuity_decay_factor=0.7,
                 history_len=3, 
                 start_from_bottom=True):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        # Volume shape: (D, H, W) where D=64, H=16, W=16
        # print(self.mask.shape)
        self.D, self.H, self.W = self.mask.shape
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)

        self.action_space = spaces.MultiBinary(self.H * self.W)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(self.H, self.W), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "prev_slices": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "slice_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._slice_order = None
        self.current_slice_idx = None
        self.prev_preds_buffer = None  # Keep for reward calculation
        self.prev_slices_buffer = None   # New: for network input

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._slice_order = np.arange(0, self.D) if self.start_from_bottom else np.arange(self.D-1, -1, -1)
        self.current_slice_idx = 0
        self.prev_preds_buffer = deque([np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.prev_slices_buffer = deque([np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        return self._get_obs(), {}

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        # print(self.volume[slice_idx,:,:].shape)
        return {
            "slice_pixels": self.volume[slice_idx,:,:],  # shape (H, W)
            
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),  # (history_len, H, W)
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),  # (history_len, H, W)
            "slice_index": np.array([(slice_idx + 1) / self.D], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.H, self.W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        gt = self.mask[slice_idx, :, :]

        # Base rewards
        base_rewards = -np.abs(action - gt)

        # Existing continuity (keep your current strategy)
        continuity_rewards = np.zeros((self.H, self.W), dtype=np.float32)
        decay_factor = self.continuity_decay_factor
        for i, prev_slice in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            continuity_rewards += -weight * np.abs(action - prev_slice)
        continuity_rewards *= self.continuity_coef

       
        # Total pixel rewards (keep per-pixel for compatibility)
        pixel_rewards = base_rewards + continuity_rewards
        reward = float(pixel_rewards.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(self.volume[slice_idx,:,:].copy())
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "slice_index": slice_idx,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last slice, return a zeroed observation (won't be used)
        return {
            "slice_pixels": np.zeros((self.H, self.W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }
"""
Enhanced PathReconstructionEnv with DICE-based reward component.

Reward components per pixel (summed across the slice):
- Base accuracy:     -|action - gt|
- Slice continuity:  -continuity_coef * |action - prev_slices_action|
- DICE reward:       dice_coef * DICE(action, gt) per slice
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class PathReconstructionEnvDice(gym.Env):
    """Slice-wise environment with DICE-based reward.
    
    Adds a DICE score component to reward segmentation quality directly,
    which is especially important for class-imbalanced data.
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 volume, 
                 mask, 
                 continuity_coef=0.1, 
                 continuity_decay_factor=0.7,
                 dice_coef=1.0,
                 history_len=3, 
                 start_from_bottom=True):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        
        self.D, self.H, self.W = self.mask.shape
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.dice_coef = float(dice_coef)
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
        self.prev_preds_buffer = None
        self.prev_slices_buffer = None
        
        # Track predictions for episode-level DICE
        self.episode_predictions = None
        self.episode_ground_truth = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._slice_order = np.arange(0, self.D) if self.start_from_bottom else np.arange(self.D-1, -1, -1)
        self.current_slice_idx = 0
        self.prev_preds_buffer = deque(
            [np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)], 
            maxlen=self.history_len
        )
        self.prev_slices_buffer = deque(
            [np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)], 
            maxlen=self.history_len
        )
        
        # Reset episode tracking
        self.episode_predictions = np.zeros((self.D, self.H, self.W), dtype=np.float32)
        self.episode_ground_truth = self.mask.copy()
        
        return self._get_obs(), {}

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        return {
            "slice_pixels": self.volume[slice_idx, :, :],
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([(slice_idx + 1) / self.D], dtype=np.float32),
        }

    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute DICE score between prediction and ground truth."""
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        
        intersection = (pred_bin * gt_bin).sum()
        total = pred_bin.sum() + gt_bin.sum()
        
        if total == 0:
            return 1.0  # Both empty = perfect match
        
        return 2.0 * intersection / (total + 1e-8)

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.H, self.W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        gt = self.mask[slice_idx, :, :]

        # ============================================================
        # 1. Base rewards: per-pixel accuracy
        # ============================================================
        base_rewards = -np.abs(action - gt)

        # ============================================================
        # 2. Continuity rewards: smooth transitions between slices
        # ============================================================
        continuity_rewards = np.zeros((self.H, self.W), dtype=np.float32)
        decay_factor = self.continuity_decay_factor
        for i, prev_slice in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            continuity_rewards += -weight * np.abs(action - prev_slice)
        continuity_rewards *= self.continuity_coef

        # ============================================================
        # 3. DICE reward: slice-level segmentation quality
        # ============================================================
        slice_dice = self._compute_dice(action, gt)
        # Scale DICE to be comparable with other rewards
        # DICE is in [0, 1], we want positive reward for good segmentation
        dice_reward = self.dice_coef * slice_dice
        
        # Store prediction for episode-level tracking
        self.episode_predictions[slice_idx] = action
        
        # ============================================================
        # Total reward
        # ============================================================
        # Per-pixel rewards (for DQN training)
        pixel_rewards = base_rewards + continuity_rewards
        
        # Add DICE as a bonus distributed across foreground/all pixels
        # Option 1: Add uniformly to all pixels
        # Option 2: Add only to foreground pixels (more targeted)
        # We use option 1 for simplicity
        dice_bonus_per_pixel = dice_reward / (self.H * self.W)
        pixel_rewards_with_dice = pixel_rewards + dice_bonus_per_pixel
        
        reward = float(pixel_rewards_with_dice.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(self.volume[slice_idx, :, :].copy())
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        # Compute episode DICE if terminated
        episode_dice = None
        if terminated:
            episode_dice = self._compute_dice(self.episode_predictions, self.episode_ground_truth)

        info = {
            "pixel_rewards": pixel_rewards_with_dice.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "dice_reward": dice_reward,
            "slice_dice": slice_dice,
            "slice_index": slice_idx,
            "episode_dice": episode_dice,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "slice_pixels": np.zeros((self.H, self.W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }


# Alias for compatibility
PathReconstructionEnv = PathReconstructionEnvDice

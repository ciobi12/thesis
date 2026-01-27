"""
Enhanced PathReconstructionEnv with DICE-based reward component.

Reward components per pixel (summed across the slice):
- Base accuracy:     -|action - gt|
- Slice continuity:  -continuity_coef * |action - prev_slices_action|
- Manhattan reward:  manhattan_coef * manhattan_distance(action, gt) per slice
- DICE reward:       dice_coef * DICE(pred_volume, gt_volume) at episode end
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class PathReconstructionEnv(gym.Env):
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
                 gradient_coef=0.1,
                 dice_coef=1.0,
                 manhattan_coef=0.0,
                 history_len=5,
                 future_len=3,
                 start_from_top = True):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        
        self.D, self.H, self.W = self.mask.shape
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.gradient_coef = float(gradient_coef)
        self.dice_coef = float(dice_coef)
        self.manhattan_coef = float(manhattan_coef)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.start_from_top = bool(start_from_top)

        self.action_space = spaces.MultiBinary(self.H * self.W)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(self.H, self.W), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "prev_slices": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "future_slices": spaces.Box(0.0, 1.0, shape=(self.future_len, self.H, self.W), dtype=np.float32),
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
        self._slice_order = np.arange(self.D-1, -1, -1) if self.start_from_top  else np.arange(0, self.D) 
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
        
        # Get future slices (next future_len slices in processing order)
        future_slices = []
        for i in range(1, self.future_len + 1):
            future_step_idx = self.current_slice_idx + i
            if future_step_idx < self.D:
                future_slice_idx = self._slice_order[future_step_idx]
                future_slices.append(self.volume[future_slice_idx, :, :])
            else:
                # Pad with zeros if we're near the end
                future_slices.append(np.zeros((self.H, self.W), dtype=np.float32))
        
        return {
            "slice_pixels": self.volume[slice_idx, :, :],
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "future_slices": np.array(future_slices, dtype=np.float32),
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

    def _compute_slice_manhattan_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute average Manhattan distance from predicted pixels to nearest GT pixels (2D slice).
        
        Returns negative distance (as reward should be higher when distance is lower).
        Normalized by slice diagonal to keep scale reasonable.
        
        Args:
            pred: Binary prediction slice (H, W)
            gt: Binary ground truth slice (H, W)
            
        Returns:
            Negative normalized average Manhattan distance (higher = better)
        """
        pred_bin = (pred > 0.5)
        gt_bin = (gt > 0.5)
        
        pred_coords = np.argwhere(pred_bin)  # (N_pred, 2)
        gt_coords = np.argwhere(gt_bin)      # (N_gt, 2)
        
        # Edge cases
        if len(pred_coords) == 0 and len(gt_coords) == 0:
            return 0.0  # Both empty = perfect
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            # One is empty - return large penalty
            return -1.0  # Normalized max penalty
        
        # For each predicted pixel, find distance to nearest GT pixel
        # Using broadcasting: (N_pred, 1, 2) - (1, N_gt, 2) -> (N_pred, N_gt, 2)
        # Sample if needed for efficiency
        max_samples = 500
        if len(pred_coords) > max_samples:
            sample_idx = np.random.choice(len(pred_coords), max_samples, replace=False)
            pred_coords = pred_coords[sample_idx]
        if len(gt_coords) > max_samples:
            sample_idx = np.random.choice(len(gt_coords), max_samples, replace=False)
            gt_coords = gt_coords[sample_idx]
        
        # Compute pairwise Manhattan distances
        # |h1 - h2| + |w1 - w2|
        diff = np.abs(pred_coords[:, np.newaxis, :] - gt_coords[np.newaxis, :, :])  # (N_pred, N_gt, 2)
        manhattan = diff.sum(axis=2)  # (N_pred, N_gt)
        
        # For each pred pixel, get distance to nearest GT pixel
        min_distances = manhattan.min(axis=1)  # (N_pred,)
        avg_distance = min_distances.mean()
        
        # Normalize by slice diagonal
        diag = self.H + self.W  # Max possible Manhattan distance in 2D
        normalized_distance = avg_distance / diag
        
        # Return negative (reward should increase as distance decreases)
        return -normalized_distance * diag

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.H, self.W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        gt = self.mask[slice_idx, :, :]

        # ============================================================
        # 1. Base rewards: per-pixel accuracy
        # ============================================================
        base_rewards = -np.abs(action - gt)

        # ============================================================
        # 2. Continuity AND gradient rewards: smooth transitions between slices
        # ============================================================
        continuity_rewards = np.zeros((self.H, self.W), dtype=np.float32)
        for i, prev_pred in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (self.continuity_decay_factor ** i)

            # Identify active pixels in the current slice
            active_pixels = np.argwhere(action == 1)

            for pixel in active_pixels:
                h, w = pixel

                # Define the 3x3 patch around the active pixel
                h_min, h_max = max(0, h - 1), min(self.H, h + 2)
                w_min, w_max = max(0, w - 1), min(self.W, w + 2)

                # Check if there are active pixels in the patch in the previous slice
                patch = prev_pred[h_min:h_max, w_min:w_max]
                if np.any(patch == 1):
                    continuity_rewards[h, w] += 0  # Reward 0 if there are active pixels in the patch
                else:
                    continuity_rewards[h, w] += -weight * 5  # Negative rewards otherwise

        continuity_rewards *= self.continuity_coef

        gradient_rewards = 0.0
        decay_factor = self.continuity_decay_factor
        for i, (prev_pred, prev_slice) in enumerate(zip(reversed(list(self.prev_preds_buffer)), reversed(list(self.prev_slices_buffer)))):
            weight = (decay_factor ** i)
            # if len(self.prev_preds_buffer) > 0:
            #     print(self.volume[slice_idx, :, :][action == 1].sum() - prev_slice[prev_pred == 1].sum())
            gradient_rewards += -weight * np.abs(action - prev_pred)

            # Assuming the volume has one root channel (grayscale intensity)
            curr_slice_avg_root_intensity = self.volume[slice_idx, :, :][action == 1].mean() if np.any(action == 1) else 0.0
            prev_slice_avg_root_intensity = prev_slice[prev_pred == 1].mean() if np.any(prev_pred == 1) else 0.0
            gradient_rewards += -np.abs(curr_slice_avg_root_intensity - prev_slice_avg_root_intensity)
        
        continuity_rewards *= self.continuity_coef
        gradient_rewards *= self.gradient_coef
        
        # ============================================================
        # 3. Manhattan reward: per-slice distance quality
        # ============================================================
        manhattan_reward = 0.0
        if self.manhattan_coef > 0:
            slice_manhattan = self._compute_slice_manhattan_distance(action, gt)
            manhattan_reward = self.manhattan_coef * slice_manhattan
        
        # Store prediction for episode-level tracking
        self.episode_predictions[slice_idx] = action
        
        # ============================================================
        # Total reward
        # ============================================================
        # Per-pixel rewards (for DQN training)
        pixel_rewards = base_rewards + continuity_rewards + gradient_rewards
        
        # Add Manhattan as a bonus distributed across all pixels
        manhattan_bonus_per_pixel = manhattan_reward / (self.H * self.W)
        pixel_rewards_with_manhattan = pixel_rewards + manhattan_bonus_per_pixel
        
        reward = float(pixel_rewards_with_manhattan.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(self.volume[slice_idx, :, :].copy())
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        # Compute episode-level metrics and DICE reward if terminated
        episode_dice = None
        dice_reward = 0.0
        if terminated:
            episode_dice = self._compute_dice(self.episode_predictions, self.episode_ground_truth)
            # Compute DICE reward at episode end
            if self.dice_coef > 0:
                dice_reward = self.dice_coef * episode_dice
                # Add DICE reward to the final step's total reward
                reward += dice_reward

        info = {
            "pixel_rewards": pixel_rewards_with_manhattan.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "gradient_rewards": gradient_rewards.astype(np.float32),
            "manhattan_reward": manhattan_reward,
            "dice_reward": dice_reward,
            "slice_index": slice_idx,
            "episode_dice": episode_dice,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "slice_pixels": np.zeros((self.H, self.W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "future_slices": np.zeros((self.future_len, self.H, self.W), dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }

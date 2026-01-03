"""
Enhanced PathReconstructionEnv with Global Coherence Reward.

Key improvements over env_dice.py:
1. Boundary consistency: Penalizes discontinuities at subvolume edges
2. Connectivity reward: Rewards connected components, penalizes fragmentation
3. Neighbor context: Can incorporate predictions from adjacent subvolumes
4. Multi-scale continuity: Considers both local (slice) and regional (subvolume) coherence

Reward components per pixel (summed across the slice):
- Base accuracy:     -|action - gt|
- Slice continuity:  -continuity_coef * |action - prev_slices_action|
- DICE reward:       dice_coef * DICE(action, gt) per slice
- Boundary reward:   boundary_coef * consistency with neighbor predictions (at episode end)
- Connectivity:      connectivity_coef * (1 - fragmentation_penalty) (at episode end)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage


class PathReconstructionEnvGlobalCoherence(gym.Env):
    """Slice-wise environment with global coherence signals.
    
    Adds boundary consistency and connectivity rewards to capture
    inter-subvolume structure and reduce fragmentation.
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 volume, 
                 mask, 
                 continuity_coef=0.1, 
                 continuity_decay_factor=0.7,
                 dice_coef=1.0,
                 boundary_coef=0.5,
                 connectivity_coef=0.3,
                 history_len=3, 
                 start_from_bottom=True,
                 neighbor_context=None):
        """
        Args:
            volume: 3D volume array (D, H, W)
            mask: 3D binary mask (D, H, W)
            continuity_coef: Weight for slice-to-slice continuity
            continuity_decay_factor: Exponential decay for history
            dice_coef: Weight for DICE-based reward
            boundary_coef: Weight for boundary consistency with neighbors
            connectivity_coef: Weight for connectivity/fragmentation penalty
            history_len: Number of previous slices to consider
            start_from_bottom: Process slices bottom-up (True) or top-down (False)
            neighbor_context: Dict with predictions from neighboring subvolumes
                              Keys: 'top', 'bottom', 'left', 'right', 'front', 'back'
                              Values: 2D arrays of boundary predictions
        """
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)
        
        self.D, self.H, self.W = self.mask.shape
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.dice_coef = float(dice_coef)
        self.boundary_coef = float(boundary_coef)
        self.connectivity_coef = float(connectivity_coef)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)
        
        # Neighbor context for boundary consistency
        self.neighbor_context = neighbor_context or {}

        self.action_space = spaces.MultiBinary(self.H * self.W)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(self.H, self.W), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "prev_slices": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "slice_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # New: neighbor boundary context
            "neighbor_hints": spaces.Box(0.0, 1.0, shape=(6, self.H, self.W), dtype=np.float32),
        })

        self._slice_order = None
        self.current_slice_idx = None
        self.prev_preds_buffer = None
        self.prev_slices_buffer = None
        
        # Track predictions for episode-level metrics
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
        
        self.episode_predictions = np.zeros((self.D, self.H, self.W), dtype=np.float32)
        self.episode_ground_truth = self.mask.copy()
        
        return self._get_obs(), {}

    def _get_neighbor_hints(self, slice_idx):
        """Get boundary hints from neighboring subvolumes."""
        hints = np.zeros((6, self.H, self.W), dtype=np.float32)
        
        # Order: [front, back, top, bottom, left, right]
        # front/back: along D axis
        # top/bottom: along H axis  
        # left/right: along W axis
        
        if 'front' in self.neighbor_context and slice_idx == 0:
            hints[0] = self.neighbor_context['front']
        if 'back' in self.neighbor_context and slice_idx == self.D - 1:
            hints[1] = self.neighbor_context['back']
        
        # H/W boundaries are constant across all slices
        if 'top' in self.neighbor_context:
            hints[2, 0, :] = self.neighbor_context['top']
        if 'bottom' in self.neighbor_context:
            hints[3, -1, :] = self.neighbor_context['bottom']
        if 'left' in self.neighbor_context:
            hints[4, :, 0] = self.neighbor_context['left']
        if 'right' in self.neighbor_context:
            hints[5, :, -1] = self.neighbor_context['right']
            
        return hints

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        return {
            "slice_pixels": self.volume[slice_idx, :, :],
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([(slice_idx + 1) / self.D], dtype=np.float32),
            "neighbor_hints": self._get_neighbor_hints(slice_idx),
        }

    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute DICE score between prediction and ground truth."""
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        
        intersection = (pred_bin * gt_bin).sum()
        total = pred_bin.sum() + gt_bin.sum()
        
        if total == 0:
            return 1.0
        
        return 2.0 * intersection / (total + 1e-8)

    def _compute_connectivity_score(self, pred_volume: np.ndarray) -> dict:
        """
        Compute connectivity metrics for the predicted volume.
        
        Returns:
            dict with:
                - n_components: number of connected components
                - largest_component_ratio: fraction of foreground in largest component
                - fragmentation_penalty: 0 (perfect) to 1 (highly fragmented)
        """
        pred_binary = (pred_volume > 0.5).astype(np.int32)
        
        if pred_binary.sum() == 0:
            return {
                "n_components": 0,
                "largest_component_ratio": 1.0,
                "fragmentation_penalty": 0.0
            }
        
        # Find connected components (26-connectivity for 3D)
        labeled, n_components = ndimage.label(pred_binary)
        
        if n_components == 0:
            return {
                "n_components": 0,
                "largest_component_ratio": 1.0,
                "fragmentation_penalty": 0.0
            }
        
        # Find size of each component
        component_sizes = ndimage.sum(pred_binary, labeled, range(1, n_components + 1))
        largest_size = max(component_sizes) if len(component_sizes) > 0 else 0
        total_foreground = pred_binary.sum()
        
        largest_ratio = largest_size / total_foreground if total_foreground > 0 else 1.0
        
        # Fragmentation penalty: more components = more fragmented
        # Also penalize if largest component is small fraction of total
        # Perfect: 1 component containing 100% of foreground -> penalty = 0
        # Worst: many tiny disconnected components -> penalty approaches 1
        fragmentation_penalty = 1.0 - largest_ratio + np.log(n_components) / (np.log(n_components + 1) + 1)
        fragmentation_penalty = np.clip(fragmentation_penalty, 0, 1)
        
        return {
            "n_components": n_components,
            "largest_component_ratio": largest_ratio,
            "fragmentation_penalty": fragmentation_penalty
        }

    def _compute_boundary_consistency(self, pred_volume: np.ndarray) -> float:
        """
        Compute consistency between predictions and neighbor boundaries.
        Higher = better consistency (0 to 1).
        """
        if not self.neighbor_context:
            return 1.0  # No neighbors = perfect consistency (no penalty)
        
        consistency_scores = []
        pred_binary = (pred_volume > 0.5).astype(np.float32)
        
        # Check each boundary
        if 'front' in self.neighbor_context:
            our_front = pred_binary[0, :, :]
            neighbor_front = self.neighbor_context['front']
            consistency_scores.append(1.0 - np.abs(our_front - neighbor_front).mean())
            
        if 'back' in self.neighbor_context:
            our_back = pred_binary[-1, :, :]
            neighbor_back = self.neighbor_context['back']
            consistency_scores.append(1.0 - np.abs(our_back - neighbor_back).mean())
            
        if 'top' in self.neighbor_context:
            our_top = pred_binary[:, 0, :]
            neighbor_top = self.neighbor_context['top']
            consistency_scores.append(1.0 - np.abs(our_top - neighbor_top).mean())
            
        if 'bottom' in self.neighbor_context:
            our_bottom = pred_binary[:, -1, :]
            neighbor_bottom = self.neighbor_context['bottom']
            consistency_scores.append(1.0 - np.abs(our_bottom - neighbor_bottom).mean())
            
        if 'left' in self.neighbor_context:
            our_left = pred_binary[:, :, 0]
            neighbor_left = self.neighbor_context['left']
            consistency_scores.append(1.0 - np.abs(our_left - neighbor_left).mean())
            
        if 'right' in self.neighbor_context:
            our_right = pred_binary[:, :, -1]
            neighbor_right = self.neighbor_context['right']
            consistency_scores.append(1.0 - np.abs(our_right - neighbor_right).mean())
        
        return np.mean(consistency_scores) if consistency_scores else 1.0

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
        dice_reward = self.dice_coef * slice_dice
        
        # Store prediction for episode-level tracking
        self.episode_predictions[slice_idx] = action
        
        # ============================================================
        # Total per-step reward
        # ============================================================
        pixel_rewards = base_rewards + continuity_rewards
        dice_bonus_per_pixel = dice_reward / (self.H * self.W)
        pixel_rewards_with_dice = pixel_rewards + dice_bonus_per_pixel
        
        reward = float(pixel_rewards_with_dice.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(self.volume[slice_idx, :, :].copy())
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        # ============================================================
        # Episode-level rewards (global coherence)
        # ============================================================
        episode_dice = None
        connectivity_info = None
        boundary_consistency = None
        global_coherence_bonus = 0.0
        
        if terminated:
            episode_dice = self._compute_dice(self.episode_predictions, self.episode_ground_truth)
            
            # Connectivity reward
            connectivity_info = self._compute_connectivity_score(self.episode_predictions)
            connectivity_reward = self.connectivity_coef * (1.0 - connectivity_info["fragmentation_penalty"])
            
            # Boundary consistency reward
            boundary_consistency = self._compute_boundary_consistency(self.episode_predictions)
            boundary_reward = self.boundary_coef * boundary_consistency
            
            # Add global coherence bonus to final reward
            global_coherence_bonus = connectivity_reward + boundary_reward
            reward += global_coherence_bonus

        info = {
            "pixel_rewards": pixel_rewards_with_dice.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "dice_reward": dice_reward,
            "slice_dice": slice_dice,
            "slice_index": slice_idx,
            "episode_dice": episode_dice,
            "connectivity_info": connectivity_info,
            "boundary_consistency": boundary_consistency,
            "global_coherence_bonus": global_coherence_bonus,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "slice_pixels": np.zeros((self.H, self.W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
            "neighbor_hints": np.zeros((6, self.H, self.W), dtype=np.float32),
        }
    
    def get_boundary_predictions(self) -> dict:
        """
        Get predictions at boundaries for sharing with neighboring subvolumes.
        Call after episode completion.
        """
        pred = (self.episode_predictions > 0.5).astype(np.float32)
        return {
            "front": pred[0, :, :],      # First slice (D=0)
            "back": pred[-1, :, :],      # Last slice (D=-1)
            "top": pred[:, 0, :],        # Top row (H=0)
            "bottom": pred[:, -1, :],    # Bottom row (H=-1)  
            "left": pred[:, :, 0],       # Left column (W=0)
            "right": pred[:, :, -1],     # Right column (W=-1)
        }


# Aliases for compatibility
PathReconstructionEnvDice = PathReconstructionEnvGlobalCoherence
PathReconstructionEnv = PathReconstructionEnvGlobalCoherence

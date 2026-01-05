"""
Environment with Inter-Subvolume Global Coherence Reward.

Key innovation: Maintains a global prediction buffer and rewards predictions
that align with adjacent subvolumes at their shared boundaries.

This gives the agent awareness of the broader 3D structure beyond its local subvolume.

Boundary alignment:
- Subvolume at (d, h, w) shares faces with 6 neighbors:
  - (d-1, h, w): shares z=0 face with neighbor's z=D-1
  - (d+1, h, w): shares z=D-1 face with neighbor's z=0
  - (d, h-1, w): shares y=0 face with neighbor's y=H-1
  - (d, h+1, w): shares y=H-1 face with neighbor's y=0
  - (d, h, w-1): shares x=0 face with neighbor's x=W-1
  - (d, h, w+1): shares x=W-1 face with neighbor's x=0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage
from typing import Dict, Tuple, Optional


class GlobalPredictionBuffer:
    """
    Manages predictions across all subvolumes for inter-subvolume rewards.
    
    Stores predictions indexed by grid position (d, h, w) and provides
    methods to query adjacent subvolume predictions at boundaries.
    """
    
    def __init__(self, subvol_shape: Tuple[int, int, int] = (100, 64, 64)):
        self.subvol_shape = subvol_shape  # (D, H, W) of each subvolume
        self.predictions: Dict[Tuple[int, int, int], np.ndarray] = {}
        
    def store_prediction(self, position: Tuple[int, int, int], prediction: np.ndarray):
        """Store the prediction for a subvolume at given grid position."""
        self.predictions[position] = prediction.copy()
    
    def get_prediction(self, position: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Get prediction for a subvolume, or None if not yet processed."""
        return self.predictions.get(position, None)
    
    def clear(self):
        """Clear all predictions (call at start of each epoch)."""
        self.predictions.clear()
    
    def get_adjacent_boundaries(self, position: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        """
        Get boundary slices from adjacent subvolumes that have been processed.
        
        Returns dict with keys like 'z_minus', 'z_plus', 'y_minus', etc.
        Each value is the boundary slice from the adjacent subvolume.
        """
        d, h, w = position
        D, H, W = self.subvol_shape
        boundaries = {}
        
        # Check z-minus neighbor (d-1, h, w) - their z=D-1 face is our z=0 boundary
        if (d-1, h, w) in self.predictions:
            neighbor_pred = self.predictions[(d-1, h, w)]
            boundaries['z_minus'] = neighbor_pred[-1, :, :]  # Last slice of neighbor
        
        # Check z-plus neighbor (d+1, h, w) - their z=0 face is our z=D-1 boundary
        if (d+1, h, w) in self.predictions:
            neighbor_pred = self.predictions[(d+1, h, w)]
            boundaries['z_plus'] = neighbor_pred[0, :, :]  # First slice of neighbor
        
        # Check y-minus neighbor (d, h-1, w) - their y=H-1 edge is our y=0 boundary
        if (d, h-1, w) in self.predictions:
            neighbor_pred = self.predictions[(d, h-1, w)]
            boundaries['y_minus'] = neighbor_pred[:, -1, :]  # Last row of neighbor
        
        # Check y-plus neighbor (d, h+1, w) - their y=0 edge is our y=H-1 boundary
        if (d, h+1, w) in self.predictions:
            neighbor_pred = self.predictions[(d, h+1, w)]
            boundaries['y_plus'] = neighbor_pred[:, 0, :]  # First row of neighbor
        
        # Check x-minus neighbor (d, h, w-1) - their x=W-1 edge is our x=0 boundary
        if (d, h, w-1) in self.predictions:
            neighbor_pred = self.predictions[(d, h, w-1)]
            boundaries['x_minus'] = neighbor_pred[:, :, -1]  # Last column of neighbor
        
        # Check x-plus neighbor (d, h, w+1) - their x=0 edge is our x=W-1 boundary
        if (d, h, w+1) in self.predictions:
            neighbor_pred = self.predictions[(d, h, w+1)]
            boundaries['x_plus'] = neighbor_pred[:, :, 0]  # First column of neighbor
        
        return boundaries


class PathReconstructionEnvGlobalCoherence(gym.Env):
    """
    Slice-wise environment with inter-subvolume global coherence rewards.
    
    In addition to local rewards (base, continuity, DICE), this environment
    receives a GlobalPredictionBuffer and rewards predictions that align
    with adjacent subvolumes at their shared boundaries.
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 volume: np.ndarray, 
                 mask: np.ndarray,
                 grid_position: Tuple[int, int, int] = (0, 0, 0),
                 global_buffer: GlobalPredictionBuffer = None,
                 # Reward coefficients
                 continuity_coef: float = 0.1, 
                 continuity_decay_factor: float = 0.7,
                 dice_coef: float = 1.0,
                 boundary_coef: float = 0.2,  # NEW: inter-subvolume boundary alignment
                 connectivity_coef: float = 0.05,
                 smoothness_coef: float = 0.02,
                 # Other params
                 history_len: int = 5, 
                 start_from_bottom: bool = True):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)
        
        self.D, self.H, self.W = self.mask.shape
        self.grid_position = grid_position
        self.global_buffer = global_buffer
        
        # Reward coefficients
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.dice_coef = float(dice_coef)
        self.boundary_coef = float(boundary_coef)
        self.connectivity_coef = float(connectivity_coef)
        self.smoothness_coef = float(smoothness_coef)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)

        self.action_space = spaces.MultiBinary(self.H * self.W)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(self.H, self.W), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "prev_slices": spaces.Box(0.0, 1.0, shape=(self.history_len, self.H, self.W), dtype=np.float32),
            "slice_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # NEW: boundary context from adjacent subvolumes
            "boundary_context": spaces.Box(0.0, 1.0, shape=(self.H, self.W), dtype=np.float32),
        })

        self._slice_order = None
        self.current_slice_idx = None
        self.prev_preds_buffer = None
        self.prev_slices_buffer = None
        
        # Track predictions for this episode
        self.episode_predictions = None
        self.episode_ground_truth = None
        
        # Get adjacent boundary info at init (doesn't change during episode)
        self.adjacent_boundaries = {}
        if self.global_buffer is not None:
            self.adjacent_boundaries = self.global_buffer.get_adjacent_boundaries(grid_position)

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
        
        # Refresh adjacent boundaries (in case buffer was updated)
        if self.global_buffer is not None:
            self.adjacent_boundaries = self.global_buffer.get_adjacent_boundaries(self.grid_position)
        
        return self._get_obs(), {}

    def _get_boundary_context(self, slice_idx: int) -> np.ndarray:
        """
        Get context from adjacent subvolumes relevant to current slice.
        
        For slices at boundaries (first/last), include predictions from neighbors.
        For interior slices, blend information from y/x neighbors.
        """
        context = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Z-boundary context
        if slice_idx == 0 and 'z_minus' in self.adjacent_boundaries:
            # First slice of this subvolume - use last slice of z-minus neighbor
            context += self.adjacent_boundaries['z_minus']
        elif slice_idx == self.D - 1 and 'z_plus' in self.adjacent_boundaries:
            # Last slice - use first slice of z-plus neighbor
            context += self.adjacent_boundaries['z_plus']
        
        # Y-boundary context (edge rows)
        if 'y_minus' in self.adjacent_boundaries:
            # First row should align with last row of y-minus neighbor
            neighbor_slice = self.adjacent_boundaries['y_minus']
            if slice_idx < neighbor_slice.shape[0]:
                context[0, :] += neighbor_slice[slice_idx, :]
        if 'y_plus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['y_plus']
            if slice_idx < neighbor_slice.shape[0]:
                context[-1, :] += neighbor_slice[slice_idx, :]
        
        # X-boundary context (edge columns)
        if 'x_minus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['x_minus']
            if slice_idx < neighbor_slice.shape[0]:
                context[:, 0] += neighbor_slice[slice_idx, :]
        if 'x_plus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['x_plus']
            if slice_idx < neighbor_slice.shape[0]:
                context[:, -1] += neighbor_slice[slice_idx, :]
        
        return np.clip(context, 0, 1)

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        boundary_context = self._get_boundary_context(slice_idx)
        
        return {
            "slice_pixels": self.volume[slice_idx, :, :],
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([(slice_idx + 1) / self.D], dtype=np.float32),
            "boundary_context": boundary_context,
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

    def _compute_boundary_alignment_reward(self, action: np.ndarray, slice_idx: int) -> float:
        """
        Compute reward for aligning with adjacent subvolume predictions at boundaries.
        
        This is the KEY inter-subvolume reward that encourages global coherence.
        """
        if not self.adjacent_boundaries:
            return 0.0
        
        action_bin = (action > 0.5).astype(np.float32)
        total_reward = 0.0
        n_boundaries = 0
        
        # Z-boundary alignment
        if slice_idx == 0 and 'z_minus' in self.adjacent_boundaries:
            neighbor_pred = self.adjacent_boundaries['z_minus']
            # Reward overlap between current prediction and neighbor's boundary
            overlap = (action_bin * neighbor_pred).sum()
            union = np.maximum(action_bin, neighbor_pred).sum()
            if union > 0:
                iou = overlap / (union + 1e-8)
                total_reward += iou
                n_boundaries += 1
                
        if slice_idx == self.D - 1 and 'z_plus' in self.adjacent_boundaries:
            neighbor_pred = self.adjacent_boundaries['z_plus']
            overlap = (action_bin * neighbor_pred).sum()
            union = np.maximum(action_bin, neighbor_pred).sum()
            if union > 0:
                iou = overlap / (union + 1e-8)
                total_reward += iou
                n_boundaries += 1
        
        # Y-boundary alignment (edge rows)
        if 'y_minus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['y_minus']
            if slice_idx < neighbor_slice.shape[0]:
                # Compare first row of action with last row of neighbor
                row_overlap = (action_bin[0, :] * neighbor_slice[slice_idx, :]).sum()
                row_union = np.maximum(action_bin[0, :], neighbor_slice[slice_idx, :]).sum()
                if row_union > 0:
                    total_reward += 0.5 * row_overlap / (row_union + 1e-8)
                    n_boundaries += 0.5
                    
        if 'y_plus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['y_plus']
            if slice_idx < neighbor_slice.shape[0]:
                row_overlap = (action_bin[-1, :] * neighbor_slice[slice_idx, :]).sum()
                row_union = np.maximum(action_bin[-1, :], neighbor_slice[slice_idx, :]).sum()
                if row_union > 0:
                    total_reward += 0.5 * row_overlap / (row_union + 1e-8)
                    n_boundaries += 0.5
        
        # X-boundary alignment (edge columns)
        if 'x_minus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['x_minus']
            if slice_idx < neighbor_slice.shape[0]:
                col_overlap = (action_bin[:, 0] * neighbor_slice[slice_idx, :]).sum()
                col_union = np.maximum(action_bin[:, 0], neighbor_slice[slice_idx, :]).sum()
                if col_union > 0:
                    total_reward += 0.5 * col_overlap / (col_union + 1e-8)
                    n_boundaries += 0.5
                    
        if 'x_plus' in self.adjacent_boundaries:
            neighbor_slice = self.adjacent_boundaries['x_plus']
            if slice_idx < neighbor_slice.shape[0]:
                col_overlap = (action_bin[:, -1] * neighbor_slice[slice_idx, :]).sum()
                col_union = np.maximum(action_bin[:, -1], neighbor_slice[slice_idx, :]).sum()
                if col_union > 0:
                    total_reward += 0.5 * col_overlap / (col_union + 1e-8)
                    n_boundaries += 0.5
        
        # Average across all boundaries
        if n_boundaries > 0:
            return total_reward / n_boundaries
        return 0.0

    def _compute_connectivity_reward(self, action: np.ndarray, prev_pred: np.ndarray) -> float:
        """Reward for maintaining connected structure."""
        action_bin = (action > 0.5).astype(np.int32)
        prev_bin = (prev_pred > 0.5).astype(np.int32)
        
        if action_bin.sum() == 0:
            return 0.0
        
        labeled, n_components = ndimage.label(action_bin)
        if n_components == 0:
            return 0.0
        
        component_penalty = -0.1 * (n_components - 1)
        
        if prev_bin.sum() > 0:
            overlap = (action_bin * prev_bin).sum()
            overlap_ratio = overlap / max(prev_bin.sum(), 1)
            continuity_bonus = 0.5 * overlap_ratio
        else:
            continuity_bonus = 0.0
        
        return component_penalty + continuity_bonus

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.H, self.W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        gt = self.mask[slice_idx, :, :]
        current_slice = self.volume[slice_idx, :, :]

        # ============================================================
        # 1. Base rewards: per-pixel accuracy
        # ============================================================
        base_rewards = -np.abs(action - gt)

        # ============================================================
        # 2. Continuity rewards (within subvolume)
        # ============================================================
        continuity_rewards = np.zeros((self.H, self.W), dtype=np.float32)
        for i, prev_pred in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (self.continuity_decay_factor ** i)
            continuity_rewards += -weight * np.abs(action - prev_pred)
        continuity_rewards *= self.continuity_coef

        # ============================================================
        # 3. DICE reward: slice-level segmentation quality
        # ============================================================
        slice_dice = self._compute_dice(action, gt)
        dice_reward = self.dice_coef * slice_dice

        # ============================================================
        # 4. Boundary alignment reward (INTER-SUBVOLUME)
        # ============================================================
        boundary_reward = self.boundary_coef * self._compute_boundary_alignment_reward(action, slice_idx)

        # ============================================================
        # 5. Connectivity reward (within subvolume)
        # ============================================================
        prev_pred = list(self.prev_preds_buffer)[-1] if self.prev_preds_buffer else np.zeros_like(action)
        connectivity_reward = self.connectivity_coef * self._compute_connectivity_reward(action, prev_pred)

        # Store prediction
        self.episode_predictions[slice_idx] = action

        # ============================================================
        # Total reward
        # ============================================================
        pixel_rewards = base_rewards + continuity_rewards
        scalar_bonus = (dice_reward + boundary_reward + connectivity_reward) / (self.H * self.W)
        pixel_rewards_with_bonus = pixel_rewards + scalar_bonus
        
        reward = float(pixel_rewards_with_bonus.sum())

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(current_slice.copy())
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        # On termination, store predictions in global buffer
        episode_dice = None
        if terminated:
            episode_dice = self._compute_dice(self.episode_predictions, self.episode_ground_truth)
            if self.global_buffer is not None:
                self.global_buffer.store_prediction(self.grid_position, self.episode_predictions)

        info = {
            "pixel_rewards": pixel_rewards_with_bonus.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "dice_reward": dice_reward,
            "boundary_reward": boundary_reward,
            "connectivity_reward": connectivity_reward,
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
            "boundary_context": np.zeros((self.H, self.W), dtype=np.float32),
        }


# Aliases for compatibility
PathReconstructionEnv = PathReconstructionEnvGlobalCoherence
PathReconstructionEnvDice = PathReconstructionEnvGlobalCoherence

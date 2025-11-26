import numpy as np
from typing import Tuple, Optional
from collections import deque

class SliceReconstructionEnv:
    """
    Environment for slice-by-slice 3D volume reconstruction.
    Each slice is treated as a 2D image reconstruction task.
    Noise is added once during initialization.
    
    Now returns multi-channel observations:
    - Channel 0: Current noisy slice
    - Channels 1+: Previous slice predictions (up to history_len)
    
    Reward components per pixel (summed across the slice):
    - Base accuracy:     -|action - gt|
    - Slice continuity:  -continuity_coef * |action - prev_slices_action|
        (Linking to previous slices with optional decay)
    """
    
    def __init__(self, 
                 target_volume: np.ndarray, 
                 noise_level: float = 0.0,
                 continuity_coef: float = 0.1,
                 continuity_decay_factor: float = 0.7,
                 history_len: int = 3,
                 start_from_bottom: bool = True):
        """
        Args:
            target_volume: 3D numpy array (D, H, W) with binary values
            noise_level: Amount of noise to add to input slices
            continuity_coef: Weight for slice continuity reward
            continuity_decay_factor: Decay factor for previous slices (0.5-0.9)
            history_len: Number of previous slices to include in observation
            start_from_bottom: If True, process slices from bottom to top
        """
        self.target_volume = target_volume.astype(np.float32)
        self.depth, self.height, self.width = target_volume.shape
        self.noise_level = noise_level
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)
        
        # Add noise once to entire volume
        self.noisy_volume = self._add_noise_to_volume(self.target_volume)
        
        self._slice_order = None
        self.current_slice_idx = 0
        self.current_state = None
        self.target_slice = None
        self.prev_preds_buffer = None
        
    def _add_noise_to_volume(self, volume: np.ndarray) -> np.ndarray:
        """Add noise to entire volume once"""
        noisy = volume.copy().astype(float)
        
        # Add gaussian noise
        noise = np.random.randn(*volume.shape) * self.noise_level
        noisy = noisy + noise
        
        # Add salt and pepper noise
        salt_pepper = np.random.rand(*volume.shape)
        noisy[salt_pepper < 0.05] = 0
        noisy[salt_pepper > 0.95] = 1
        
        return np.clip(noisy, 0, 1).astype(np.float32)
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to first slice.
        Returns multi-channel observation: (1 + history_len, H, W)
        """
        # Set slice processing order
        self._slice_order = (np.arange(self.depth-1, -1, -1) if self.start_from_bottom 
                            else np.arange(0, self.depth))
        self.current_slice_idx = 0
        
        # Initialize history buffer with zeros
        self.prev_preds_buffer = deque(
            [np.zeros((self.height, self.width), dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
            
        # Get target slice and noisy slice
        slice_idx = self._slice_order[self.current_slice_idx]
        self.target_slice = self.target_volume[slice_idx].copy()
        
        # Build multi-channel observation
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Build multi-channel observation.
        Returns: (1 + history_len, H, W) array
        - Channel 0: Current noisy slice
        - Channels 1+: Previous predictions (most recent first)
        """
        slice_idx = self._slice_order[self.current_slice_idx]
        current_noisy = self.noisy_volume[slice_idx].copy()
        
        # Stack: [current_noisy, prev_pred_1, prev_pred_2, ...]
        channels = [current_noisy] + list(reversed(list(self.prev_preds_buffer)))
        observation = np.stack(channels, axis=0)  # (1 + history_len, H, W)
        
        return observation.astype(np.float32)
    
    def step(self, action_map: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action map (reconstructed slice) and compute reward.
        Move to next slice.
        
        Args:
            action_map: 2D array (H, W) with values in [0, 1]
            
        Returns:
            next_observation (multi-channel), reward, done, info
        """
        action_map = np.array(action_map, dtype=np.float32).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        gt = self.target_slice  # ground truth
        
        # Base pixel-wise accuracy term against ground-truth
        base_rewards = -np.abs(action_map - gt)
        
        # Exponential Decay Continuity Strategy
        continuity_rewards = np.zeros((self.height, self.width), dtype=np.float32)
        decay_factor = self.continuity_decay_factor
        for i, prev_slice in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            continuity_rewards += -weight * np.abs(action_map - prev_slice)
        continuity_rewards *= self.continuity_coef
        
        # Total per-pixel rewards
        pixel_rewards = base_rewards + continuity_rewards
        
        # Sum across all pixels for total reward
        reward = float(pixel_rewards.sum())
        
        # Compute additional metrics for monitoring
        reconstructed_binary = (action_map > 0.5).astype(np.uint8)
        gt_binary = (gt > 0.5).astype(np.uint8)
        
        correct_pixels = (reconstructed_binary == gt_binary).sum()
        total_pixels = self.height * self.width
        accuracy = correct_pixels / total_pixels
        
        # Compute IoU (Intersection over Union)
        intersection = (reconstructed_binary * gt_binary).sum()
        union = (reconstructed_binary + gt_binary).clip(0, 1).sum()
        iou = intersection / (union + 1e-7)
        
        # Compute F1 score
        tp = (reconstructed_binary * gt_binary).sum()
        fp = (reconstructed_binary * (1 - gt_binary)).sum()
        fn = ((1 - reconstructed_binary) * gt_binary).sum()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        info = {
            'accuracy': accuracy,
            'iou': iou,
            'f1_score': f1,
            'slice_idx': slice_idx,
            'pixel_rewards': pixel_rewards.astype(np.float32),
            'base_rewards': base_rewards.astype(np.float32),
            'continuity_rewards': continuity_rewards.astype(np.float32),
        }
        
        # Update history buffer
        self.prev_preds_buffer.append(action_map.copy())
        
        # Move to next slice
        self.current_slice_idx += 1
        done = self.current_slice_idx >= self.depth
        
        # Get next observation if not done
        if not done:
            slice_idx = self._slice_order[self.current_slice_idx]
            self.target_slice = self.target_volume[slice_idx].copy()
        
        # Return multi-channel observation
        next_obs = self._get_observation() if not done else self._terminal_observation()
        
        return next_obs, reward, done, info
    
    def _terminal_observation(self) -> np.ndarray:
        """Return a dummy observation for terminal state"""
        return np.zeros((1 + self.history_len, self.height, self.width), dtype=np.float32)
    
    def get_target_slice(self) -> np.ndarray:
        """Get current target slice"""
        return self.target_slice.copy()
    
    def get_noisy_volume(self) -> np.ndarray:
        """Get the entire noisy volume"""
        return self.noisy_volume.copy()
    
    def get_slice_order(self) -> np.ndarray:
        """Get the order in which slices are processed"""
        return self._slice_order.copy() if self._slice_order is not None else None
    
    def get_num_channels(self) -> int:
        """Get number of input channels (1 current + history_len previous)"""
        return 1 + self.history_len
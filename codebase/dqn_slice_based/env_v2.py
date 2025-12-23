"""
Enhanced environment for training on large real CT volumes.
Key improvements:
- Class-weighted rewards (handles imbalanced foreground/background)
- Boundary-aware rewards
- Random slice order for data augmentation
- Support for patch-based training
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage


class PathReconstructionEnvV2(gym.Env):
    """
    Enhanced slice-wise environment for large real volumes.
    
    Key improvements:
    - Class-weighted rewards for imbalanced masks
    - Boundary-aware reward component
    - Random slice ordering for augmentation
    - Optional intensity augmentation
    """
    metadata = {"render_modes": []}

    def __init__(
        self, 
        volume, 
        mask, 
        continuity_coef=0.1, 
        continuity_decay_factor=0.7,
        history_len=3, 
        start_from_bottom=True,
        random_slice_order=False,
        # New parameters for real data
        use_class_weights=True,
        foreground_weight=2.0,  # Weight for foreground (vessel) pixels (reduced to prevent collapse)
        boundary_coef=0.1,      # Weight for boundary accuracy
        use_intensity_aug=False,
        intensity_aug_range=(0.9, 1.1),
        # Patch-based training
        patch_size=None,  # If set, train on random patches
    ):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)
        
        self.D, self.H, self.W = self.mask.shape
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)
        self.random_slice_order = bool(random_slice_order)
        
        # Class weighting
        self.use_class_weights = use_class_weights
        self.foreground_weight = foreground_weight
        self.boundary_coef = boundary_coef
        
        # Intensity augmentation
        self.use_intensity_aug = use_intensity_aug
        self.intensity_aug_range = intensity_aug_range
        
        # Patch-based training
        self.patch_size = patch_size
        if patch_size is not None:
            self.effective_H = patch_size
            self.effective_W = patch_size
        else:
            self.effective_H = self.H
            self.effective_W = self.W
        
        # Precompute boundaries for all slices
        self._precompute_boundaries()
        
        # Precompute class weights per slice
        self._precompute_class_weights()
        
        self.action_space = spaces.MultiBinary(self.effective_H * self.effective_W)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(self.effective_H, self.effective_W), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.effective_H, self.effective_W), dtype=np.float32),
            "prev_slices": spaces.Box(0.0, 1.0, shape=(self.history_len, self.effective_H, self.effective_W), dtype=np.float32),
            "slice_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._slice_order = None
        self.current_slice_idx = None
        self.prev_preds_buffer = None
        self.prev_slices_buffer = None
        self.current_patch_y = 0
        self.current_patch_x = 0
        self.intensity_scale = 1.0
    
    def _precompute_boundaries(self):
        """Precompute boundary masks for all slices."""
        self.boundaries = np.zeros_like(self.mask)
        for i in range(self.D):
            # Sobel edge detection on mask
            sx = ndimage.sobel(self.mask[i], axis=0)
            sy = ndimage.sobel(self.mask[i], axis=1)
            self.boundaries[i] = np.sqrt(sx**2 + sy**2) > 0.5
    
    def _precompute_class_weights(self):
        """Precompute class weights based on foreground ratio per slice."""
        self.class_weights = np.zeros((self.D, self.H, self.W), dtype=np.float32)
        for i in range(self.D):
            fg_ratio = self.mask[i].mean()
            if fg_ratio > 0 and fg_ratio < 1:
                # Inverse frequency weighting
                bg_weight = 1.0
                fg_weight = min(self.foreground_weight, (1 - fg_ratio) / fg_ratio)
                self.class_weights[i] = np.where(
                    self.mask[i] > 0.5,
                    fg_weight,
                    bg_weight
                )
            else:
                self.class_weights[i] = 1.0
    
    def _get_patch_coords(self):
        """Get current patch coordinates."""
        if self.patch_size is None:
            return 0, 0, self.H, self.W
        return (self.current_patch_y, 
                self.current_patch_x,
                self.current_patch_y + self.patch_size,
                self.current_patch_x + self.patch_size)
    
    def _extract_patch(self, arr):
        """Extract patch from 2D array."""
        if self.patch_size is None:
            return arr
        y1, x1, y2, x2 = self._get_patch_coords()
        return arr[y1:y2, x1:x2]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Slice ordering
        if self.random_slice_order:
            self._slice_order = np.random.permutation(self.D)
        else:
            self._slice_order = np.arange(0, self.D) if self.start_from_bottom else np.arange(self.D-1, -1, -1)
        
        self.current_slice_idx = 0
        
        # Random patch position for this episode
        if self.patch_size is not None:
            self.current_patch_y = np.random.randint(0, max(1, self.H - self.patch_size + 1))
            self.current_patch_x = np.random.randint(0, max(1, self.W - self.patch_size + 1))
        
        # Intensity augmentation
        if self.use_intensity_aug:
            self.intensity_scale = np.random.uniform(*self.intensity_aug_range)
        else:
            self.intensity_scale = 1.0
        
        self.prev_preds_buffer = deque(
            [np.zeros((self.effective_H, self.effective_W), dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.prev_slices_buffer = deque(
            [np.zeros((self.effective_H, self.effective_W), dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        
        return self._get_obs(), {}

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        slice_pixels = self._extract_patch(self.volume[slice_idx])
        
        # Apply intensity augmentation
        slice_pixels = np.clip(slice_pixels * self.intensity_scale, 0, 1)
        
        return {
            "slice_pixels": slice_pixels.astype(np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([(self.current_slice_idx + 1) / self.D], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.effective_H, self.effective_W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        
        gt = self._extract_patch(self.mask[slice_idx])
        weights = self._extract_patch(self.class_weights[slice_idx])
        boundaries = self._extract_patch(self.boundaries[slice_idx])

        # 1. Base accuracy reward with balanced positive/negative
        # Correct predictions get +reward, incorrect get -reward
        error = np.abs(action - gt)
        correct = 1.0 - error  # 1 when perfect, 0 when wrong
        
        if self.use_class_weights:
            # Reward correct predictions, penalize errors
            base_rewards = (correct - error) * weights
        else:
            base_rewards = correct - error
        
        # Normalize by number of pixels to handle different slice sizes
        num_pixels = self.effective_H * self.effective_W
        base_rewards = base_rewards / np.sqrt(num_pixels)

        # 2. Continuity reward (normalized)
        continuity_rewards = np.zeros((self.effective_H, self.effective_W), dtype=np.float32)
        decay_factor = self.continuity_decay_factor
        for i, prev_slice in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            cont_error = np.abs(action - prev_slice)
            cont_correct = 1.0 - cont_error
            continuity_rewards += weight * (cont_correct - cont_error)
        continuity_rewards *= self.continuity_coef / np.sqrt(num_pixels)
        
        # 3. Boundary reward (normalized)
        boundary_rewards = np.zeros((self.effective_H, self.effective_W), dtype=np.float32)
        if self.boundary_coef > 0:
            boundary_mask = boundaries.astype(np.float32)
            # Extra reward/penalty for boundary accuracy
            boundary_error = np.abs(action - gt) * boundary_mask
            boundary_correct = (1.0 - error) * boundary_mask
            boundary_rewards = self.boundary_coef * (boundary_correct - boundary_error) / np.sqrt(num_pixels)

        # Total reward
        pixel_rewards = base_rewards + continuity_rewards + boundary_rewards
        reward = float(pixel_rewards.sum())

        # Update history
        slice_pixels = self._extract_patch(self.volume[slice_idx])
        slice_pixels = np.clip(slice_pixels * self.intensity_scale, 0, 1)
        
        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(slice_pixels.astype(np.float32))
        
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.D

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "boundary_rewards": boundary_rewards.astype(np.float32),
            "slice_index": slice_idx,
            "patch_coords": self._get_patch_coords() if self.patch_size else None,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "slice_pixels": np.zeros((self.effective_H, self.effective_W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }


class SliceSamplerEnv(gym.Env):
    """
    Environment that samples random contiguous slice sequences from a large volume.
    This allows training on subsets of slices per episode, increasing sample diversity.
    Preferentially samples regions with foreground to avoid training on empty slices.
    """
    metadata = {"render_modes": []}

    def __init__(
        self, 
        volume, 
        mask, 
        slices_per_episode=64,  # Number of slices per episode
        continuity_coef=0.1, 
        continuity_decay_factor=0.7,
        history_len=3,
        use_class_weights=True,
        foreground_weight=2.0,
        boundary_coef=0.1,
        random_direction=True,  # Randomly go up or down
        prefer_foreground=True,  # Prefer sampling regions with foreground
        foreground_threshold=0.001,  # Min foreground ratio to consider
    ):
        super().__init__()
        assert volume.shape == mask.shape, "Volume and mask must have same shape"
        
        self.volume = volume.astype(np.float32) / 255.0 if volume.max() > 1 else volume.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)
        
        self.D, self.H, self.W = self.mask.shape
        self.slices_per_episode = min(slices_per_episode, self.D)
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.random_direction = random_direction
        self.prefer_foreground = prefer_foreground
        self.foreground_threshold = foreground_threshold
        
        # Class weighting
        self.use_class_weights = use_class_weights
        self.foreground_weight = foreground_weight
        self.boundary_coef = boundary_coef
        
        self._precompute_boundaries()
        self._precompute_class_weights()
        self._find_foreground_regions()
        
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
        self.episode_start = 0
        self.go_up = True
    
    def _precompute_boundaries(self):
        self.boundaries = np.zeros_like(self.mask)
        for i in range(self.D):
            sx = ndimage.sobel(self.mask[i], axis=0)
            sy = ndimage.sobel(self.mask[i], axis=1)
            self.boundaries[i] = np.sqrt(sx**2 + sy**2) > 0.5
    
    def _precompute_class_weights(self):
        self.class_weights = np.zeros((self.D, self.H, self.W), dtype=np.float32)
        for i in range(self.D):
            fg_ratio = self.mask[i].mean()
            if fg_ratio > 0 and fg_ratio < 1:
                bg_weight = 1.0
                fg_weight = min(self.foreground_weight, (1 - fg_ratio) / fg_ratio)
                self.class_weights[i] = np.where(self.mask[i] > 0.5, fg_weight, bg_weight)
            else:
                self.class_weights[i] = 1.0
    
    def _find_foreground_regions(self):
        """Find slice ranges that contain sufficient foreground for better sampling."""
        self.foreground_slices = []
        for i in range(self.D):
            fg_ratio = self.mask[i].mean()
            if fg_ratio >= self.foreground_threshold:
                self.foreground_slices.append(i)
        
        if len(self.foreground_slices) == 0:
            # Fallback: use all slices
            self.foreground_slices = list(range(self.D))
        
        print(f"Found {len(self.foreground_slices)}/{self.D} slices with foreground >= {self.foreground_threshold}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Smart starting position - prefer regions with foreground
        if self.prefer_foreground and len(self.foreground_slices) >= self.slices_per_episode:
            # Sample from foreground regions
            # Pick a random foreground slice as anchor
            anchor = np.random.choice(self.foreground_slices)
            # Find valid start that includes this anchor
            min_start = max(0, anchor - self.slices_per_episode + 1)
            max_start = min(self.D - self.slices_per_episode, anchor)
            self.episode_start = np.random.randint(min_start, max_start + 1)
        else:
            # Random starting position
            max_start = self.D - self.slices_per_episode
            self.episode_start = np.random.randint(0, max(1, max_start + 1))
        
        # Random direction
        self.go_up = np.random.choice([True, False]) if self.random_direction else True
        
        if self.go_up:
            self._slice_order = np.arange(self.episode_start, self.episode_start + self.slices_per_episode)
        else:
            end_idx = self.episode_start + self.slices_per_episode - 1
            self._slice_order = np.arange(end_idx, self.episode_start - 1, -1)
        
        self.current_slice_idx = 0
        self.prev_preds_buffer = deque(
            [np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.prev_slices_buffer = deque(
            [np.zeros((self.H, self.W), dtype=np.float32) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        
        return self._get_obs(), {}

    def _get_obs(self):
        slice_idx = self._slice_order[self.current_slice_idx]
        return {
            "slice_pixels": self.volume[slice_idx].astype(np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([(slice_idx + 1) / self.D], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(self.H, self.W).clip(0, 1)
        slice_idx = self._slice_order[self.current_slice_idx]
        
        gt = self.mask[slice_idx]
        weights = self.class_weights[slice_idx]
        boundaries = self.boundaries[slice_idx]

        # Normalized rewards
        num_pixels = self.H * self.W
        error = np.abs(action - gt)
        correct = 1.0 - error
        
        # Weighted base reward (balanced positive/negative)
        if self.use_class_weights:
            base_rewards = (correct - error) * weights
        else:
            base_rewards = correct - error
        base_rewards = base_rewards / np.sqrt(num_pixels)

        # Continuity (normalized)
        continuity_rewards = np.zeros((self.H, self.W), dtype=np.float32)
        for i, prev_slice in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (self.continuity_decay_factor ** i)
            cont_error = np.abs(action - prev_slice)
            cont_correct = 1.0 - cont_error
            continuity_rewards += weight * (cont_correct - cont_error)
        continuity_rewards *= self.continuity_coef / np.sqrt(num_pixels)
        
        # Boundary reward (normalized)
        boundary_mask = boundaries.astype(np.float32)
        boundary_error = error * boundary_mask
        boundary_correct = correct * boundary_mask
        boundary_rewards = self.boundary_coef * (boundary_correct - boundary_error) / np.sqrt(num_pixels)

        pixel_rewards = base_rewards + continuity_rewards + boundary_rewards
        reward = float(pixel_rewards.sum())

        self.prev_preds_buffer.append(action.copy())
        self.prev_slices_buffer.append(self.volume[slice_idx].copy())
        
        self.current_slice_idx += 1
        terminated = self.current_slice_idx >= self.slices_per_episode

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "boundary_rewards": boundary_rewards.astype(np.float32),
            "slice_index": slice_idx,
        }
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "slice_pixels": np.zeros((self.H, self.W), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_slices": np.array(self.prev_slices_buffer, dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }

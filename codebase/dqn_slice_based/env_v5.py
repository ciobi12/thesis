"""
Environment v5: Shape-Constrained Root Segmentation

Key fixes from v4:
1. SHAPE CONSTRAINTS: Roots are thin elongated structures, not blobs
   - Penalize predictions that are too "wide" or "blobby"
   - Use skeleton-based thinness metric
   - Aspect ratio of connected components

2. MASKED INTENSITY: Only use intensity in ground truth ROI
   - Prevents exploiting soil particles with similar intensity
   - Dilate GT mask and only compute intensity reward inside

3. FALSE POSITIVE PENALTY: Explicit penalty for predicting too much
   - Model was predicting huge areas to game continuity

4. CURRICULUM: Start with accuracy-only, gradually add continuity
   - Let model first learn what roots look like
   - Then add continuity to refine predictions

5. LOCAL CONTRAST: Roots have edges, not just absolute intensity
   - Use gradient magnitude at predicted boundaries
   - Roots should have clear boundaries with soil

6. REDUCED CONTINUITY INITIALLY: Accuracy must come first
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage
from scipy.ndimage import label as connected_components
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize, thin


class ShapeMetrics:
    """
    Compute shape-based metrics to ensure predictions are thin/elongated.
    Roots are tubular structures, not blobs.
    """
    
    @staticmethod
    def thinness_ratio(pred: np.ndarray, min_area: int = 10) -> float:
        """
        Compute thinness ratio: skeleton_length / area.
        Higher = thinner, more root-like.
        Roots should have high thinness (skeleton ~= area for very thin structures).
        """
        pred_bin = (pred > 0.5).astype(np.uint8)
        
        if pred_bin.sum() < min_area:
            return 0.5  # Not enough to judge
        
        # Skeletonize
        try:
            skel = skeletonize(pred_bin)
            skel_length = skel.sum()
            area = pred_bin.sum()
            
            # Thinness: skeleton_length / sqrt(area)
            # For a line: skel = area, so ratio = sqrt(area)
            # For a circle: skel << area
            # Normalize to [0, 1] range
            thinness = skel_length / (np.sqrt(area) + 1e-6)
            return float(np.clip(thinness / 5.0, 0, 1))  # 5.0 is typical for thin roots
        except:
            return 0.5
    
    @staticmethod
    def avg_width(pred: np.ndarray, min_area: int = 10) -> float:
        """
        Compute average width of predicted structures.
        Uses distance transform from skeleton.
        """
        pred_bin = (pred > 0.5).astype(np.uint8)
        
        if pred_bin.sum() < min_area:
            return 0.0
        
        # Distance transform
        dist = distance_transform_edt(pred_bin)
        
        # Average distance (width ~ 2 * avg_distance)
        avg_dist = dist[pred_bin > 0].mean() if pred_bin.any() else 0
        
        return float(avg_dist)
    
    @staticmethod
    def compactness_penalty(pred: np.ndarray, max_avg_width: float = 5.0) -> float:
        """
        Penalize predictions that are too wide/blobby.
        Returns 1.0 if thin enough, lower if too wide.
        """
        avg_w = ShapeMetrics.avg_width(pred)
        
        if avg_w <= max_avg_width:
            return 1.0
        else:
            # Penalty increases with width
            return float(np.exp(-(avg_w - max_avg_width) / 5.0))
    
    @staticmethod
    def connected_component_stats(pred: np.ndarray, min_size: int = 5):
        """
        Analyze connected components to filter out noise and blobs.
        Returns ratio of "good" (elongated) components to total.
        """
        pred_bin = (pred > 0.5).astype(np.int32)
        labeled, n_cc = connected_components(pred_bin)
        
        if n_cc == 0:
            return 1.0, 0
        
        good_pixels = 0
        total_pixels = 0
        
        for i in range(1, n_cc + 1):
            cc_mask = (labeled == i)
            size = cc_mask.sum()
            
            if size < min_size:
                continue
            
            total_pixels += size
            
            # Check if elongated using bounding box aspect ratio
            rows = np.any(cc_mask, axis=1)
            cols = np.any(cc_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            height = rmax - rmin + 1
            width = cmax - cmin + 1
            
            # Fill ratio: how much of bbox is filled
            fill_ratio = size / (height * width + 1e-6)
            
            # Aspect ratio: elongated structures have high aspect ratio
            aspect = max(height, width) / (min(height, width) + 1e-6)
            
            # Good if: elongated (high aspect) OR thin (low fill)
            if aspect >= 2.0 or fill_ratio < 0.5:
                good_pixels += size
        
        if total_pixels == 0:
            return 1.0, n_cc
        
        return float(good_pixels / total_pixels), n_cc


class LocalContrast:
    """
    Roots should have clear edges - use local contrast features.
    """
    
    @staticmethod
    def boundary_gradient(pred: np.ndarray, volume_slice: np.ndarray) -> float:
        """
        Compute average gradient magnitude at prediction boundaries.
        Roots should have strong edges (high gradient at boundaries).
        """
        pred_bin = (pred > 0.5).astype(np.float32)
        
        if pred_bin.sum() < 10:
            return 0.5
        
        # Find boundaries via erosion
        struct = np.ones((3, 3))
        eroded = binary_erosion(pred_bin, structure=struct)
        boundary = pred_bin.astype(bool) & ~eroded
        
        if boundary.sum() < 5:
            return 0.5
        
        # Compute gradients
        gy = ndimage.sobel(volume_slice, axis=0)
        gx = ndimage.sobel(volume_slice, axis=1)
        grad_mag = np.sqrt(gy**2 + gx**2)
        
        # Average gradient at boundaries
        boundary_grad = grad_mag[boundary].mean()
        
        # Normalize (typical grad range 0-0.5 for normalized images)
        return float(np.clip(boundary_grad / 0.3, 0, 1))
    
    @staticmethod
    def interior_homogeneity(pred: np.ndarray, volume_slice: np.ndarray) -> float:
        """
        Roots should be relatively homogeneous inside.
        Low variance inside predicted regions = good.
        """
        pred_bin = (pred > 0.5)
        
        if pred_bin.sum() < 20:
            return 0.5
        
        # Interior via erosion
        struct = np.ones((3, 3))
        interior = binary_erosion(pred_bin, structure=struct, iterations=2)
        
        if interior.sum() < 10:
            return 0.5
        
        # Variance inside
        interior_vals = volume_slice[interior]
        variance = np.var(interior_vals)
        
        # Low variance = good (homogeneous)
        # Typical variance for roots: 0.001-0.01
        return float(np.exp(-variance * 50))


class RootShapeEnv(gym.Env):
    """
    Environment with shape constraints for thin, elongated root structures.
    
    Key differences from v4:
    1. Shape penalty for wide/blobby predictions
    2. Masked intensity (only in dilated GT region)
    3. Explicit false positive penalty
    4. Boundary gradient reward (roots have edges)
    5. Curriculum: start with accuracy, add continuity later
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        # Core coefficients
        dice_coef: float = 1.0,
        recall_coef: float = 0.8,          # High - finding roots is critical
        precision_coef: float = 0.5,       # NEW: penalize false positives
        # Shape constraints
        shape_coef: float = 0.3,           # Thinness/shape reward
        max_avg_width: float = 8.0,        # Max average width in pixels
        # Intensity (masked)
        intensity_coef: float = 0.2,       # Lower - only use in ROI
        roi_dilation: int = 15,            # Dilate GT mask for intensity
        # Continuity (curriculum - increases over time)
        continuity_coef: float = 0.1,      # Start low
        # Local contrast
        boundary_coef: float = 0.2,        # Reward for clear boundaries
        # History
        history_len: int = 3,
        # Episode
        slices_per_episode: int = 64,
        random_direction: bool = True,
        # Patch mode
        patch_size: int = None,            # If set, use patches
        patch_jitter: int = 20,
    ):
        super().__init__()
        
        # Store volume
        self.volume_raw = volume.astype(np.float32).copy()
        self.volume = volume.astype(np.float32)
        if self.volume.max() > 1:
            self.volume = self.volume / 255.0
            self.volume_raw = self.volume_raw / 255.0
        
        self.mask = (mask > 0).astype(np.float32)
        self.D, self.H, self.W = self.mask.shape
        
        # Coefficients
        self.dice_coef = float(dice_coef)
        self.recall_coef = float(recall_coef)
        self.precision_coef = float(precision_coef)
        self.shape_coef = float(shape_coef)
        self.max_avg_width = float(max_avg_width)
        self.intensity_coef = float(intensity_coef)
        self.roi_dilation = int(roi_dilation)
        self.continuity_coef = float(continuity_coef)
        self.boundary_coef = float(boundary_coef)
        
        self.history_len = int(history_len)
        self.slices_per_episode = min(slices_per_episode, self.D)
        self.random_direction = random_direction
        
        # Patch settings
        self.use_patches = patch_size is not None
        self.patch_size = patch_size or self.H
        self.patch_jitter = patch_jitter
        
        # Compute intensity profile from GT regions only
        self._compute_intensity_profile()
        
        # Precompute ROI masks (dilated GT)
        self._compute_roi_masks()
        
        # Find root regions for patch extraction
        self._find_root_regions()
        
        # Spaces
        obs_size = self.patch_size if self.use_patches else self.H
        self.action_space = spaces.MultiBinary(obs_size * obs_size)
        self.observation_space = spaces.Dict({
            "slice_pixels": spaces.Box(0.0, 1.0, shape=(obs_size, obs_size), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(history_len, obs_size, obs_size), dtype=np.float32),
            "roi_mask": spaces.Box(0.0, 1.0, shape=(obs_size, obs_size), dtype=np.float32),
            "slice_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        
        # Episode state
        self._slice_order = None
        self.current_step = 0
        self.prev_preds_buffer = None
        self.current_patch_center = None
        
        # Curriculum tracking
        self.total_steps = 0
        
    def _compute_intensity_profile(self):
        """Compute intensity statistics from ground truth regions only."""
        root_intensities = self.volume_raw[self.mask > 0]
        
        if len(root_intensities) > 100:
            self.intensity_mean = float(np.mean(root_intensities))
            self.intensity_std = float(np.std(root_intensities))
            self.intensity_min = float(np.percentile(root_intensities, 5))
            self.intensity_max = float(np.percentile(root_intensities, 95))
        else:
            # Fallback
            self.intensity_mean = 0.22
            self.intensity_std = 0.025
            self.intensity_min = 0.18
            self.intensity_max = 0.26
        
        print(f"Intensity profile: mean={self.intensity_mean:.3f}, std={self.intensity_std:.3f}, "
              f"range=[{self.intensity_min:.3f}, {self.intensity_max:.3f}]")
    
    def _compute_roi_masks(self):
        """Precompute dilated ROI masks around ground truth regions."""
        print("Computing ROI masks...")
        struct = np.ones((self.roi_dilation * 2 + 1, self.roi_dilation * 2 + 1))
        self.roi_masks = np.zeros_like(self.mask)
        
        for i in range(self.D):
            if self.mask[i].sum() > 0:
                self.roi_masks[i] = binary_dilation(
                    self.mask[i], structure=struct
                ).astype(np.float32)
        
        # Fill gaps using neighbors
        for i in range(self.D):
            if self.roi_masks[i].sum() == 0:
                for offset in range(1, min(30, self.D)):
                    if i - offset >= 0 and self.roi_masks[i - offset].sum() > 0:
                        self.roi_masks[i] = self.roi_masks[i - offset]
                        break
                    if i + offset < self.D and self.roi_masks[i + offset].sum() > 0:
                        self.roi_masks[i] = self.roi_masks[i + offset]
                        break
        
        print(f"ROI coverage: {self.roi_masks.mean() * 100:.2f}% of volume")
    
    def _find_root_regions(self):
        """Find root centroids per slice for patch extraction."""
        self.slice_info = []
        
        for i in range(self.D):
            if self.mask[i].sum() > 0:
                y_coords, x_coords = np.where(self.mask[i] > 0)
                cy, cx = y_coords.mean(), x_coords.mean()
                self.slice_info.append({
                    "has_root": True,
                    "centroid": (float(cy), float(cx)),
                    "fg_ratio": float(self.mask[i].mean()),
                })
            else:
                self.slice_info.append({
                    "has_root": False,
                    "centroid": None,
                    "fg_ratio": 0.0,
                })
        
        n_with_roots = sum(1 for s in self.slice_info if s["has_root"])
        print(f"Slices with roots: {n_with_roots}/{self.D}")
    
    def _get_patch_center(self, slice_idx: int) -> tuple:
        """Get center for patch extraction."""
        info = self.slice_info[slice_idx]
        
        if info["has_root"]:
            cy, cx = info["centroid"]
            if self.patch_jitter > 0:
                cy += np.random.randint(-self.patch_jitter, self.patch_jitter + 1)
                cx += np.random.randint(-self.patch_jitter, self.patch_jitter + 1)
        else:
            if self.current_patch_center is not None:
                cy, cx = self.current_patch_center
                cy += np.random.randint(-5, 6)
                cx += np.random.randint(-5, 6)
            else:
                cy, cx = self.H // 2, self.W // 2
        
        half = self.patch_size // 2
        cy = int(np.clip(cy, half, self.H - half))
        cx = int(np.clip(cx, half, self.W - half))
        
        return cy, cx
    
    def _extract_patch(self, arr: np.ndarray, cy: int, cx: int) -> np.ndarray:
        """Extract patch centered at (cy, cx)."""
        half = self.patch_size // 2
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        
        if y1 < 0 or y2 > self.H or x1 < 0 or x2 > self.W:
            patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            src_y1, src_y2 = max(0, y1), min(self.H, y2)
            src_x1, src_x2 = max(0, x1), min(self.W, x2)
            dst_y1 = src_y1 - y1
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            dst_x1 = src_x1 - x1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            patch[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
            return patch
        
        return arr[y1:y2, x1:x2].copy()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sample starting slice preferring foreground
        fg_slices = [i for i, s in enumerate(self.slice_info) if s["has_root"]]
        
        if len(fg_slices) >= self.slices_per_episode // 2:
            anchor = np.random.choice(fg_slices)
        else:
            anchor = self.D // 2
        
        half = self.slices_per_episode // 2
        start = max(0, anchor - half)
        end = min(self.D, start + self.slices_per_episode)
        start = max(0, end - self.slices_per_episode)
        
        self._slice_order = np.arange(start, end)
        
        if self.random_direction and np.random.random() < 0.5:
            self._slice_order = self._slice_order[::-1]
        
        self.current_step = 0
        self.current_patch_center = None
        
        obs_size = self.patch_size if self.use_patches else self.H
        self.prev_preds_buffer = deque(
            [np.zeros((obs_size, obs_size), dtype=np.float32) 
             for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get current observation."""
        slice_idx = self._slice_order[self.current_step]
        
        if self.use_patches:
            cy, cx = self._get_patch_center(slice_idx)
            self.current_patch_center = (cy, cx)
            
            slice_pixels = self._extract_patch(self.volume[slice_idx], cy, cx)
            roi_mask = self._extract_patch(self.roi_masks[slice_idx], cy, cx)
        else:
            slice_pixels = self.volume[slice_idx].astype(np.float32)
            roi_mask = self.roi_masks[slice_idx].astype(np.float32)
        
        prev_preds = np.stack(list(self.prev_preds_buffer), axis=0)
        
        return {
            "slice_pixels": slice_pixels,
            "prev_preds": prev_preds,
            "roi_mask": roi_mask,
            "slice_index": np.array([slice_idx / self.D], dtype=np.float32),
        }
    
    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum()
        if union == 0:
            return 1.0
        return float(2 * intersection / (union + 1e-6))
    
    def _compute_recall(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        tp = (pred_bin * gt_bin).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()
        if tp + fn == 0:
            return 1.0
        return float(tp / (tp + fn + 1e-6))
    
    def _compute_precision(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)
        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        if tp + fp == 0:
            return 1.0
        return float(tp / (tp + fp + 1e-6))
    
    def _compute_intensity_reward(self, pred: np.ndarray, intensity_slice: np.ndarray, 
                                   roi_mask: np.ndarray) -> float:
        """Compute intensity reward ONLY within ROI mask."""
        pred_bin = (pred > 0.5).astype(np.float32)
        
        # Only consider predictions within ROI
        pred_in_roi = pred_bin * roi_mask
        
        if pred_in_roi.sum() < 5:
            return 0.5  # Neutral if no predictions in ROI
        
        # Get intensities at predicted locations within ROI
        pred_intensities = intensity_slice[pred_in_roi > 0.5]
        
        # Check if within expected range
        in_range = ((pred_intensities >= self.intensity_min) & 
                    (pred_intensities <= self.intensity_max))
        
        return float(np.mean(in_range))
    
    def _compute_shape_reward(self, pred: np.ndarray) -> float:
        """Reward thin, elongated predictions; penalize blobs."""
        pred_bin = (pred > 0.5).astype(np.float32)
        
        if pred_bin.sum() < 20:
            return 0.5  # Not enough to evaluate
        
        # Compactness penalty
        compact = ShapeMetrics.compactness_penalty(pred, self.max_avg_width)
        
        # Thinness reward
        thinness = ShapeMetrics.thinness_ratio(pred)
        
        # Component quality
        good_ratio, n_cc = ShapeMetrics.connected_component_stats(pred)
        
        # Combine: 40% compactness, 30% thinness, 30% component quality
        return 0.4 * compact + 0.3 * thinness + 0.3 * good_ratio
    
    def _compute_continuity_reward(self, pred: np.ndarray, prev_pred: np.ndarray) -> float:
        """Vertical continuity between slices."""
        pred_bin = (pred > 0.5)
        prev_bin = (prev_pred > 0.5)
        
        if not prev_bin.any():
            return 1.0 if not pred_bin.any() else 0.5
        if not pred_bin.any():
            return 0.0  # Had roots before, none now = bad
        
        # Dilate previous prediction
        struct = np.ones((5, 5))
        prev_dilated = binary_dilation(prev_bin, structure=struct)
        
        # Check overlap
        overlap = np.logical_and(pred_bin, prev_dilated).sum()
        
        return float(overlap / (pred_bin.sum() + 1e-6))
    
    def _compute_boundary_reward(self, pred: np.ndarray, volume_slice: np.ndarray) -> float:
        """Reward predictions with clear boundaries."""
        return LocalContrast.boundary_gradient(pred, volume_slice)
    
    def step(self, action):
        slice_idx = self._slice_order[self.current_step]
        
        # Get size
        obs_size = self.patch_size if self.use_patches else self.H
        pred = action.reshape(obs_size, obs_size).astype(np.float32)
        
        # Get ground truth and other data
        if self.use_patches:
            cy, cx = self.current_patch_center
            gt = self._extract_patch(self.mask[slice_idx], cy, cx)
            intensity = self._extract_patch(self.volume_raw[slice_idx], cy, cx)
            roi = self._extract_patch(self.roi_masks[slice_idx], cy, cx)
            volume_slice = self._extract_patch(self.volume[slice_idx], cy, cx)
        else:
            gt = self.mask[slice_idx]
            intensity = self.volume_raw[slice_idx]
            roi = self.roi_masks[slice_idx]
            volume_slice = self.volume[slice_idx]
        
        # --- Compute rewards ---
        
        # Core accuracy metrics
        dice = self._compute_dice(pred, gt)
        recall = self._compute_recall(pred, gt)
        precision = self._compute_precision(pred, gt)
        
        # Shape constraint
        shape = self._compute_shape_reward(pred)
        
        # Intensity (masked to ROI)
        intensity_rew = self._compute_intensity_reward(pred, intensity, roi)
        
        # Continuity (curriculum - weight increases over time)
        prev_pred = self.prev_preds_buffer[-1] if len(self.prev_preds_buffer) > 0 else np.zeros_like(pred)
        continuity = self._compute_continuity_reward(pred, prev_pred)
        
        # Boundary clarity
        boundary = self._compute_boundary_reward(pred, volume_slice)
        
        # Curriculum: continuity weight increases with total steps
        # Start at 0.1, increase to full value over 50k steps
        curr_continuity_coef = self.continuity_coef * min(1.0, self.total_steps / 50000)
        
        # Combined reward
        reward = (
            self.dice_coef * dice +
            self.recall_coef * recall +
            self.precision_coef * precision +
            self.shape_coef * shape +
            self.intensity_coef * intensity_rew +
            curr_continuity_coef * continuity +
            self.boundary_coef * boundary
        )
        
        info = {
            "dice": dice,
            "recall": recall,
            "precision": precision,
            "shape": shape,
            "intensity": intensity_rew,
            "continuity": continuity,
            "boundary": boundary,
            "slice_idx": slice_idx,
            "pred_sum": float((pred > 0.5).sum()),
            "gt_sum": float((gt > 0.5).sum()),
            "curr_continuity_coef": curr_continuity_coef,
        }
        
        # Update state
        self.prev_preds_buffer.append(pred.copy())
        self.current_step += 1
        self.total_steps += 1
        
        done = self.current_step >= len(self._slice_order)
        
        if done:
            obs = self._terminal_obs()
        else:
            obs = self._get_obs()
        
        return obs, reward, done, False, info
    
    def _terminal_obs(self):
        obs_size = self.patch_size if self.use_patches else self.H
        return {
            "slice_pixels": np.zeros((obs_size, obs_size), dtype=np.float32),
            "prev_preds": np.zeros((self.history_len, obs_size, obs_size), dtype=np.float32),
            "roi_mask": np.zeros((obs_size, obs_size), dtype=np.float32),
            "slice_index": np.array([1.0], dtype=np.float32),
        }


def test_shape_metrics():
    """Test shape metrics on synthetic data."""
    print("Testing ShapeMetrics...")
    
    # Create a thin line (root-like)
    line = np.zeros((64, 64), dtype=np.float32)
    line[30:34, 10:50] = 1.0  # Thin horizontal line
    
    # Create a blob (not root-like)
    blob = np.zeros((64, 64), dtype=np.float32)
    blob[20:45, 20:45] = 1.0  # Square blob
    
    print(f"Line - thinness: {ShapeMetrics.thinness_ratio(line):.3f}, "
          f"avg_width: {ShapeMetrics.avg_width(line):.1f}, "
          f"compactness: {ShapeMetrics.compactness_penalty(line):.3f}")
    
    print(f"Blob - thinness: {ShapeMetrics.thinness_ratio(blob):.3f}, "
          f"avg_width: {ShapeMetrics.avg_width(blob):.1f}, "
          f"compactness: {ShapeMetrics.compactness_penalty(blob):.3f}")
    
    good_line, _ = ShapeMetrics.connected_component_stats(line)
    good_blob, _ = ShapeMetrics.connected_component_stats(blob)
    print(f"Line good ratio: {good_line:.3f}, Blob good ratio: {good_blob:.3f}")


if __name__ == "__main__":
    test_shape_metrics()

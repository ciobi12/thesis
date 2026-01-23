import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatchClassificationEnv(gym.Env):
    """
    Patch-wise environment: bottom-left to top-right traversal.
    One step = one patch (N x N). The agent acts with a per-pixel MultiBinary (N*N) decision.

    Context for each patch includes:
    - Current patch pixels
    - All 8 neighboring patches (left, right, top-left, top, top-right, bottom-left, bottom, bottom-right)
    - Predicted masks of already-visited neighbor patches

    Reward per patch (scalar) combines:
    - Base accuracy: mean of |action - gt| across all pixels in patch
    - Boundary consistency: penalizes edge pixels that don't connect to neighbor predictions
    - Sparsity bonus: small reward for not over-predicting (reduces false positives at edges)
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 image, 
                 mask, 
                 patch_size=16, 
                 base_coef=1.0,
                 boundary_coef=0.2,
                 sparsity_coef=0.05,
                 start_from_bottom_left=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        self.image = img if img.ndim == 3 else img[..., None]  # (H, W, C)
        self.mask = (mask > 0).astype(np.float32)  # (H, W)

        self.H, self.W = self.mask.shape
        self.C = self.image.shape[2]
        self.N = int(patch_size)
        self.base_coef = float(base_coef)
        self.boundary_coef = float(boundary_coef)
        self.sparsity_coef = float(sparsity_coef)
        self.start_from_bottom_left = bool(start_from_bottom_left)

        # Pad image and mask to multiples of N for fixed-size patches
        H_pad = (self.N - (self.H % self.N)) % self.N
        W_pad = (self.N - (self.W % self.N)) % self.N
        self.pad = (H_pad, W_pad)
        if H_pad or W_pad:
            self.image = np.pad(self.image, ((0, H_pad), (0, W_pad), (0, 0)), mode="constant", constant_values=0)
            self.mask = np.pad(self.mask, ((0, H_pad), (0, W_pad)), mode="constant", constant_values=0)
        self.Hp, self.Wp = self.mask.shape

        self.patch_rows = self.Hp // self.N
        self.patch_cols = self.Wp // self.N

        # Build traversal order: bottom row to top row; within row left->right
        order = []
        row_range = range(self.patch_rows-1, -1, -1) if self.start_from_bottom_left else range(self.patch_rows)
        for pr in row_range:
            for pc in range(self.patch_cols):
                y0 = pr * self.N
                x0 = pc * self.N
                order.append((y0, x0))
        self._order = order
        self._idx = 0

        # Spaces: action is MultiBinary over N*N pixels in the patch
        self.action_space = spaces.MultiBinary(self.N * self.N)
        # 8 neighbor patches + current patch info
        self.observation_space = spaces.Dict({
            "patch_pixels": spaces.Box(0.0, 1.0, shape=(self.N, self.N, self.C), dtype=np.float32),
            "neighbor_patches": spaces.Box(0.0, 1.0, shape=(8, self.N, self.N, self.C), dtype=np.float32),  # all 8 neighbors
            "neighbor_masks": spaces.Box(0.0, 1.0, shape=(8, self.N, self.N), dtype=np.float32),  # predicted masks of all 8 neighbors
            "neighbor_valid": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32),  # which neighbors are valid (not out of bounds)
            "patch_coords": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),  # (row_norm, col_norm)
        })

        # Store predictions for neighbor mask lookup
        self.predictions = None
        
        # Episode-level accumulators for metrics
        self.episode_predictions = None
        self.episode_ground_truth = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        
        # Initialize predictions storage (for neighbor mask lookup)
        self.predictions = np.zeros((self.Hp, self.Wp), dtype=np.float32)
        
        # Initialize episode accumulators
        self.episode_predictions = np.zeros((self.Hp, self.Wp), dtype=np.float32)
        self.episode_ground_truth = self.mask.copy()
        
        return self._get_obs(), {}

    def _is_valid_patch(self, row, col):
        """Check if patch position is valid."""
        return 0 <= row < self.patch_rows and 0 <= col < self.patch_cols

    def _get_patch_at(self, row, col):
        """Get patch pixels at grid position (row, col). Returns zeros if out of bounds."""
        if not self._is_valid_patch(row, col):
            return np.zeros((self.N, self.N, self.C), dtype=np.float32)
        y0 = row * self.N
        x0 = col * self.N
        return self.image[y0:y0+self.N, x0:x0+self.N, :].astype(np.float32)

    def _get_pred_mask_at(self, row, col):
        """Get predicted mask at grid position (row, col). Returns zeros if out of bounds."""
        if not self._is_valid_patch(row, col):
            return np.zeros((self.N, self.N), dtype=np.float32)
        y0 = row * self.N
        x0 = col * self.N
        return self.predictions[y0:y0+self.N, x0:x0+self.N].astype(np.float32)

    def _get_neighbor_offsets(self):
        """
        Returns offsets for all 8 neighbors in order:
        [left, right, top-left, top, top-right, bottom-left, bottom, bottom-right]
        Note: "top" and "bottom" are relative to image coordinates, not traversal order.
        """
        return [
            (0, -1),   # left
            (0, 1),    # right
            (-1, -1),  # top-left
            (-1, 0),   # top
            (-1, 1),   # top-right
            (1, -1),   # bottom-left
            (1, 0),    # bottom
            (1, 1),    # bottom-right
        ]

    def _get_obs(self):
        y0, x0 = self._order[self._idx]
        patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N, N, C)
        pr = y0 // self.N  # current patch row
        pc = x0 // self.N  # current patch col
        coords = np.array([(pr + 1) / self.patch_rows, (pc + 1) / self.patch_cols], dtype=np.float32)
        
        # Get all 8 neighboring patches and their masks
        offsets = self._get_neighbor_offsets()
        neighbor_patches = []
        neighbor_masks = []
        neighbor_valid = []
        
        for dr, dc in offsets:
            nr, nc = pr + dr, pc + dc
            neighbor_patches.append(self._get_patch_at(nr, nc))
            neighbor_masks.append(self._get_pred_mask_at(nr, nc))
            neighbor_valid.append(1.0 if self._is_valid_patch(nr, nc) else 0.0)
        
        return {
            "patch_pixels": patch.astype(np.float32),
            "neighbor_patches": np.stack(neighbor_patches, axis=0),  # (8, N, N, C)
            "neighbor_masks": np.stack(neighbor_masks, axis=0),  # (8, N, N)
            "neighbor_valid": np.array(neighbor_valid, dtype=np.float32),  # (8,)
            "patch_coords": coords,
        }

    def _compute_boundary_penalty(self, action, pr, pc):
        """
        Compute penalty for predicting edge pixels that don't connect to neighbor predictions.
        This discourages the grid-like artifact pattern.
        
        Returns a penalty (negative value) for disconnected edge predictions.
        """
        N = self.N
        action_binary = (action > 0.5).astype(np.float32)
        penalty = 0.0
        edge_violations = 0
        total_edge_predictions = 0
        
        # Get neighbor masks
        # Neighbor order: [left, right, top-left, top, top-right, bottom-left, bottom, bottom-right]
        left_mask = self._get_pred_mask_at(pr, pc - 1)
        right_mask = self._get_pred_mask_at(pr, pc + 1)  # Not visited yet in left-to-right traversal
        top_mask = self._get_pred_mask_at(pr - 1, pc)     # Not visited yet in bottom-to-top
        bottom_mask = self._get_pred_mask_at(pr + 1, pc)  # Already visited
        
        # Check LEFT edge (column 0) - should connect to right edge of left neighbor
        left_edge_preds = action_binary[:, 0]
        if np.any(left_edge_preds > 0.5):
            total_edge_predictions += np.sum(left_edge_preds > 0.5)
            if self._is_valid_patch(pr, pc - 1):
                # Check if left neighbor's right edge has corresponding predictions
                left_neighbor_right_edge = left_mask[:, -1]
                # For each predicted pixel on our left edge, check if there's a nearby prediction on neighbor
                for i in range(N):
                    if left_edge_preds[i] > 0.5:
                        # Check if any pixel in a small window on neighbor's right edge is predicted
                        start_i = max(0, i - 2)
                        end_i = min(N, i + 3)
                        if not np.any(left_neighbor_right_edge[start_i:end_i] > 0.5):
                            edge_violations += 1
        
        # Check BOTTOM edge (row N-1) - should connect to top edge of bottom neighbor
        bottom_edge_preds = action_binary[-1, :]
        if np.any(bottom_edge_preds > 0.5):
            total_edge_predictions += np.sum(bottom_edge_preds > 0.5)
            if self._is_valid_patch(pr + 1, pc):
                bottom_neighbor_top_edge = bottom_mask[0, :]
                for j in range(N):
                    if bottom_edge_preds[j] > 0.5:
                        start_j = max(0, j - 2)
                        end_j = min(N, j + 3)
                        if not np.any(bottom_neighbor_top_edge[start_j:end_j] > 0.5):
                            edge_violations += 1
        
        # Penalize edge violations proportionally
        if total_edge_predictions > 0:
            violation_ratio = edge_violations / (total_edge_predictions + 1e-8)
            penalty = -violation_ratio
        
        return penalty

    def _compute_sparsity_bonus(self, action, gt_patch):
        """
        Compute a bonus for not over-predicting.
        Penalizes false positives more heavily, especially isolated ones.
        """
        action_binary = (action > 0.5).astype(np.float32)
        
        # Count predictions vs ground truth
        pred_count = np.sum(action_binary)
        gt_count = np.sum(gt_patch)
        
        if pred_count == 0:
            return 0.0
        
        # Penalize over-prediction (predicting more than GT)
        if pred_count > gt_count * 1.5:  # Allow some tolerance
            over_prediction_ratio = (pred_count - gt_count) / (self.N * self.N)
            return -over_prediction_ratio
        
        return 0.0

    def step(self, action):
        # action: (N*N,) or (N,N)
        if action.ndim == 1:
            action = action.reshape(self.N, self.N)
        action = action.astype(np.float32)
        y0, x0 = self._order[self._idx]
        pr = y0 // self.N
        pc = x0 // self.N

        gt_patch = self.mask[y0:y0+self.N, x0:x0+self.N]  # (N,N)

        # ========== SCALAR REWARD FOR PATCH ==========
        
        # 1. BASE REWARD: Mean accuracy against ground truth (scalar)
        pixel_errors = np.abs(action - gt_patch)  # (N, N)
        base_reward = -self.base_coef * pixel_errors.mean()  # scalar
        
        # Track per-patch stats
        action_binary = (action > 0.5).astype(np.float32)
        true_positives = np.sum((action_binary == 1) & (gt_patch == 1))
        false_positives = np.sum((action_binary == 1) & (gt_patch == 0))
        false_negatives = np.sum((action_binary == 0) & (gt_patch == 1))
        
        # Store prediction for this patch
        self.predictions[y0:y0+self.N, x0:x0+self.N] = action_binary
        self.episode_predictions[y0:y0+self.N, x0:x0+self.N] = action_binary

        # 2. BOUNDARY CONSISTENCY REWARD: Penalize disconnected edge predictions
        boundary_reward = self._compute_boundary_penalty(action, pr, pc) * self.boundary_coef

        # 3. SPARSITY REWARD: Discourage over-prediction
        sparsity_reward = self._compute_sparsity_bonus(action, gt_patch) * self.sparsity_coef

        # Combine all rewards (scalar)
        reward = float(base_reward + boundary_reward + sparsity_reward)

        self._idx += 1
        terminated = self._idx >= len(self._order)

        info = {
            "reward": reward,
            "base_reward": float(base_reward),
            "boundary_reward": float(boundary_reward),
            "sparsity_reward": float(sparsity_reward),
            "patch_origin": (int(y0), int(x0)),
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "patch_pixels": np.zeros((self.N, self.N, self.C), dtype=np.float32),
            "neighbor_patches": np.zeros((8, self.N, self.N, self.C), dtype=np.float32),
            "neighbor_masks": np.zeros((8, self.N, self.N), dtype=np.float32),
            "neighbor_valid": np.zeros((8,), dtype=np.float32),
            "patch_coords": np.array([1.0, 1.0], dtype=np.float32),
        }

    def reconstruct_blank(self):
        """Helper to allocate a padded-size prediction canvas."""
        return np.zeros((self.Hp, self.Wp), dtype=np.uint8)

    def crop_to_original(self, arr2d):
        H_pad, W_pad = self.pad
        if H_pad or W_pad:
            return arr2d[: self.H, : self.W]
        return arr2d

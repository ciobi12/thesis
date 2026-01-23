import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatchClassificationEnv(gym.Env):
    """
    Patch-wise environment: bottom-left to top-right traversal.
    One step = one patch (N x N). The agent acts with a per-pixel MultiBinary (N*N) decision.

    Context for each patch includes:
    - Current patch pixels
    - 3 patches below (bottom-left, directly below, bottom-right)
    - 3 patches above (top-left, directly above, top-right)
    - Left neighbor patch

    Reward per patch (scalar) combines:
    - Base accuracy: mean of |action - gt| across all pixels in patch
    - Continuity: checks similarity with neighbor patches for structural coherence
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 image, 
                 mask, 
                 patch_size=16, 
                 base_coef=1.0,
                 continuity_coef=0.1, 
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
        self.continuity_coef = float(continuity_coef)
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
        self.observation_space = spaces.Dict({
            "patch_pixels": spaces.Box(0.0, 1.0, shape=(self.N, self.N, self.C), dtype=np.float32),
            "below_patches": spaces.Box(0.0, 1.0, shape=(3, self.N, self.N, self.C), dtype=np.float32),  # bottom-left, below, bottom-right
            "above_patches": spaces.Box(0.0, 1.0, shape=(3, self.N, self.N, self.C), dtype=np.float32),  # top-left, above, top-right
            "left_patch": spaces.Box(0.0, 1.0, shape=(self.N, self.N, self.C), dtype=np.float32),
            "neighbor_masks": spaces.Box(0.0, 1.0, shape=(4, self.N, self.N), dtype=np.float32),  # masks of left + 3 below patches
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

    def _get_patch_at(self, row, col):
        """Get patch pixels at grid position (row, col). Returns zeros if out of bounds."""
        if row < 0 or row >= self.patch_rows or col < 0 or col >= self.patch_cols:
            return np.zeros((self.N, self.N, self.C), dtype=np.float32)
        y0 = row * self.N
        x0 = col * self.N
        return self.image[y0:y0+self.N, x0:x0+self.N, :].astype(np.float32)

    def _get_pred_mask_at(self, row, col):
        """Get predicted mask at grid position (row, col). Returns zeros if out of bounds."""
        if row < 0 or row >= self.patch_rows or col < 0 or col >= self.patch_cols:
            return np.zeros((self.N, self.N), dtype=np.float32)
        y0 = row * self.N
        x0 = col * self.N
        return self.predictions[y0:y0+self.N, x0:x0+self.N].astype(np.float32)

    def _get_obs(self):
        y0, x0 = self._order[self._idx]
        patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N, N, C)
        pr = y0 // self.N  # current patch row
        pc = x0 // self.N  # current patch col
        coords = np.array([(pr + 1) / self.patch_rows, (pc + 1) / self.patch_cols], dtype=np.float32)
        
        # For bottom-left to top-right traversal:
        # - "below" means higher row index (already visited)
        # - "above" means lower row index (not yet visited)
        if self.start_from_bottom_left:
            below_row = pr + 1
            above_row = pr - 1
        else:
            below_row = pr - 1
            above_row = pr + 1
        
        # Get 3 patches below (bottom-left, directly below, bottom-right)
        below_patches = np.stack([
            self._get_patch_at(below_row, pc - 1),  # bottom-left
            self._get_patch_at(below_row, pc),      # directly below
            self._get_patch_at(below_row, pc + 1),  # bottom-right
        ], axis=0)  # (3, N, N, C)
        
        # Get 3 patches above (top-left, directly above, top-right)
        above_patches = np.stack([
            self._get_patch_at(above_row, pc - 1),  # top-left
            self._get_patch_at(above_row, pc),      # directly above
            self._get_patch_at(above_row, pc + 1),  # top-right
        ], axis=0)  # (3, N, N, C)
        
        # Get left patch
        left_patch = self._get_patch_at(pr, pc - 1)  # (N, N, C)
        
        # Get predicted masks of neighbor patches (left + 3 below) for continuity reward
        neighbor_masks = np.stack([
            self._get_pred_mask_at(pr, pc - 1),       # left
            self._get_pred_mask_at(below_row, pc - 1),  # bottom-left
            self._get_pred_mask_at(below_row, pc),      # directly below
            self._get_pred_mask_at(below_row, pc + 1),  # bottom-right
        ], axis=0)  # (4, N, N)
        
        return {
            "patch_pixels": patch.astype(np.float32),
            "below_patches": below_patches,  # (3, N, N, C)
            "above_patches": above_patches,  # (3, N, N, C)
            "left_patch": left_patch,  # (N, N, C)
            "neighbor_masks": neighbor_masks,  # (4, N, N)
            "patch_coords": coords,
        }

    def _compute_patch_similarity(self, patch1, patch2):
        """Compute similarity between two patches based on pixel intensities."""
        # Use mean intensity correlation
        p1_flat = patch1.flatten()
        p2_flat = patch2.flatten()
        
        # Normalize to avoid division issues
        p1_norm = p1_flat - p1_flat.mean()
        p2_norm = p2_flat - p2_flat.mean()
        
        std1 = p1_norm.std()
        std2 = p2_norm.std()
        
        if std1 < 1e-8 or std2 < 1e-8:
            return 0.0
        
        correlation = np.dot(p1_norm, p2_norm) / (len(p1_flat) * std1 * std2)
        return float(correlation)

    def step(self, action):
        # action: (N*N,) or (N,N)
        if action.ndim == 1:
            action = action.reshape(self.N, self.N)
        action = action.astype(np.float32)
        y0, x0 = self._order[self._idx]
        pr = y0 // self.N
        pc = x0 // self.N

        gt_patch = self.mask[y0:y0+self.N, x0:x0+self.N]  # (N,N)
        img_patch = self.image[y0:y0+self.N, x0:x0+self.N, :]  # (N,N,C)

        # ========== SCALAR REWARD FOR PATCH ==========
        
        # 1. BASE REWARD: Mean accuracy against ground truth (scalar)
        pixel_errors = np.abs(action - gt_patch)  # (N, N)
        base_reward = -self.base_coef * pixel_errors.mean()  # scalar
        
        # Track per-patch stats
        true_positives = np.sum((action == 1) & (gt_patch == 1))
        false_positives = np.sum((action == 1) & (gt_patch == 0))
        false_negatives = np.sum((action == 0) & (gt_patch == 1))
        
        # Store prediction for this patch
        self.predictions[y0:y0+self.N, x0:x0+self.N] = (action > 0.5).astype(np.float32)
        self.episode_predictions[y0:y0+self.N, x0:x0+self.N] = (action > 0.5).astype(np.float32)

        # 2. CONTINUITY REWARD: Based on neighbor similarity (scalar)
        continuity_reward = 0.0
        
        # Check if current patch has predicted on-path pixels
        has_on_path_pixels = np.any(action > 0.5)
        
        if has_on_path_pixels:
            # Get neighbor patches (left + 3 below) and their masks
            if self.start_from_bottom_left:
                below_row = pr + 1
            else:
                below_row = pr - 1
            
            neighbor_patches = [
                (self._get_patch_at(pr, pc - 1), self._get_pred_mask_at(pr, pc - 1)),       # left
                (self._get_patch_at(below_row, pc - 1), self._get_pred_mask_at(below_row, pc - 1)),  # bottom-left
                (self._get_patch_at(below_row, pc), self._get_pred_mask_at(below_row, pc)),      # directly below
                (self._get_patch_at(below_row, pc + 1), self._get_pred_mask_at(below_row, pc + 1)),  # bottom-right
            ]
            
            # Calculate similarity scores with each neighbor
            similarities = []
            for neighbor_img, neighbor_mask in neighbor_patches:
                sim = self._compute_patch_similarity(img_patch, neighbor_img)
                similarities.append((sim, neighbor_mask))
            
            # Find patch with highest similarity
            if len(similarities) > 0:
                best_sim, best_mask = max(similarities, key=lambda x: x[0])
                
                # Check if the most similar patch has on-path pixels
                if np.any(best_mask > 0.5):
                    continuity_reward = 0.0  # Connected to a path, no penalty
                else:
                    continuity_reward = -1.0  # Isolated, penalize
        else:
            # No on-path pixels predicted, no continuity penalty
            continuity_reward = 0.0
        
        continuity_reward *= self.continuity_coef

        # Combine all rewards (scalar)
        reward = float(base_reward + continuity_reward)

        self._idx += 1
        terminated = self._idx >= len(self._order)

        info = {
            "reward": reward,
            "base_reward": float(base_reward),
            "continuity_reward": float(continuity_reward),
            "patch_origin": (int(y0), int(x0)),
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
        }
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        return {
            "patch_pixels": np.zeros((self.N, self.N, self.C), dtype=np.float32),
            "below_patches": np.zeros((3, self.N, self.N, self.C), dtype=np.float32),
            "above_patches": np.zeros((3, self.N, self.N, self.C), dtype=np.float32),
            "left_patch": np.zeros((self.N, self.N, self.C), dtype=np.float32),
            "neighbor_masks": np.zeros((4, self.N, self.N), dtype=np.float32),
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

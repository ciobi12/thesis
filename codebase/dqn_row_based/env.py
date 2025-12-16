import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage

class PathReconstructionEnv(gym.Env):
    """Row-wise environment: bottom to top. One step = one row.
    
    Reward components per pixel (summed across the row):
    - Base accuracy:     -|action - gt|
    - Row continuity:    -continuity_coef * |action - prev_rows_action|
        (Only in-bounds neighbours are included.)
    """
    metadata = {"render_modes": []}

    def __init__(self, image, 
                 mask, 
                 continuity_coef=0.1, 
                 continuity_decay_factor=0.7,
                 history_len=3, 
                 start_from_bottom=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        self.image = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        # print(np.unique(self.mask))
        self.H, self.W = self.mask.shape
        self.C = 1 if self.image.ndim == 2 else self.image.shape[2]
        self.continuity_coef = float(continuity_coef)
        self.continuity_decay_factor = float(continuity_decay_factor)
        self.history_len = int(history_len)
        self.start_from_bottom = bool(start_from_bottom)
        
        # Expand image to add channel dimension if needed
        if self.image.ndim == 2:
            self.image = self.image[:, :, None]

        self.action_space = spaces.MultiBinary(self.W)
        self.observation_space = spaces.Dict({
            "row_pixels": spaces.Box(0.0, 1.0, shape=(self.W, self.C), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.W), dtype=np.float32),
            "prev_rows": spaces.Box(0.0, 1.0, shape=(self.history_len, self.W, self.C), dtype=np.float32),
            "row_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._row_order = None
        self.current_row_idx = None
        self.prev_preds_buffer = None  # Keep for reward calculation
        self.prev_rows_buffer = None   # New: for network input

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._row_order = np.arange(self.H-1, -1, -1) if self.start_from_bottom else np.arange(0, self.H)
        self.current_row_idx = 0
        self.prev_preds_buffer = deque([np.zeros((self.W,), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.prev_rows_buffer = deque([np.zeros((self.W, self.C), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self._row_order[self.current_row_idx]
        return {
            "row_pixels": self.image[row],  # shape (W, C)
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),  # (history_len, W)
            "prev_rows": np.array(self.prev_rows_buffer, dtype=np.float32),  # (history_len, W, C)
            "row_index": np.array([(row + 1) / self.H], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).clip(0, 1)
        row = self._row_order[self.current_row_idx]
        gt = self.mask[row]

        # ========== ENHANCED REWARD FOR VESSEL SEGMENTATION ==========
        
        # 1. WEIGHTED ACCURACY: Combat class imbalance (reduced weight - saturating)
        vessel_weight = 1.0  # Reduced from 3.0 since it saturates
        background_weight = 0.3  # Further reduced background weight
        
        vessel_mask = (gt > 0.5)
        weighted_accuracy = np.where(
            vessel_mask,
            -vessel_weight * np.abs(action - gt),      # Penalty for missing vessels
            -background_weight * np.abs(action - gt)   # Penalty for background errors
        )
        
        # 2. F1-BALANCED REWARD: Balance recall and precision
        true_positives = np.sum((action > 0.5) & (gt > 0.5))
        false_positives = np.sum((action > 0.5) & (gt < 0.5))
        false_negatives = np.sum((action < 0.5) & (gt > 0.5))
        
        # Stronger FP penalty to narrow predictions
        detection_reward = 2.0 * true_positives - 2.5 * false_positives - 1.5 * false_negatives
        
        # 3. THICKNESS CONTROL: Penalize over-thick predictions
        thickness_penalty = 0.0
        width_bonus = 0.0
        
        if np.sum(action > 0.5) > 0 and np.sum(gt > 0.5) > 0:
            from scipy.ndimage import binary_dilation, binary_erosion
            binary_pred = (action > 0.5).astype(np.uint8)
            binary_gt = (gt > 0.5).astype(np.uint8)
            
            # Compare local widths: count active pixels in sliding windows
            pred_width = np.sum(binary_pred)
            gt_width = np.sum(binary_gt)
            
            # STRONG penalty for being wider than ground truth
            if pred_width > gt_width:
                thickness_penalty = -3.0 * (pred_width - gt_width) / gt_width
            # Small bonus for matching width
            elif pred_width <= gt_width:
                width_bonus = 1.0 * (pred_width / gt_width)
            
            # Penalize isolated noisy pixels
            neighbor_count = ndimage.convolve(binary_pred.astype(float), np.ones(3), mode='constant')
            isolated_pixels = np.sum((binary_pred > 0) & (neighbor_count <= 1))
            if isolated_pixels > 0:
                thickness_penalty -= 2.0 * isolated_pixels
        
        # 4. CONTINUITY: Enhanced to allow branching
        continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        
        if len(self.prev_preds_buffer) >= 1:
            prev_row = list(self.prev_preds_buffer)[-1]
            
            # For each active pixel, check if it connects to previous row
            curr_active_indices = np.where(action > 0.5)[0]
            for idx in curr_active_indices:
                # Check 3-pixel neighborhood (allow diagonal connections for branching)
                left = max(0, idx - 1)
                right = min(self.W, idx + 2)
                
                # If connected to previous row, give positive reward
                if np.any(prev_row[left:right] > 0.5):
                    continuity_rewards[idx] = 0.8  # Reduced from 1.0
                else:
                    # STRONG penalty for isolated pixels (not connected to structure)
                    continuity_rewards[idx] = -3.0  # Increased from -2.0 to combat noise
            
            # Also consider inactive pixels that should maintain gaps
            # (don't penalize correctly predicting background between vessels)
            inactive_with_inactive_above = ((action < 0.5) & 
                                           (np.convolve(prev_row, [1,1,1], mode='same') < 0.5))
            continuity_rewards[inactive_with_inactive_above] = 0.1  # Reduced from 0.2
        
        continuity_rewards *= self.continuity_coef
        
        # 5. CONFIDENCE PENALTY: Penalize uncertain predictions (middle values)
        # Encourage confident decisions (close to 0 or 1)
        confidence_penalty = 0.0
        uncertain_pixels = ((action > 0.2) & (action < 0.8))
        if np.any(uncertain_pixels):
            # Penalize predictions near 0.5 (uncertain)
            uncertainty = np.abs(action[uncertain_pixels] - 0.5)
            confidence_penalty = -0.8 * np.sum(1.0 - 2.0 * uncertainty)  # Higher penalty closer to 0.5
        
        # 6. TOPOLOGY: Prevent creating disconnected fragments
        connectivity_reward = 0.0
        conn_info = {}
        
        if len(self.prev_preds_buffer) >= 2:
            # Build local 4-row window
            window_rows = list(self.prev_preds_buffer)[-3:]
            window_rows.append(action)
            window = np.vstack(window_rows)
            binary_window = (window > 0.5).astype(np.uint8)
            
            # Count components - penalize fragmentation
            labeled, num_components = ndimage.label(binary_window, structure=np.ones((3, 3)))
            
            if num_components > 2:  # Allow some branching (2-3 main branches)
                # Strong penalty for excessive fragmentation
                connectivity_reward = -2.0 * (num_components - 3)  # Increased from -1.5
                conn_info["num_components"] = num_components
                conn_info["fragmentation_penalty"] = connectivity_reward
        
        # Combine all rewards
        pixel_rewards = weighted_accuracy + continuity_rewards
        reward = float(pixel_rewards.sum()) + detection_reward + thickness_penalty + width_bonus + confidence_penalty + connectivity_reward
        
        conn_info.update({
            "detection_reward": detection_reward,
            "thickness_penalty": thickness_penalty,
            "width_bonus": width_bonus,
            "confidence_penalty": confidence_penalty,
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
        })

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_rows_buffer.append(self.image[row].copy())
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "weighted_accuracy": weighted_accuracy.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "connectivity_reward": connectivity_reward,
            "row_index": row,
        }
        info.update(conn_info)
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last row, return a zeroed observation (won't be used)
        return {
            "row_pixels": np.zeros((self.W, self.C), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_rows": np.array(self.prev_rows_buffer, dtype=np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
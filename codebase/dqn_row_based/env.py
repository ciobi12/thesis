import gymnasium as gym
import numpy as np

from gymnasium import spaces
from collections import deque
from scipy import ndimage

class PathReconstructionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, image, 
                 mask,
                 base_coef = 1,
                 continuity_coef=0.1, 
                 gradient_coef = 1,
                 history_len=3,
                 future_len=3,
                 start_from_bottom=True):
        super().__init__()
        assert image.shape[:2] == mask.shape, "Image and mask must have same height and width"
        self.image = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        self.mask = (mask > 0).astype(np.float32)  # binary ground truth
        # print(np.unique(self.mask))
        self.H, self.W = self.mask.shape
        self.C = 1 if self.image.ndim == 2 else self.image.shape[2]
        self.base_coef = float(base_coef)
        self.continuity_coef = float(continuity_coef)
        self.gradient_coef = float(gradient_coef)
        self.history_len = int(history_len)
        self.future_len = int(future_len)
        self.start_from_bottom = bool(start_from_bottom)
        
        # Expand image to add channel dimension if needed
        if self.image.ndim == 2:
            self.image = self.image[:, :, None]

        self.action_space = spaces.MultiBinary(self.W)
        self.observation_space = spaces.Dict({
            "row_pixels": spaces.Box(0.0, 1.0, shape=(self.W, self.C), dtype=np.float32),
            "prev_preds": spaces.Box(0.0, 1.0, shape=(self.history_len, self.W), dtype=np.float32),
            "prev_rows": spaces.Box(0.0, 1.0, shape=(self.history_len, self.W, self.C), dtype=np.float32),
            "future_rows": spaces.Box(0.0, 1.0, shape=(self.future_len, self.W, self.C), dtype=np.float32),
            "row_index": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._row_order = None
        self.current_row_idx = None
        self.prev_preds_buffer = None  # Keep for reward calculation
        self.prev_rows_buffer = None   # For network input (past)
        
        # Episode-level accumulators for DICE calculation
        self.episode_predictions = None
        self.episode_ground_truth = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._row_order = np.arange(self.H-1, -1, -1) if self.start_from_bottom else np.arange(0, self.H)
        self.current_row_idx = 0
        self.prev_preds_buffer = deque([np.zeros((self.W,), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        self.prev_rows_buffer = deque([np.zeros((self.W, self.C), dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len)
        
        # Initialize episode accumulators
        self.episode_predictions = np.zeros((self.H, self.W), dtype=np.float32)
        self.episode_ground_truth = self.mask.copy()
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self._row_order[self.current_row_idx]
        
        # Get future rows (lookahead)
        future_rows = []
        for i in range(1, self.future_len + 1):
            future_idx = self.current_row_idx + i
            if future_idx < self.H:
                future_row = self._row_order[future_idx]
                future_rows.append(self.image[future_row])
            else:
                # Pad with zeros if we're near the end
                future_rows.append(np.zeros((self.W, self.C), dtype=np.float32))
        
        return {
            "row_pixels": self.image[row],  # shape (W, C)
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),  # (history_len, W)
            "prev_rows": np.array(self.prev_rows_buffer, dtype=np.float32),  # (history_len, W, C)
            "future_rows": np.array(future_rows, dtype=np.float32),  # (future_len, W, C)
            "row_index": np.array([(row + 1) / self.H], dtype=np.float32),
        }

    def step(self, action):
        action = np.array(action, dtype=np.float32).clip(0, 1)
        row = self._row_order[self.current_row_idx]
        gt = self.mask[row]

        # ========== ENHANCED REWARD FOR VESSEL SEGMENTATION ==========
        
        # 1. WEIGHTED ACCURACY [BASE REWARD]: Combat class imbalance (reduced weight - saturating)
        # vessel_weight = 1.0  # Reduced from 3.0 since it saturates
        # background_weight = 0.3  # Further reduced background weight
        
        # vessel_mask = (gt == 1)
        # weighted_accuracy = np.where(
        #     vessel_mask,
        #     -vessel_weight * np.abs(action - gt),      # Penalty for missing vessels
        #     -background_weight * np.abs(action - gt))   # Penalty for background errors

        base_rewards = - self.base_coef * np.abs(action - gt)
        
        # Track per-row stats (for logging only, reward computed at episode end)
        true_positives = np.sum((action == 1) & (gt == 1))
        false_positives = np.sum((action == 1) & (gt == 0))
        false_negatives = np.sum((action == 0) & (gt == 1))
        
        # Store prediction for this row (for episode-level DICE)
        self.episode_predictions[row] = (action > 0.5).astype(np.float32)

        # 2. CONTINUITY REWARD: Enhanced to allow branching
        continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        
        if len(self.prev_preds_buffer) >= 1 and row < self.H - 1:
            prev_row = list(self.prev_preds_buffer)[-1]
            
            # For each active pixel, check if it connects to previous row
            curr_active_indices = np.where(action == 1)[0]
            # print(curr_active_indices)
            for idx in curr_active_indices:
                # Check 5-pixel neighborhood (allow diagonal connections for branching)
                left = max(0, idx - 2)
                right = min(self.W, idx + 3)
                
                # If connected to previous row, don't penalize
                if np.any(prev_row[left:right] == 1) and np.any(self.mask[row + 1][left:right] == 1):
                    continuity_rewards[idx] = 0
                else:
                    # Penalty for isolated pixels (not connected to structure)
                    continuity_rewards[idx] = -1.0  
            
            # Also consider inactive pixels that should maintain gaps
            # (don't penalize correctly predicting background between vessels)
            inactive_with_inactive_above = ((action == 0) & (np.convolve(prev_row, np.ones(3), mode='same') == 0))
            continuity_rewards[inactive_with_inactive_above] = 0.0 
        continuity_rewards *= self.continuity_coef

        # 3. GRADIENT-BASED REWARD: Encourage smooth intensity transitions
        gradient_rewards = np.zeros((self.W,), dtype=np.float32)
        
        if len(self.prev_preds_buffer) >= 1 and self.gradient_coef > 0:
            prev_pred = list(self.prev_preds_buffer)[-1]
            prev_active_indices = np.where(prev_pred == 1)[0]
            curr_active_indices = np.where(action == 1)[0]

            # Assign labels to groups of active pixels
            labels, num_comp = ndimage.label(prev_pred)
            
            # Current row intensity (squeeze channel dim if needed)
            curr_intensity = self.image[row].squeeze()  # (W,)
            # Previous row intensity
            prev_intensity = self.prev_rows_buffer[-1].squeeze()
            
            if len(prev_active_indices) > 0 and len(curr_active_indices) > 0:
                for idx in curr_active_indices:
                    # Find the closest active pixel in the previous row
                    distances = np.abs(prev_active_indices - idx)
                    closest_prev_idx = prev_active_indices[np.argmin(distances)]

                    if np.abs(closest_prev_idx - idx) > 5:
                        continue  # Skip if too far away
                    group = prev_intensity[labels == labels[closest_prev_idx]]
                    if len(group) <= 1:
                        continue
                    # Calculate intensity difference between current pixel and closest previous pixel
                    # intensity_diff = np.abs(curr_intensity[idx] - prev_intensity[closest_prev_idx])
                    belongs_to_group = curr_intensity[idx] > group.min() and curr_intensity[idx] < group.max()
                    intensity_diff = np.abs(curr_intensity[idx] - group.mean()) / (group.max() - group.min() + 1e-5) # DON'T USE NOW
                    # Reward small intensity differences (smooth transitions)
                    # Penalize large intensity jumps (indicates noise or discontinuity)
                    # intensity_diff is 0-1, so we reward when it's small
                    gradient_rewards[idx] = - int(belongs_to_group)
        gradient_rewards *= self.gradient_coef
        # print(gradient_rewards)

        conn_info = {}
        
        # Combine all rewards
        pixel_rewards = base_rewards + continuity_rewards + gradient_rewards
        reward = float(pixel_rewards.sum()) 
        
        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_rows_buffer.append(self.image[row].copy())
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H
        
        
        conn_info.update({
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
        })

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "gradient_rewards": gradient_rewards.astype(np.float32),
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
            "future_rows": np.zeros((self.future_len, self.W, self.C), dtype=np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
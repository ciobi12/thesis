import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from scipy import ndimage

class PathReconstructionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, image, 
                 mask, 
                 continuity_coef=0.1, 
                 thickness_coef=1,
                 connectivity_coef=1,
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
        self.continuity_coef = float(continuity_coef)
        self.thickness_coef = float(thickness_coef)  
        self.connectivity_coef = float(connectivity_coef)
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
        vessel_weight = 1.0  # Reduced from 3.0 since it saturates
        background_weight = 0.3  # Further reduced background weight
        
        vessel_mask = (gt == 1)
        weighted_accuracy = np.where(
            vessel_mask,
            -vessel_weight * np.abs(action - gt),      # Penalty for missing vessels
            -background_weight * np.abs(action - gt))   # Penalty for background errors
        
        
        # Track per-row stats (for logging only, reward computed at episode end)
        true_positives = np.sum((action == 1) & (gt == 1))
        false_positives = np.sum((action == 1) & (gt == 0))
        false_negatives = np.sum((action == 0) & (gt == 1))
        
        # Store prediction for this row (for episode-level DICE)
        self.episode_predictions[row] = (action > 0.5).astype(np.float32)

        # 2. CONTINUITY REWARD: Enhanced to allow branching
        continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        
        if len(self.prev_preds_buffer) >= 1:
            prev_row = list(self.prev_preds_buffer)[-1]
            
            # For each active pixel, check if it connects to previous row
            curr_active_indices = np.where(action == 1)[0]
            print(curr_active_indices)
            for idx in curr_active_indices:
                # Check 5-pixel neighborhood (allow diagonal connections for branching)
                left = max(0, idx - 2)
                right = min(self.W, idx + 3)
                
                # If connected to previous row, give positive reward
                if np.any(prev_row[left:right] == 1):
                    continuity_rewards[idx] = 0.8  # Reduced from 1.0
                else:
                    # STRONG penalty for isolated pixels (not connected to structure)
                    continuity_rewards[idx] = -3.0  # Increased from -2.0 to combat noise
            
            # Also consider inactive pixels that should maintain gaps
            # (don't penalize correctly predicting background between vessels)
            inactive_with_inactive_above = ((action == 0) & (np.convolve(prev_row, np.ones(5), mode='same') < 0.5))
            continuity_rewards[inactive_with_inactive_above] = 0.1  # Reduced from 0.2
        
        continuity_rewards *= self.continuity_coef
        
        # 3. THICKNESS CONTROL: Penalize over-thick predictions
        thickness_penalty = 0.0
        
        if np.sum(action== 1) > 0 and np.sum(gt == 1) > 0:
            from scipy.ndimage import binary_dilation, binary_erosion
            binary_pred = (action == 1).astype(np.uint8)
            binary_gt = (gt == 1).astype(np.uint8)
            
            # Compare local widths: count active pixels in sliding windows
            pred_width = np.sum(binary_pred)
            gt_width = np.sum(binary_gt)
            
            # STRONG penalty for being wider than ground truth
            if pred_width > gt_width:
                thickness_penalty = -3.0 * (pred_width - gt_width) / gt_width
            
            # Penalize isolated noisy pixels
            neighbor_count = ndimage.convolve(binary_pred.astype(float), np.ones(3), mode='constant')
            isolated_pixels = np.sum((binary_pred > 0) & (neighbor_count <= 1))
            if isolated_pixels > 0:
                thickness_penalty -= 2.0 * isolated_pixels
                
        # 4. TOPOLOGY: Prevent creating disconnected fragments - NOT USED NOW
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
        reward = float(pixel_rewards.sum()) \
                + self.thickness_coef * thickness_penalty \
                + self.connectivity_coef * connectivity_reward
        
        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_rows_buffer.append(self.image[row].copy())
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H
        
        # Calculate episode-level DICE coefficient at the end
        dice_coefficient = 0.0
        dice_reward = 0.0
        if terminated:
            pred_binary = (self.episode_predictions > 0.5).astype(np.float32)
            gt_binary = (self.episode_ground_truth > 0.5).astype(np.float32)
            
            intersection = np.sum(pred_binary * gt_binary)
            pred_sum = np.sum(pred_binary)
            gt_sum = np.sum(gt_binary)
            
            # DICE = 2 * |A ∩ B| / (|A| + |B|)
            dice_coefficient = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
            
            # Scale DICE reward to be significant (DICE is 0-1, scale to match other rewards)
            # Higher DICE = better segmentation = positive reward
            dice_reward = 100.0 * (dice_coefficient - 0.5)  # Centered around 0.5 DICE
            reward += dice_reward
        
        conn_info.update({
            "thickness_penalty": thickness_penalty,
            "true_positives": float(true_positives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
            "dice_coefficient": dice_coefficient,
            "dice_reward": dice_reward,
        })

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
            "future_rows": np.zeros((self.future_len, self.W, self.C), dtype=np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
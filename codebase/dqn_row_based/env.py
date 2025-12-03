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

        # Base rewards
        base_rewards = -np.abs(action - gt)

        # Existing continuity (keep your current strategy)
        continuity_rewards = np.zeros((self.W,), dtype=np.float32)
        decay_factor = self.continuity_decay_factor
        for i, prev_row in enumerate(reversed(list(self.prev_preds_buffer))):
            weight = (decay_factor ** i)
            continuity_rewards += -weight * np.abs(action - prev_row)
        continuity_rewards *= self.continuity_coef

        # ========== NEW: CONNECTIVITY-AWARE REWARDS ==========
        connectivity_reward = 0.0
        conn_info = {}
        
        if len(self.prev_preds_buffer) >= 2:  # Need history for component analysis
            # Build mini-window (last 3 rows + current)
            window_rows = list(self.prev_preds_buffer)[-3:]
            window_rows.append(action)
            window = np.vstack(window_rows)
            binary_window = (window > 0.5).astype(np.uint8)
            
            # Count connected components in 2D window
            labeled, num_components = ndimage.label(binary_window, structure=np.ones((3, 3)))
            
            # STRONG penalty for fragmentation (scales with component count)
            if num_components > 1:
                fragmentation_penalty = -2.0 * (num_components - 1)  # -2 per extra component
                connectivity_reward += fragmentation_penalty
                conn_info["fragmentation_penalty"] = fragmentation_penalty
                conn_info["num_components"] = num_components
            
            # Bridge bonus: reward pixels that connect previous disconnected regions
            prev_row_binary = (list(self.prev_preds_buffer)[-1] > 0.5).astype(np.uint8)
            labeled_prev, num_prev_components = ndimage.label(prev_row_binary)
            
            if num_prev_components > 1:
                curr_active = np.where(action > 0.5)[0]
                bridges_formed = 0
                
                for pixel_idx in curr_active:
                    # Check 3-pixel neighborhood in previous row
                    neighborhood_start = max(0, pixel_idx - 1)
                    neighborhood_end = min(self.W, pixel_idx + 2)
                    neighborhood_labels = labeled_prev[neighborhood_start:neighborhood_end]
                    
                    # Count unique components this pixel connects to
                    connected_to = set(neighborhood_labels[neighborhood_labels > 0])
                    if len(connected_to) >= 2:
                        bridges_formed += 1
                
                if bridges_formed > 0:
                    bridge_bonus = 1.5 * bridges_formed  # Strong positive signal
                    connectivity_reward += bridge_bonus
                    conn_info["bridge_bonus"] = bridge_bonus
                    conn_info["bridges_formed"] = bridges_formed
            
            # Compactness reward: penalize sparse active regions
            active_pixels = np.where(action > 0.5)[0]
            if len(active_pixels) > 1:
                span = active_pixels[-1] - active_pixels[0] + 1
                density = len(active_pixels) / span
                
                # Reward high density (compact paths), penalize low density (fragmented)
                if density < 0.4:
                    compactness_penalty = -1.0 * (0.4 - density) * 5  # Scale penalty
                    connectivity_reward += compactness_penalty
                    conn_info["compactness_penalty"] = compactness_penalty
                    conn_info["path_density"] = density

        # Total pixel rewards (keep per-pixel for compatibility)
        pixel_rewards = base_rewards + continuity_rewards
        
        # Add connectivity as episode-level reward (not per-pixel)
        reward = float(pixel_rewards.sum()) + connectivity_reward

        # Update history
        self.prev_preds_buffer.append(action.copy())
        self.prev_rows_buffer.append(self.image[row].copy())
        self.current_row_idx += 1
        terminated = self.current_row_idx >= self.H

        info = {
            "pixel_rewards": pixel_rewards.astype(np.float32),
            "base_rewards": base_rewards.astype(np.float32),
            "continuity_rewards": continuity_rewards.astype(np.float32),
            "connectivity_reward": connectivity_reward,  # NEW
            "row_index": row,
        }
        info.update(conn_info)  # Add detailed connectivity info
        
        return (self._get_obs() if not terminated else self._terminal_obs()), reward, terminated, False, info

    def _terminal_obs(self):
        # After last row, return a zeroed observation (won't be used)
        return {
            "row_pixels": np.zeros((self.W, self.C), dtype=np.float32),
            "prev_preds": np.array(self.prev_preds_buffer, dtype=np.float32),
            "prev_rows": np.array(self.prev_rows_buffer, dtype=np.float32),
            "row_index": np.array([1.0], dtype=np.float32),
        }
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))

def obs_to_tensor(obs, device, as_tensor = False):
    if as_tensor:
        x = np.concatenate([obs["slice_pixels"].reshape(-1), obs["prev_slices"].reshape(-1), obs["slice_index"]], axis=0).astype(np.float32)
        return torch.from_numpy(x).to(device)

    return {
        "slice_pixels": torch.tensor(obs["slice_pixels"], dtype=torch.float32, device=device),
        "prev_preds": torch.tensor(obs["prev_preds"], dtype=torch.float32, device=device),
        "prev_slices": torch.tensor(obs["prev_slices"], dtype=torch.float32, device=device),
        "slice_index": torch.tensor(obs["slice_index"], dtype=torch.float32, device=device),
    }

def batch_obs_to_tensor(obs_list, device):
    """Convert list of observations to batched tensors efficiently."""
    slice_pixels = torch.tensor(np.stack([o["slice_pixels"] for o in obs_list]), dtype=torch.float32, device=device)
    prev_preds = torch.tensor(np.stack([o["prev_preds"] for o in obs_list]), dtype=torch.float32, device=device)
    prev_slices = torch.tensor(np.stack([o["prev_slices"] for o in obs_list]), dtype=torch.float32, device=device)
    slice_index = torch.tensor(np.stack([o["slice_index"] for o in obs_list]), dtype=torch.float32, device=device)
    return {
        "slice_pixels": slice_pixels,
        "prev_preds": prev_preds,
        "prev_slices": prev_slices,
        "slice_index": slice_index,
    }

class PerPixelCNN(nn.Module):
    def __init__(self, H, W, history_len=3, hidden_channels=32):
        super().__init__()
        self.H = H
        self.W = W
        self.history_len = history_len
        self.hidden_channels = hidden_channels
        in_channels = 1 + history_len  # current slice + K previous slices
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1)
        )

    def forward(self, obs):
        # obs["slice_pixels"]: (B, H, W) or (H, W), obs["prev_slices"]: (B, history_len, H, W) or (history_len, H, W)
        if obs["slice_pixels"].dim() == 2:  # Single observation
            x_img = obs["slice_pixels"].unsqueeze(0)  # (1, H, W)
            x_hist = obs["prev_slices"]  # (history_len, H, W)
            x = torch.cat([x_img, x_hist], dim=0).unsqueeze(0)  # (1, 1+history_len, H, W)
            q = self.conv(x).squeeze(0).permute(1, 2, 0)  # (H, W, 2)
        else:  # Batched observations
            B = obs["slice_pixels"].size(0)
            x_img = obs["slice_pixels"].unsqueeze(1)  # (B, 1, H, W)
            x_hist = obs["prev_slices"]  # (B, history_len, H, W)
            x = torch.cat([x_img, x_hist], dim=1)  # (B, 1+history_len, H, W)
            q = self.conv(x).permute(0, 2, 3, 1)  # (B, H, W, 2)
        return q

class PerPixelCNNWithHistory(nn.Module):
    """Enhanced architecture that explicitly processes historical and future context.
    
    Args:
        input_channels: Number of input channels (usually 1 for grayscale)
        history_len: Number of previous prediction slices to use
        future_len: Number of future slices to look ahead
        height, width: Spatial dimensions of input slices
        small_input: If True, use lighter architecture for 32x32 or 64x64 inputs
    """
    def __init__(self, input_channels, history_len, height, width, future_len=3, small_input=False):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.height = height
        self.width = width
        self.small_input = small_input
        
        if small_input:
            # Lighter architecture for small inputs (32x32, 64x64)
            # Smaller kernels, fewer channels to prevent overfitting
            slice_ch = 32
            hist_ch = 16
            future_ch = 16
            
            self.slice_encoder = nn.Sequential(
                nn.Conv2d(input_channels, slice_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            
            self.history_encoder = nn.Sequential(
                nn.Conv2d(history_len, hist_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hist_ch, hist_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            
            self.future_encoder = nn.Sequential(
                nn.Conv2d(future_len, future_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(future_ch, future_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            
            self.attention = nn.Sequential(
                nn.Conv2d(slice_ch + hist_ch + future_ch, slice_ch, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(slice_ch, slice_ch, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.decision = nn.Sequential(
                nn.Conv2d(slice_ch + hist_ch + future_ch, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(0.1),  # Light regularization
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=1)
            )
        else:
            # Original architecture for larger inputs (100x64, 128x128, etc.)
            slice_ch = 64
            hist_ch = 32
            future_ch = 32
            
            self.slice_encoder = nn.Sequential(
                nn.Conv2d(input_channels, slice_ch, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv2d(slice_ch, slice_ch, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            
            self.history_encoder = nn.Sequential(
                nn.Conv2d(history_len, hist_ch, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv2d(hist_ch, hist_ch, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            
            self.future_encoder = nn.Sequential(
                nn.Conv2d(future_len, future_ch, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv2d(future_ch, future_ch, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            
            self.attention = nn.Sequential(
                nn.Conv2d(slice_ch + hist_ch + future_ch, slice_ch, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(slice_ch, slice_ch, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.decision = nn.Sequential(
                nn.Conv2d(slice_ch + hist_ch + future_ch, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1)
            )
        
        # Store channel sizes for forward pass
        self.slice_ch = slice_ch
        self.hist_ch = hist_ch
        self.future_ch = future_ch
    
    def forward(self, slice_pixels, prev_slices, future_slices=None):
        """
        slice_pixels: (batch, height, width) -> needs unsqueeze for channel dim
        prev_slices: (batch, history_len, height, width) - previous image slices
        future_slices: (batch, future_len, height, width) - optional, zeros if not provided
        """
        B = slice_pixels.size(0) if slice_pixels.dim() > 2 else 1
        
        # Add channel dimension to slice_pixels
        x_slice = slice_pixels.unsqueeze(1)  # (B, 1, H, W)
        x_hist = prev_slices  # (B, history_len, H, W)
        
        # Handle future slices (default to zeros if not provided)
        if future_slices is None:
            x_future = torch.zeros(B, self.future_len, self.height, self.width, 
                                   device=slice_pixels.device, dtype=slice_pixels.dtype)
        else:
            x_future = future_slices  # (B, future_len, H, W)
        
        # Encode current slice
        slice_features = self.slice_encoder(x_slice)  # (B, slice_ch, H, W)
        
        # Encode history
        hist_features = self.history_encoder(x_hist)  # (B, hist_ch, H, W)
        
        # Encode future
        future_features = self.future_encoder(x_future)  # (B, future_ch, H, W)
        
        # Combine all for attention
        combined = torch.cat([slice_features, hist_features, future_features], dim=1)
        attention_weights = self.attention(combined)  # (B, slice_ch, H, W)
        
        # Apply attention to slice features
        attended_slice = slice_features * attention_weights  # (B, slice_ch, H, W)
        
        # Final decision with attended slice, history, and future
        final_features = torch.cat([attended_slice, hist_features, future_features], dim=1)
        q_values = self.decision(final_features)  # (B, 2, H, W)
        
        return q_values.permute(0, 2, 3, 1)  # (B, H, W, 2)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.push if hasattr(self.buffer, "push") else None
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Stack fields
        return batch

    def __len__(self):
        return len(self.buffer)

def epsilon_greedy_action(q_values, epsilon):
    # q_values: (H, W, 2) or (B, H, W, 2) tensor
    if q_values.dim() == 3:  # Single observation
        H, W, _ = q_values.shape
        if random.random() < epsilon:
            return torch.randint(low=0, high=2, size=(H, W), dtype=torch.int64, device=q_values.device)
        else:
            return q_values.argmax(dim=-1)  # (H, W)
    else:  # Batched
        B, H, W, _ = q_values.shape
        mask = torch.rand(B, device=q_values.device) < epsilon
        random_actions = torch.randint(low=0, high=2, size=(B, H, W), dtype=torch.int64, device=q_values.device)
        greedy_actions = q_values.argmax(dim=-1)  # (B, H, W)
        return torch.where(mask.unsqueeze(1).unsqueeze(2), random_actions, greedy_actions)
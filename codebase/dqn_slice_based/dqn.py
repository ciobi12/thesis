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
    """Enhanced architecture that explicitly processes historical context"""
    def __init__(self, input_channels, history_len, height, width, dropout_rate=0.2):
        super().__init__()
        self.history_len = history_len
        self.height = height
        self.width = width
        self.dropout_rate = dropout_rate
        
        # Current slice encoder
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
        )
        
        # History encoder (processes previous predictions)
        self.history_encoder = nn.Sequential(
            nn.Conv2d(history_len, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
        )
        
        # Cross-attention between current and history
        self.attention = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final decision layers
        self.decision = nn.Sequential(
            nn.Conv2d(64 + 32, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
    
    def forward(self, slice_pixels, prev_preds):
        """
        slice_pixels: (batch, height, width) -> needs unsqueeze for channel dim
        prev_preds: (batch, history_len, height, width)
        """
        # Add channel dimension to slice_pixels
        x_slice = slice_pixels.unsqueeze(1)  # (B, 1, H, W)
        x_hist = prev_preds  # (B, history_len, H, W)
        
        # Encode current slice
        slice_features = self.slice_encoder(x_slice)  # (B, 64, H, W)
        
        # Encode history
        hist_features = self.history_encoder(x_hist)  # (B, 32, H, W)
        
        # Combine for attention
        combined = torch.cat([slice_features, hist_features], dim=1)  # (B, 96, H, W)
        attention_weights = self.attention(combined)  # (B, 64, H, W)
        
        # Apply attention to slice features
        attended_slice = slice_features * attention_weights  # (B, 64, H, W)
        
        # Final decision with both attended slice and history
        final_features = torch.cat([attended_slice, hist_features], dim=1)  # (B, 96, H, W)
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
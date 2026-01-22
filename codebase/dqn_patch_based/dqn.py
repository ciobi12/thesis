import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))


def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    patch_pixels = torch.tensor(obs["patch_pixels"], dtype=torch.float32, device=device).permute(2, 0, 1)  # (C, N, N)
    prev_preds = torch.tensor(obs["prev_preds"], dtype=torch.float32, device=device)  # (history_len, N, N)
    prev_patches = torch.tensor(obs["prev_patches"], dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # (history_len, C, N, N)
    future_patches = torch.tensor(obs["future_patches"], dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # (future_len, C, N, N)
    patch_coords = torch.tensor(obs["patch_coords"], dtype=torch.float32, device=device)  # (2,)
    return patch_pixels, prev_patches, future_patches


def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    patch_pixels = torch.tensor(
        np.stack([o["patch_pixels"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)  # (B, C, N, N)
    prev_preds = torch.tensor(
        np.stack([o["prev_preds"] for o in obs_list]), dtype=torch.float32, device=device
    )  # (B, history_len, N, N)
    prev_patches = torch.tensor(
        np.stack([o["prev_patches"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 1, 4, 2, 3)  # (B, history_len, C, N, N)
    future_patches = torch.tensor(
        np.stack([o["future_patches"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 1, 4, 2, 3)  # (B, future_len, C, N, N)
    patch_coords = torch.tensor(
        np.stack([o["patch_coords"] for o in obs_list]), dtype=torch.float32, device=device
    )  # (B, 2)
    return patch_pixels, prev_patches, future_patches


class PerPatchCNN(nn.Module):
    """Per-patch pixel-level Q-network: outputs (2, N, N) Q-values (for actions 0/1 per pixel)."""
    def __init__(self, N, C, hidden_channels=32):
        super().__init__()
        self.N = N
        self.C = C
        in_ch = C + 1  # image channels + prev_pred
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),  # per-pixel Q-values for 0/1
        )

    def forward(self, obs):
        # obs["patch_pixels"]: (C, N, N), obs["prev_pred"]: (1, N, N)
        # Concatenate along channel dimension before adding batch dim
        x = torch.cat([obs["patch_pixels"], obs["prev_pred"]], dim=0)  # (in_ch, N, N)
        x = x.unsqueeze(0)  # (1, in_ch, N, N)
        q = self.net(x)  # (B, 2, N, N)
        return q.squeeze(0)  # (2, N, N) for unbatched case


class PerPatchCNNWithHistory(nn.Module):
    """Enhanced architecture that processes historical and future context for patches.
    
    The network sees:
    - Current patch pixels
    - Previous K patches (history_len) image pixels
    - Future K patches (future_len) image pixels (lookahead)
    """
    def __init__(self, input_channels, history_len, future_len, patch_size):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.input_channels = input_channels
        self.patch_size = patch_size
        
        # Current patch encoder
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # History encoder (processes previous patch pixels)
        self.history_encoder = nn.Sequential(
            nn.Conv2d(history_len * input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Future encoder (processes upcoming patch pixels for lookahead)
        self.future_encoder = nn.Sequential(
            nn.Conv2d(future_len * input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Cross-attention between current and temporal context (history + future)
        self.attention = nn.Sequential(
            nn.Conv2d(64 + 32 + 32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final decision layers (includes history and future features)
        self.decision = nn.Sequential(
            nn.Conv2d(64 + 32 + 32, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
    
    def forward(self, patch_pixels, prev_patches, future_patches):
        """
        patch_pixels: (batch, C, N, N)
        prev_patches: (batch, history_len, C, N, N) - previous patch image pixels
        future_patches: (batch, future_len, C, N, N)
        
        Returns: (batch, N, N, 2) Q-values per pixel
        """
        # Handle single observation (no batch dimension)
        if patch_pixels.dim() == 3:
            patch_pixels = patch_pixels.unsqueeze(0)
            prev_patches = prev_patches.unsqueeze(0)
            future_patches = future_patches.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B = patch_pixels.size(0)
        N = patch_pixels.size(-1)
        
        # Flatten history patches: (B, history_len, C, N, N) -> (B, history_len*C, N, N)
        x_hist = prev_patches.reshape(B, self.history_len * self.input_channels, N, N)
        
        # Flatten future patches: (B, future_len, C, N, N) -> (B, future_len*C, N, N)
        x_future = future_patches.reshape(B, self.future_len * self.input_channels, N, N)
        
        # Encode current patch
        patch_features = self.patch_encoder(patch_pixels)  # (B, 64, N, N)
        
        # Encode history (past patch pixels)
        hist_features = self.history_encoder(x_hist)  # (B, 32, N, N)
        
        # Encode future (upcoming patch pixels)
        future_features = self.future_encoder(x_future)  # (B, 32, N, N)
        
        # Combine all for attention
        combined = torch.cat([patch_features, hist_features, future_features], dim=1)  # (B, 128, N, N)
        attention_weights = self.attention(combined)  # (B, 64, N, N)
        
        # Apply attention to patch features
        attended_patch = patch_features * attention_weights  # (B, 64, N, N)
        
        # Final decision with attended patch, history, and future
        final_features = torch.cat([attended_patch, hist_features, future_features], dim=1)  # (B, 128, N, N)
        q_values = self.decision(final_features)  # (B, 2, N, N)
        
        # Permute to (B, N, N, 2)
        q_values = q_values.permute(0, 2, 3, 1)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)  # (N, N, 2)
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def epsilon_greedy_action(q_values, epsilon):
    """
    q_values: tensor (N, N, 2) or (B, N, N, 2). 
    Returns LongTensor actions (N, N) or (B, N, N) in {0,1}.
    """
    if q_values.dim() == 3:  # Single observation (N, N, 2)
        N = q_values.shape[0]
        if random.random() < epsilon:
            return torch.randint(low=0, high=2, size=(N, N), dtype=torch.int64, device=q_values.device)
        else:
            return q_values.argmax(dim=-1)  # (N, N)
    else:  # Batched (B, N, N, 2)
        B, N, _, _ = q_values.shape
        mask = torch.rand(B, device=q_values.device) < epsilon
        random_actions = torch.randint(low=0, high=2, size=(B, N, N), dtype=torch.int64, device=q_values.device)
        greedy_actions = q_values.argmax(dim=-1)  # (B, N, N)
        return torch.where(mask.unsqueeze(1).unsqueeze(2), random_actions, greedy_actions)

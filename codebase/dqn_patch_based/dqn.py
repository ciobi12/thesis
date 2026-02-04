import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    patch_pixels = torch.tensor(obs["patch_pixels"], dtype=torch.float32, device=device).permute(2, 0, 1)  # (C, N, N)
    below_patches = torch.tensor(obs["below_patches"], dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # (3, C, N, N)
    above_patches = torch.tensor(obs["above_patches"], dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # (3, C, N, N)
    neighbor_masks = torch.tensor(obs["neighbor_masks"], dtype=torch.float32, device=device)  # (4, N, N)
    return patch_pixels, below_patches, above_patches, neighbor_masks


def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    patch_pixels = torch.tensor(
        np.stack([o["patch_pixels"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)  # (B, C, N, N)
    below_patches = torch.tensor(
        np.stack([o["below_patches"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 1, 4, 2, 3)  # (B, 3, C, N, N)
    above_patches = torch.tensor(
        np.stack([o["above_patches"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 1, 4, 2, 3)  # (B, 3, C, N, N)
    neighbor_masks = torch.tensor(
        np.stack([o["neighbor_masks"] for o in obs_list]), dtype=torch.float32, device=device
    )  # (B, 4, N, N)
    return patch_pixels, below_patches, above_patches, neighbor_masks

class NeighborContextCNN(nn.Module):
    """Enhanced architecture that uses spatial neighbor context.
    
    The network sees:
    - Current patch pixels
    - 3 patches below (bottom-left, directly below, bottom-right)
    - 3 patches above (top-left, directly above, top-right)
    - Predicted masks of left + 3 below patches (for continuity awareness)
    """
    def __init__(self, input_channels, patch_size):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        
        # Current patch encoder
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Below patches encoder (3 patches: bottom-left, below, bottom-right)
        self.below_encoder = nn.Sequential(
            nn.Conv2d(3 * input_channels, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Above patches encoder (3 patches: top-left, above, top-right)
        self.above_encoder = nn.Sequential(
            nn.Conv2d(3 * input_channels, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Neighbor masks encoder (4 masks: left + 3 below)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Cross-attention between current patch and all context
        # 64 (patch) + 48 (below) + 48 (above) + 32 (masks) = 192
        self.attention = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final decision layers
        self.decision = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
    
    def forward(self, patch_pixels, below_patches, above_patches, neighbor_masks):
        """
        patch_pixels: (batch, C, N, N) or (C, N, N)
        below_patches: (batch, 3, C, N, N) or (3, C, N, N)
        above_patches: (batch, 3, C, N, N) or (3, C, N, N)
        neighbor_masks: (batch, 4, N, N) or (4, N, N)
        
        Returns: (batch, N, N, 2) Q-values per pixel
        """
        # Handle single observation (no batch dimension)
        if patch_pixels.dim() == 3:
            patch_pixels = patch_pixels.unsqueeze(0)
            below_patches = below_patches.unsqueeze(0)
            above_patches = above_patches.unsqueeze(0)
            neighbor_masks = neighbor_masks.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B = patch_pixels.size(0)
        N = patch_pixels.size(-1)
        
        # Flatten below patches: (B, 3, C, N, N) -> (B, 3*C, N, N)
        x_below = below_patches.reshape(B, 3 * self.input_channels, N, N)
        
        # Flatten above patches: (B, 3, C, N, N) -> (B, 3*C, N, N)
        x_above = above_patches.reshape(B, 3 * self.input_channels, N, N)
        
        # Encode current patch
        patch_features = self.patch_encoder(patch_pixels)  # (B, 64, N, N)
        
        # Encode below context
        below_features = self.below_encoder(x_below)  # (B, 48, N, N)
        
        # Encode above context
        above_features = self.above_encoder(x_above)  # (B, 48, N, N)
        
        # Encode neighbor masks
        mask_features = self.mask_encoder(neighbor_masks)  # (B, 32, N, N)
        
        # Combine all features
        combined = torch.cat([
            patch_features, below_features, above_features, mask_features
        ], dim=1)  # (B, 192, N, N)
        
        # Apply attention
        attention_weights = self.attention(combined)  # (B, 64, N, N)
        attended_patch = patch_features * attention_weights  # (B, 64, N, N)
        
        # Reconstruct combined with attended patch
        final_features = torch.cat([
            attended_patch, below_features, above_features, mask_features
        ], dim=1)  # (B, 192, N, N)
        
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

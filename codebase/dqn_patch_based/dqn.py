import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


def obs_to_tensor(obs, device):
    """Convert observation dict to tensors on device."""
    patch_pixels = torch.tensor(obs["patch_pixels"], dtype=torch.float32, device=device).permute(2, 0, 1)  # (C, N, N)
    neighbor_patches = torch.tensor(obs["neighbor_patches"], dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # (8, C, N, N)
    neighbor_masks = torch.tensor(obs["neighbor_masks"], dtype=torch.float32, device=device)  # (8, N, N)
    neighbor_valid = torch.tensor(obs["neighbor_valid"], dtype=torch.float32, device=device)  # (8,)
    return patch_pixels, neighbor_patches, neighbor_masks, neighbor_valid


def batch_obs_to_tensor(obs_list, device):
    """Convert list of observation dicts to batched tensors."""
    patch_pixels = torch.tensor(
        np.stack([o["patch_pixels"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)  # (B, C, N, N)
    neighbor_patches = torch.tensor(
        np.stack([o["neighbor_patches"] for o in obs_list]), dtype=torch.float32, device=device
    ).permute(0, 1, 4, 2, 3)  # (B, 8, C, N, N)
    neighbor_masks = torch.tensor(
        np.stack([o["neighbor_masks"] for o in obs_list]), dtype=torch.float32, device=device
    )  # (B, 8, N, N)
    neighbor_valid = torch.tensor(
        np.stack([o["neighbor_valid"] for o in obs_list]), dtype=torch.float32, device=device
    )  # (B, 8)
    return patch_pixels, neighbor_patches, neighbor_masks, neighbor_valid


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


class NeighborContextCNN(nn.Module):
    """Enhanced architecture that uses all 8 spatial neighbor patches.
    
    The network sees:
    - Current patch pixels
    - All 8 neighbor patches: [left, right, top-left, top, top-right, bottom-left, bottom, bottom-right]
    - Predicted masks of all 8 neighbors (for continuity awareness)
    - Validity mask indicating which neighbors are in bounds
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
        
        # All 8 neighbor patches encoder - process them together
        # 8 neighbors * C channels = 8*C input channels
        self.neighbor_encoder = nn.Sequential(
            nn.Conv2d(8 * input_channels, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Neighbor masks encoder (8 masks, one for each neighbor)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(8, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Edge-aware encoder: special focus on boundary pixels
        # Input: current patch + attention to edge regions
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Cross-attention between current patch and all context
        # 64 (patch) + 96 (neighbors) + 48 (masks) + 32 (edge) = 240
        self.attention = nn.Sequential(
            nn.Conv2d(240, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final decision layers
        self.decision = nn.Sequential(
            nn.Conv2d(240, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
    
    def forward(self, patch_pixels, neighbor_patches, neighbor_masks, neighbor_valid):
        """
        patch_pixels: (batch, C, N, N) or (C, N, N)
        neighbor_patches: (batch, 8, C, N, N) or (8, C, N, N)
        neighbor_masks: (batch, 8, N, N) or (8, N, N)
        neighbor_valid: (batch, 8) or (8,) - which neighbors are valid
        
        Returns: (batch, N, N, 2) Q-values per pixel
        """
        # Handle single observation (no batch dimension)
        if patch_pixels.dim() == 3:
            patch_pixels = patch_pixels.unsqueeze(0)
            neighbor_patches = neighbor_patches.unsqueeze(0)
            neighbor_masks = neighbor_masks.unsqueeze(0)
            neighbor_valid = neighbor_valid.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B = patch_pixels.size(0)
        N = patch_pixels.size(-1)
        
        # Mask out invalid neighbors by zeroing them
        # neighbor_valid: (B, 8) -> (B, 8, 1, 1, 1) for broadcasting
        validity_mask_patches = neighbor_valid.view(B, 8, 1, 1, 1)
        validity_mask_masks = neighbor_valid.view(B, 8, 1, 1)
        
        neighbor_patches = neighbor_patches * validity_mask_patches
        neighbor_masks = neighbor_masks * validity_mask_masks
        
        # Flatten neighbor patches: (B, 8, C, N, N) -> (B, 8*C, N, N)
        x_neighbors = neighbor_patches.reshape(B, 8 * self.input_channels, N, N)
        
        # Encode current patch
        patch_features = self.patch_encoder(patch_pixels)  # (B, 64, N, N)
        
        # Encode all 8 neighbors
        neighbor_features = self.neighbor_encoder(x_neighbors)  # (B, 96, N, N)
        
        # Encode neighbor masks
        mask_features = self.mask_encoder(neighbor_masks)  # (B, 48, N, N)
        
        # Edge-aware features from current patch
        edge_features = self.edge_encoder(patch_pixels)  # (B, 32, N, N)
        
        # Combine all features
        combined = torch.cat([
            patch_features, neighbor_features, mask_features, edge_features
        ], dim=1)  # (B, 240, N, N)
        
        # Apply attention
        attention_weights = self.attention(combined)  # (B, 64, N, N)
        attended_patch = patch_features * attention_weights  # (B, 64, N, N)
        
        # Reconstruct combined with attended patch
        final_features = torch.cat([
            attended_patch, neighbor_features, mask_features, edge_features
        ], dim=1)  # (B, 240, N, N)
        
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

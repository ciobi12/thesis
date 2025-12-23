"""
Enhanced DQN architecture for large real volumes.
Key improvements:
- Multi-scale feature extraction
- Residual connections
- Attention mechanism
- Mixed precision support
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention (SE-like) module."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleEncoder(nn.Module):
    """Multi-scale feature encoder with residual connections."""
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # Encoder blocks (no downsampling to maintain resolution)
        self.enc1 = ResidualBlock(base_channels, base_channels * 2)
        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.enc3 = ResidualBlock(base_channels * 4, base_channels * 4)
        
        # Attention
        self.channel_attn = ChannelAttention(base_channels * 4)
        self.spatial_attn = SpatialAttention(base_channels * 4)
        
        self.out_channels = base_channels * 4
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class PerPixelDQNV2(nn.Module):
    """
    Enhanced per-pixel DQN for real CT volumes.
    
    Features:
    - Multi-scale feature extraction
    - Separate slice and history encoders
    - Attention-based fusion
    - Residual connections
    """
    def __init__(self, history_len=5, base_channels=32):
        super().__init__()
        self.history_len = history_len
        
        # Slice encoder (processes current slice)
        self.slice_encoder = MultiScaleEncoder(1, base_channels)
        
        # History encoder (processes previous predictions)
        self.history_encoder = nn.Sequential(
            nn.Conv2d(history_len, base_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )
        
        # Previous slices encoder
        self.prev_slices_encoder = nn.Sequential(
            nn.Conv2d(history_len, base_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels, base_channels * 2),
        )
        
        # Fusion with attention
        fusion_channels = self.slice_encoder.out_channels + base_channels * 2 + base_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, base_channels * 4, 1),
            nn.ReLU(),
            ChannelAttention(base_channels * 4),
        )
        
        # Decision head (outputs Q-values)
        self.decision = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2, 1)  # Q-values for 0 and 1
        )
    
    def forward(self, slice_pixels, prev_preds, prev_slices=None):
        """
        Args:
            slice_pixels: (B, H, W) current slice
            prev_preds: (B, history_len, H, W) previous predictions
            prev_slices: (B, history_len, H, W) previous slice pixels (optional)
        
        Returns:
            Q-values: (B, H, W, 2)
        """
        # Handle single observation
        if slice_pixels.dim() == 2:
            slice_pixels = slice_pixels.unsqueeze(0)
            prev_preds = prev_preds.unsqueeze(0)
            if prev_slices is not None:
                prev_slices = prev_slices.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Add channel dim
        x_slice = slice_pixels.unsqueeze(1)  # (B, 1, H, W)
        
        # Encode slice
        slice_features = self.slice_encoder(x_slice)  # (B, C, H, W)
        
        # Encode history
        hist_features = self.history_encoder(prev_preds)  # (B, C', H, W)
        
        # Encode previous slices if provided
        if prev_slices is not None:
            prev_slice_features = self.prev_slices_encoder(prev_slices)
        else:
            prev_slice_features = torch.zeros_like(hist_features)
        
        # Fuse features
        combined = torch.cat([slice_features, hist_features, prev_slice_features], dim=1)
        fused = self.fusion(combined)
        
        # Decision
        q_values = self.decision(fused)  # (B, 2, H, W)
        q_values = q_values.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
        
        return q_values


class UNetDQN(nn.Module):
    """
    U-Net style architecture for per-pixel Q-values.
    Better for capturing multi-scale context in large images.
    """
    def __init__(self, history_len=5, base_channels=32):
        super().__init__()
        self.history_len = history_len
        
        in_channels = 1 + history_len  # current slice + history
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        # Output (Q-values)
        self.out = nn.Conv2d(base_channels, 2, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, slice_pixels, prev_preds, prev_slices=None):
        # Handle single observation
        if slice_pixels.dim() == 2:
            slice_pixels = slice_pixels.unsqueeze(0)
            prev_preds = prev_preds.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, H, W = slice_pixels.shape
        
        # Combine inputs
        x_slice = slice_pixels.unsqueeze(1)  # (B, 1, H, W)
        x = torch.cat([x_slice, prev_preds], dim=1)  # (B, 1+history, H, W)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        # Handle size mismatch
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Output
        q_values = self.out(d1)  # (B, 2, H, W)
        q_values = q_values.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better sample efficiency."""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def push(self, *args):
        transition = Transition(*args)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        size = len(self.buffer)
        priorities = self.priorities[:size]
        
        # Compute probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(size, batch_size, replace=False, p=probs)
        
        # Importance sampling weights
        weights = (size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    """Standard replay buffer."""
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
    """Epsilon-greedy action selection."""
    if q_values.dim() == 3:  # Single observation (H, W, 2)
        H, W, _ = q_values.shape
        if random.random() < epsilon:
            return torch.randint(0, 2, (H, W), dtype=torch.int64, device=q_values.device)
        else:
            return q_values.argmax(dim=-1)
    else:  # Batched (B, H, W, 2)
        B, H, W, _ = q_values.shape
        mask = torch.rand(B, device=q_values.device) < epsilon
        random_actions = torch.randint(0, 2, (B, H, W), dtype=torch.int64, device=q_values.device)
        greedy_actions = q_values.argmax(dim=-1)
        return torch.where(mask.unsqueeze(1).unsqueeze(2), random_actions, greedy_actions)

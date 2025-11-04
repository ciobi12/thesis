import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))

def obs_to_tensor(obs, device):
    return {
        "patch_pixels": torch.tensor(obs["patch_pixels"], dtype=torch.float32, device=device).permute(2, 0, 1),  # (C,N,N)
        "prev_pred": torch.tensor(obs["prev_pred"], dtype=torch.float32, device=device).unsqueeze(0),             # (1,N,N)
        "patch_coords": torch.tensor(obs["patch_coords"], dtype=torch.float32, device=device),                   # (2,)
    }

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
    """q_values: tensor (2, N, N). Returns LongTensor actions (N, N) in {0,1}."""
    N = q_values.shape[-1]
    if random.random() < epsilon:
        return torch.randint(low=0, high=2, size=(N, N), dtype=torch.int64, device=q_values.device)
    else:
        return q_values.argmax(dim=0)  # (N, N)

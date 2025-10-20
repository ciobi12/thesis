import numpy as np
import torch
import torch.nn as nn

from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))

def obs_to_tensor(obs, device):
    # Concatenate features: [row_pixels, prev_pred, row_index]
    x = np.concatenate([obs["row_pixels"], obs["prev_pred"], obs["row_index"]], axis=0).astype(np.float32)
    return torch.from_numpy(x).to(device)  # shape (2W + 1,)

class PerPixelDQN(nn.Module):
    """
    Input: concatenated features of length (2W + 1)
    Output: Q-values per pixel: shape (W, 2) => Q(a=0), Q(a=1)
    """
    def __init__(self, W, hidden=256):
        super().__init__()
        self.W = W
        in_dim = 2*W + 1
        out_dim = W * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (B, 2W+1)
        q = self.net(x)  # (B, W*2)
        q = q.view(-1, self.W, 2)  # (B, W, 2)
        return q

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
    # q_values: (W, 2) tensor
    W = q_values.size(0)
    if random.random() < epsilon:
        return torch.randint(low=0, high=2, size=(W,), dtype=torch.int64, device=q_values.device)
    else:
        return q_values.argmax(dim=-1)  # (W,)
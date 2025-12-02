import numpy as np
import torch
import torch.nn as nn

from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))

def obs_to_tensor(obs, device, as_tensor = False):
    if as_tensor:
        x = np.concatenate([obs["row_pixels"], obs["prev_rows"].reshape(-1), obs["row_index"]], axis=0).astype(np.float32)
        return torch.from_numpy(x).to(device)

    return {
        "row_pixels": torch.tensor(obs["row_pixels"], dtype=torch.float32, device=device),
        "prev_rows": torch.tensor(obs["prev_rows"], dtype=torch.float32, device=device),
        "row_index": torch.tensor(obs["row_index"], dtype=torch.float32, device=device),
    }

def batch_obs_to_tensor(obs_list, device):
    """Convert list of observations to batched tensors efficiently."""
    row_pixels = torch.tensor(np.stack([o["row_pixels"] for o in obs_list]), dtype=torch.float32, device=device)
    prev_rows = torch.tensor(np.stack([o["prev_rows"] for o in obs_list]), dtype=torch.float32, device=device)
    row_index = torch.tensor(np.stack([o["row_index"] for o in obs_list]), dtype=torch.float32, device=device)
    return {
        "row_pixels": row_pixels,
        "prev_rows": prev_rows,
        "row_index": row_index,
    }

class PerPixelCNN(nn.Module):
    def __init__(self, W, C, history_len=3, hidden_channels=32):
        super().__init__()
        self.W = W
        self.C = C
        self.history_len = history_len
        in_channels = C + (history_len * C)  # current row + K previous image rows
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1)
        )

    def forward(self, obs):
        # obs["row_pixels"]: (B, W, C) or (W, C), obs["prev_rows"]: (B, history_len, W, C) or (history_len, W, C)
        if obs["row_pixels"].dim() == 2:  # Single observation
            x_img = obs["row_pixels"].permute(1, 0)  # (C, W)
            x_hist = obs["prev_rows"].permute(2, 0, 1).reshape(self.history_len * self.C, self.W)  # (history_len*C, W)
            x = torch.cat([x_img, x_hist], dim=0).unsqueeze(0)  # (1, C+history_len*C, W)
            q = self.conv(x).squeeze(0).permute(1, 0)  # (W, 2)
        else:  # Batched observations
            B = obs["row_pixels"].size(0)
            x_img = obs["row_pixels"].permute(0, 2, 1)  # (B, C, W)
            x_hist = obs["prev_rows"].permute(0, 3, 1, 2).reshape(B, self.history_len * self.C, self.W)  # (B, history_len*C, W)
            x = torch.cat([x_img, x_hist], dim=1)  # (B, C+history_len*C, W)
            q = self.conv(x).permute(0, 2, 1)  # (B, W, 2)
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
    # q_values: (W, 2) or (B, W, 2) tensor
    if q_values.dim() == 2:  # Single observation
        W = q_values.size(0)
        if random.random() < epsilon:
            return torch.randint(low=0, high=2, size=(W,), dtype=torch.int64, device=q_values.device)
        else:
            return q_values.argmax(dim=-1)  # (W,)
    else:  # Batched
        B, W, _ = q_values.shape
        mask = torch.rand(B, device=q_values.device) < epsilon
        random_actions = torch.randint(low=0, high=2, size=(B, W), dtype=torch.int64, device=q_values.device)
        greedy_actions = q_values.argmax(dim=-1)  # (B, W)
        return torch.where(mask.unsqueeze(1), random_actions, greedy_actions)
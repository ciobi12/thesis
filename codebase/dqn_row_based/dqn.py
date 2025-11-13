import numpy as np
import torch
import torch.nn as nn

from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("obs", "action", "pixel_rewards", "next_obs", "done"))

def obs_to_tensor(obs, device, as_tensor = False):
    if as_tensor:
        x = np.concatenate([obs["row_pixels"], obs["prev_pred"], obs["row_index"]], axis=0).astype(np.float32)
        return torch.from_numpy(x).to(device)  # shape (2W + 1,)

    return {
        "row_pixels": torch.tensor(obs["row_pixels"], dtype=torch.float32, device=device),
        "prev_preds": torch.tensor(obs["prev_preds"], dtype=torch.float32, device=device),
        "row_index": torch.tensor(obs["row_index"], dtype=torch.float32, device=device),
    }

class PerPixelCNN(nn.Module):
    def __init__(self, W, C, history_len=3, hidden_channels=32):
        super().__init__()
        self.W = W
        self.C = C
        self.history_len = history_len
        in_channels = C + history_len  # image + K previous rows
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1)
        )

    def forward(self, obs):
        # obs["row_pixels"]: (W, C), obs["prev_preds"]: (history_len, W)
        x_img = obs["row_pixels"].permute(1, 0)  # (C, W)
        x_hist = obs["prev_preds"]  # (history_len, W)
        x = torch.cat([x_img, x_hist], dim=0).unsqueeze(0)  # (1, C+history_len, W)
        q = self.conv(x).squeeze(0).permute(1, 0)  # (W, 2)
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
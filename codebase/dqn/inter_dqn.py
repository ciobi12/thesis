import torch
import torch.nn as nn
from collections import deque
import random

class PatchSelNet(nn.Module):
    def __init__(self, in_ch=4, grid=3, coord_dim=2, k_actions=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        feat_dim = 64 * grid * grid
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + coord_dim + 2, 256), nn.ReLU(),  # +2 for coverage stats
            nn.Linear(256, k_actions)
        )
    def forward(self, patch_grid, coords_cov):
        z = self.cnn(patch_grid)
        return self.fc(torch.cat([z, coords_cov], dim=1))
    
class InterReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def push(self, s_pg, s_cc, s_mask, a, r, n_pg, n_cc, n_mask, d):
        self.buf.append((s_pg, s_cc, s_mask, a, r, n_pg, n_cc, n_mask, d))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s_pg, s_cc, s_mask, a, r, n_pg, n_cc, n_mask, d = zip(*batch)

        return (
            torch.stack(s_pg),              # [B,C,H,W]
            torch.stack(s_cc),              # [B,feat_dim]
            torch.stack(s_mask),            # [B,K]
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(n_pg),
            torch.stack(n_cc),
            torch.stack(n_mask),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buf)

def masked_argmax(q, mask):
    if mask.dim() < q.dim():
        mask = mask[None, :]
    elif mask.dim() > q.dim():
        mask = mask.squeeze()

    q = q.clone()
    # print(mask.shape)
    # print(q.shape)
    q[mask == 0] = -1e9
    return q.argmax(dim=1)
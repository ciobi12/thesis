import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class PatchNavNet(nn.Module):
    def __init__(self, in_ch=3, n_actions=8, N=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*N*N, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        return self.head(self.conv(x))

class ReplayBuffer:
    def __init__(self, cap=50000): 
        self.buf = deque(maxlen=cap)

    def push(self, s,a,r,s2,d): 
        self.buf.append((s,a,r,s2,d))
    def sample(self, b):
        batch = random.sample(self.buf, b)
        s,a,r,s2,d = zip(*batch)
        return torch.stack(s), torch.tensor(a), torch.tensor(r, dtype=torch.float32), \
               torch.stack(s2), torch.tensor(d, dtype=torch.float32)
    def __len__(self): 
        return len(self.buf)

def dqn_update(net, tgt, opt, rb, gamma=0.99, batch=64):
    s,a,r,s2,d = rb.sample(batch)
    q = net(s).gather(1, a.view(-1,1)).squeeze(1)
    with torch.no_grad():
        a_star = net(s2).argmax(dim=1, keepdim=True)
        q_tgt = tgt(s2).gather(1, a_star).squeeze(1)
        y = r + (1 - d) * gamma * q_tgt
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad() 
    loss.backward() 
    opt.step()
    return float(loss.item())
import torch
import torch.nn as nn

class PatchDQN(nn.Module):
    def __init__(self, patch_size=5, n_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * patch_size * patch_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        # x shape: (batch, patch_size, patch_size)
        x = x.unsqueeze(1).float() / 2.0  # normalize to [0,1]
        return self.net(x)
from collections import defaultdict
from typing import Tuple, Dict

import numpy as np

class PatchQLearner:
    def __init__(self, patch_size=5, n_actions=8, alpha=0.2, gamma=0.95, eps_start=0.2, eps_min=0.01, eps_decay=0.999):
        self.P = patch_size
        self.A = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q: Dict[Tuple[int,int], np.ndarray] = defaultdict(lambda: np.zeros((self.P*self.P, self.A), dtype=np.float32))

    def state_index(self, obs):
        x_local, y_local = int(obs[0]), int(obs[1])
        return y_local * self.P + x_local

    def select_action(self, patch_idx, s_idx):
        if np.random.rand() < self.eps:
            return np.random.randint(self.A)
        q = self.Q[patch_idx][s_idx]
        return int(np.argmax(q))

    def update(self, patch_idx, s, a, r, s_next, done):
        q = self.Q[patch_idx]
        best_next = 0.0 if done else float(np.max(q[s_next]))
        td = r + self.gamma * best_next - q[s, a]
        q[s, a] += self.alpha * td
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
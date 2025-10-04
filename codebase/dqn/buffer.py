from collections import deque
import torch
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = np.array(state)
        next_state = np.array(next_state)
        return (
            torch.tensor(state),
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
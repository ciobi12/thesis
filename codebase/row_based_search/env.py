import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class PixelPathEnv(gym.Env):
    """
    Gym-style environment for pixel path exploration.
    Each pixel is an agent deciding if it's on the path (1=255) or not (0).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, image):
        super(PixelPathEnv, self).__init__()
        self.image = image
        print(np.unique_values(self.image))
        self.H, self.W = image.shape

        # Action space: 0 (background), 1 (path)
        self.action_space = spaces.Discrete(2)

        # Observation space: (row, col, pixel_value)
        # row and col are normalized to [0,1], pixel_value in [0,1]
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()

    def reset(self):
        self.row, self.col = self.H - 1, 0
        self.output = np.zeros_like(self.image)  # agent’s decisions
        return self._get_obs()

    # def _get_obs(self):
    #     return np.array([
    #         self.row / self.H,
    #         self.col / self.W,
    #         float(self.image[self.row, self.col])
    #     ], dtype=np.float32)
    
    def _get_obs(self):
        return (self.row / self.H, self.col / self.W, float(self.image[self.row, self.col]))

    def step(self, action):
        pixel_val = self.image[self.row, self.col]
        reward = -abs(pixel_val - action)
        # print(reward)

        # Save agent's decision
        self.output[self.row, self.col] = action

        # Move to next pixel
        if self.col < self.W - 1:
            self.col += 1
        else:
            self.col = 0
            self.row -= 1

        done = self.row < 0
        obs = None if done else self._get_obs()
        return obs, reward, done, {}

    def render(self, mode="human"):
        plt.imshow(self.output, cmap="gray", vmin=0, vmax=255)
        plt.title("Agent’s Path Decisions")
        plt.axis("off")
        plt.show()

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

    def __init__(self, image, render_mode: str = "human"):
        super(PixelPathEnv, self).__init__()
        # Normalize image to binary 0/1 values
        self.image = (image > 0).astype(np.uint8)
        print(np.unique(self.image))
        self.H, self.W = self.image.shape

        # Action space: 0 (background), 1 (path)
        self.action_space = spaces.Discrete(2)

        # Observation space: (row, col, pixel_value)
        # row and col are normalized to [0,1], pixel_value in [0,1]
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.render_mode = render_mode
        if self.render_mode == "human":
            self._fig_ax = None

        self.reset()

    def reset(self):
        self.row, self.col = self.H - 1, 0
        self.output = np.zeros_like(self.image)  # agent’s decisions
        return self._get_obs()
    
    def _get_obs(self):
        return (self.row / self.H, self.col / self.W, float(self.image[self.row, self.col]))

    def step(self, action):
        pixel_val = self.image[self.row, self.col]
        
        reward = -float(abs(pixel_val - action))
        # print(reward)
            # print(type(pixel_val), pixel_val, action)

        self.output[self.row, self.col] = action

        # Move to next pixel
        if self.col < self.W - 1:
            self.col += 1
        else:
            self.col = 0
            self.row -= 1

        done = self.row < 0
        obs = None if done else self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, done, {}
    
    def coverage(self):
        return (self.output & self.image).sum()/self.image.sum()

    def render(self):
        if self.render_mode == "array":
            return self.output
        elif self.render_mode == "human":
            if self._fig_ax is None:
                fig, axs = plt.subplots(1, 2, figsize=(12,6))
                self._fig_ax = (fig, axs)
            fig, axs = self._fig_ax
            axs[0].clear(); axs[1].clear()
            axs[0].imshow(self.image, cmap="gray")
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(self.output, cmap="gray")
            axs[1].set_title(f"Coverage: {self.coverage()*100:.1f}%")
            axs[1].axis("off")
          
            axs[0].axis("off"); axs[1].axis("off")
            fig.canvas.draw_idle()
            plt.pause(1e-27)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not implemented.")

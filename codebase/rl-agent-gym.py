# gym_lsystem_path_env.py
from __future__ import annotations
import math
import random
from typing import Optional, Tuple, Dict

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from l_systems import LSystemGenerator

# Actions: dy, dx in image (row, col) space
ACTIONS = np.array([
    (-1,  0),  # 0: N
    ( 1,  0),  # 1: S
    ( 0,  1),  # 2: E
    ( 0, -1),  # 3: W
    (-1,  1),  # 4: NE
    (-1, -1),  # 5: NW
    ( 1,  1),  # 6: SE
    ( 1, -1),  # 7: SW
], dtype=np.int8)
ACTION_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

class PathLSystemEnv(gym.Env):
    """
    Gymnasium-compatible environment where the agent must traverse a connected path:

    - observation_space: Discrete(3^9) encodes a 3x3 patch (values: 0 off, 1 unvisited path, 2 visited)
    - action_space: Discrete(8) for the 8 compass moves
    - reward:
        +1 when entering an unvisited path pixel
         0 when entering a visited path pixel
        -1 when attempting to step off-path (move rejected; agent stays)
    - terminated when all path pixels have been visited
    - truncated when max_steps (optional) is reached

    Render modes:
      - "rgb_array": returns an HxWx3 uint8 frame
      - "human": uses matplotlib to show the frame
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self,
                 mask: np.ndarray,
                 render_mode: Optional[str] = None,
                 max_steps: Optional[int] = None,
                 seed: Optional[int] = None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.path = mask.astype(np.uint8)
        assert self.path.ndim == 2
        self.H, self.W = self.path.shape
        self.path_total = int(self.path.sum())
        if self.path_total == 0:
            raise ValueError("Path mask is empty.")

        # Spaces
        self.action_space = spaces.Discrete(8)
        # State id encodes the 3x3 patch in base
        self.observation_space = spaces.Discrete(3**9)

        # RNG / seeding
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.rng = random.Random(int(self._np_random.integers(0, 2**31 - 1)))

        # Starting position: choose a path pixel near the bottom
        ys, xs = np.where(self.path == 1)
        idx = int(np.argmax(ys))
        self.start = (int(ys[idx]), int(xs[idx]))

        # Episode state holders
        self.visited = None
        self.pos = None
        self._step_count = 0

        # For "human" rendering
        self._fig_ax = None

    # ----- Core helpers -----

    def _encode_patch(self, y: int, x: int) -> Tuple[int, np.ndarray]:
        """
        Encode 3x3 patch around (y,x) to base-3 integer in {0,1,2}^9.
          0 = off-path or out-of-bounds
          1 = unvisited path
          2 = visited path
        """
        vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < self.H and 0 <= xx < self.W and self.path[yy, xx] == 1:
                    vals.append(2 if self.visited[yy, xx] == 1 else 1)
                else:
                    vals.append(0)

        # Base-3 little-endian encoding
        state_id = 0
        base = 1
        for v in vals:
            state_id += v * base
            base *= 3

        patch = np.array(vals, dtype=np.uint8).reshape(3, 3)
        return state_id, patch

    def _obs(self) -> int:
        sid, _ = self._encode_patch(*self.pos)
        return sid

    def _get_frame(self) -> np.ndarray:
        """
        Image for visualization:
          - background black
          - path light gray
          - visited green
          - agent red
        """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        path_mask = (self.path == 1)
        img[path_mask] = (200, 200, 200)  # light gray
        visited_mask = (self.visited == 1)
        img[visited_mask] = (30, 180, 80)  # green
        y, x = self.pos
        img[y, x] = (220, 40, 40)  # agent red
        return img

    # ----- Gym API -----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            self.rng.seed(int(self._np_random.integers(0, 2**31 - 1)))

        self.visited = np.zeros_like(self.path, dtype=np.uint8)
        self.pos = self.start
        y, x = self.pos
        self.visited[y, x] = 1
        self._step_count = 0
        observation = self._obs()
        info = {
            "coverage": self.coverage(),
            "position": self.pos,
        }
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        dy, dx = ACTIONS[action]
        y, x = self.pos
        ny, nx = y + int(dy), x + int(dx)

        terminated = False
        truncated = False

        # Default: assume off-path
        reward = -1.0
        moved = False
        offpath = True

        # Check bounds
        if 0 <= ny < self.H and 0 <= nx < self.W:
            if self.path[ny, nx] == 1:
                # On-path: move
                self.pos = (ny, nx)
                moved = True
                offpath = False
                # Reward based on visited
                if self.visited[ny, nx] == 0:
                    reward = +1.0
                    self.visited[ny, nx] = 1
                else:
                    reward = -0.1
            else:
                # on-grid but off-path
                reward = -1

        # Book-keeping
        self._step_count += 1

        if self.visited.sum() == self.path_total:
            terminated = True
        if self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True

        observation = self._obs()
        info = {
            "coverage": self.coverage(),
            "position": self.pos,
            "moved": moved,
            "offpath": offpath,
            "action_name": ACTION_NAMES[action],
        }

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        frame = self._get_frame()
        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
            # Lazy import to avoid hard dependency when not needed
            import matplotlib.pyplot as plt
            if self._fig_ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                self._fig_ax = (fig, ax)
            fig, ax = self._fig_ax
            ax.clear()
            ax.imshow(frame, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Coverage: {self.coverage()*100:.1f}%")
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            return None

    def close(self):
        self._fig_ax = None

    # ----- Convenience -----

    def coverage(self) -> float:
        return float(self.visited.sum()) / float(self.path_total)

#### For DQN/PPO ####

class PatchObservation(gym.ObservationWrapper):
    """
    Replace Discrete state id with the actual 3x3 patch (uint8 in {0,1,2}).
    """
    def __init__(self, env: PathLSystemEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.uint8)

    def observation(self, observation):
        # observation is the sid; recompute patch from env
        _, patch = self.env._encode_patch(*self.env.pos)
        return patch
    
if __name__ == "__main__":
    lsys_obj = LSystemGenerator(axiom = "X",
                                rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                         "F": "FF"
                                         }
                                )
    
    iterations = 2
    angle = 25.0
    step = 5

    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle)
    # lsys_obj.draw_lsystem()
    mask = lsys_obj.build_mask(canvas_size=(45, 45))

    ### Tabular-Q learning ###
    env = PathLSystemEnv(mask, render_mode = "human", max_steps=5000, seed=0)  # Discrete obs by default
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    alpha, gamma = 0.2, 0.9
    eps, eps_end, eps_decay = 0.2, 0.01, 0.999

    # Greedy-epsilon policy
    def select_action(s):
        if random.random() < eps:
            return env.action_space.sample()
        q = Q[s]
        maxq = np.max(q)
        cand = np.flatnonzero(np.isclose(q, maxq))
        return int(random.choice(cand))

    episodes = 10
    for ep in range(episodes):
        s, info = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while True:
            a = select_action(s)
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            target = r + (0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            total_r += r
            steps += 1
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
        if ep % 2 == 0:
            print(f"Episode {ep + 1:4d} | reward={total_r:6.1f} | steps={steps:6d} | coverage={info['coverage']*100:5.1f}% | eps={eps:.3f}")

    env.close()
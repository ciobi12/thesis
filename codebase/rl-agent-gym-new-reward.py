# gym_lsystem_path_env_shaped.py
from __future__ import annotations
import math
import random
from typing import Optional, Tuple, Dict

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from l_systems import LSystemGenerator

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
    Traversal task on a connected path (mask=1):
      - observation: Discrete(3^9) = 3x3 patch encoded in base-3 {0 off, 1 unvisited, 2 visited}
      - actions: 8 compass moves
      - base reward:
            + new_visit_reward when stepping onto an unvisited path pixel
            + revisit_penalty when visiting an already visited path pixel (often 0)
            + offpath_penalty when attempting to move off-path (move rejected)
      - shaping:
            + optional small step_penalty per step
            + optional distance shaping (choose one):
                * potential-based: gamma_shaping * Phi(s') - Phi(s), with Phi(s) = -lambda * distance
                * simple: shaping_coeff * (prev_distance - new_distance)
      - termination: all path pixels visited
      - truncation: when max_steps is reached
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self,
                 mask: np.ndarray,
                 render_mode: Optional[str] = None,
                 max_steps: Optional[int] = None,
                 seed: Optional[int] = None,
                 # --- base rewards ---
                 new_visit_reward: float = 1.0,
                 revisit_penalty: float = 0.0,
                 offpath_penalty: float = -1.0,
                 step_penalty: float = -0.01,
                 terminal_bonus: float = 50.0,
                 # --- shaping options ---
                 use_potential_shaping: bool = True,
                 potential_lambda: float = 0.1,   # scales Φ(s) = -λ * d_norm
                 shaping_gamma: float = 0.95,     # should match agent's gamma
                 use_simple_shaping: bool = False,
                 simple_shaping_coeff: float = 0.1,
                 normalize_distance: bool = True,
                 # --- movement constraint ---
                 forbid_corner_cutting: bool = False):
        super().__init__()

        # Config
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.new_visit_reward = new_visit_reward
        self.revisit_penalty = revisit_penalty
        self.offpath_penalty = offpath_penalty
        self.step_penalty = step_penalty
        self.terminal_bonus = terminal_bonus

        self.use_potential_shaping = use_potential_shaping
        self.potential_lambda = potential_lambda
        self.shaping_gamma = shaping_gamma

        self.use_simple_shaping = use_simple_shaping
        self.simple_shaping_coeff = simple_shaping_coeff
        self.normalize_distance = normalize_distance
        self.forbid_corner_cutting = forbid_corner_cutting

        # Mask
        self.path = mask.astype(np.uint8)
        assert self.path.ndim == 2, "mask must be 2D"
        self.H, self.W = self.path.shape
        self.path_total = int(self.path.sum())
        if self.path_total == 0:
            raise ValueError("Path mask is empty.")

        # Spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(3**9)

        # RNG
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.rng = random.Random(int(self._np_random.integers(0, 2**31 - 1)))

        # Start position: bottom-most path pixel
        ys, xs = np.where(self.path == 1)
        idx = int(np.argmax(ys))
        self.start = (int(ys[idx]), int(xs[idx]))

        # Episode state
        self.visited: Optional[np.ndarray] = None
        self.pos: Optional[Tuple[int, int]] = None
        self._step_count = 0

        # Render cache
        self._fig_ax = None

    # --------- helpers ---------

    def _encode_patch(self, y: int, x: int) -> Tuple[int, np.ndarray]:
        vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < self.H and 0 <= xx < self.W and self.path[yy, xx] == 1:
                    vals.append(2 if self.visited[yy, xx] == 1 else 1)
                else:
                    vals.append(0)
        # base-3 little-endian
        sid = 0
        base = 1
        for v in vals:
            sid += v * base
            base *= 3
        return sid, np.array(vals, dtype=np.uint8).reshape(3, 3)

    def _obs(self) -> int:
        sid, _ = self._encode_patch(*self.pos)
        return sid

    def _diagonal_blocked(self, y: int, x: int, ny: int, nx: int) -> bool:
        """If forbidding corner cutting, disallow a diagonal move when both adjacent orthogonal cells are off-path."""
        dy, dx = ny - y, nx - x
        if abs(dy) == 1 and abs(dx) == 1:
            a = (y, x + dx)
            b = (y + dy, x)
            ok_a = (0 <= a[0] < self.H and 0 <= a[1] < self.W and self.path[a] == 1)
            ok_b = (0 <= b[0] < self.H and 0 <= b[1] < self.W and self.path[b] == 1)
            return not (ok_a or ok_b)
        return False

    # ----- distance / potentials -----

    def _nearest_unvisited_distance(self, y: int, x: int) -> float:
        """Manhattan distance to the nearest UNVISITED path pixel (0 if none left)."""
        if self.visited.sum() == self.path_total:
            return 0.0
        
        ys, xs = np.where((self.path == 1) & (self.visited == 0))
        if ys.size == 0:
            return 0.0
        # Manhattan distance
        d = np.min(np.abs(ys - y) + np.abs(xs - x))
        if self.normalize_distance:
            # normalize by max possible manhattan distance within grid
            d = d / float(self.H + self.W)
        return float(d)

    def _potential(self, y: int, x: int) -> float:
        # Φ(s) = -λ * d_norm; smaller distance => higher potential (less negative)
        d = self._nearest_unvisited_distance(y, x)
        return -self.potential_lambda * d

    # --------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            self.rng.seed(int(self._np_random.integers(0, 2**31 - 1)))

        self.visited = np.zeros_like(self.path, dtype=np.uint8)
        self.pos = self.start
        self.visited[self.pos] = 1
        self._step_count = 0

        obs = self._obs()
        info = {"coverage": self.coverage(), "position": self.pos}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        y, x = self.pos
        dy, dx = ACTIONS[action]
        ny, nx = y + int(dy), x + int(dx)

        prev_phi = self._potential(y, x)
        prev_dist = self._nearest_unvisited_distance(y, x)

        reward = 0.0
        terminated = False
        truncated = False
        moved = False
        offpath = True

        # bounds check
        if 0 <= ny < self.H and 0 <= nx < self.W:
            # optional corner-cutting rule
            if self.forbid_corner_cutting and self._diagonal_blocked(y, x, ny, nx):
                # treat as off-path attempt
                reward += self.offpath_penalty
            elif self.path[ny, nx] == 1:
                # valid move
                self.pos = (ny, nx)
                moved = True
                offpath = False
                if self.visited[ny, nx] == 0:
                    reward += self.new_visit_reward
                    self.visited[ny, nx] = 1
                else:
                    reward += self.revisit_penalty
            else:
                reward += self.offpath_penalty
        else:
            reward += self.offpath_penalty

        # step penalty (applies every step)
        reward += self.step_penalty

        # shaping
        if self.use_potential_shaping:
            # policy-invariant shaping: F = gamma * Phi(s') - Phi(s)
            y2, x2 = self.pos
            next_phi = self._potential(y2, x2)
            reward += self.shaping_gamma * next_phi - prev_phi
        elif self.use_simple_shaping:
            # dense (non-invariant) shaping: +c * (dist reduction)
            y2, x2 = self.pos
            new_dist = self._nearest_unvisited_distance(y2, x2)
            reward += self.simple_shaping_coeff * (prev_dist - new_dist)

        self._step_count += 1

        if (self.visited.sum() - self.path_total)/self.path_total < 0.1:
            terminated = True
            reward += self.terminal_bonus

        if self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True

        obs = self._obs()
        info = {
            "coverage": self.coverage(),
            "position": self.pos,
            "moved": moved,
            "offpath": offpath,
            "action_name": ACTION_NAMES[action],
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _get_frame(self) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[self.path == 1] = (200, 200, 200)
        img[self.visited == 1] = (30, 180, 80)
        y, x = self.pos
        img[y, x] = (220, 40, 40)
        return img

    def render(self):
        frame = self._get_frame()
        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
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

    def close(self):
        self._fig_ax = None

    def coverage(self) -> float:
        return float(self.visited.sum()) / float(self.path_total)

### For DQN/PPO

class PatchObservation(gym.ObservationWrapper):
    def __init__(self, env: PathLSystemEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.uint8)

    def observation(self, observation):
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
    mask = lsys_obj.build_mask()

    env = PathLSystemEnv(
        mask,
        max_steps=5000,
        render_mode="human",
        step_penalty=-0.005,
        revisit_penalty=-0.01,
        terminal_bonus=50.0,
        use_potential_shaping=True,
        potential_lambda=0.15,
        shaping_gamma=0.95,
    )

    # Tabular Q
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    alpha, gamma = 0.3, 0.95
    eps, eps_end, eps_decay = 0.2, 0.01, 0.999

    for ep in range(1, 10):
        s, info = env.reset()
        total_r = 0.0
        while True:
            a = (env.action_space.sample() if random.random()<eps
                 else int(np.argmax(Q[s])))
            s_next, r, term, trunc, info = env.step(a)
            Q[s, a] += alpha * (r + (0 if (term or trunc) else gamma * np.max(Q[s_next])) - Q[s, a])
            s = s_next
            total_r += r
            if term or trunc:
                break
        eps = max(eps_end, eps * eps_decay)
        if ep % 50 == 0:
            print(f"Ep {ep:3d}  reward={total_r:6.1f}  coverage={info['coverage']*100:5.1f}%  eps={eps:.3f}")
    env.close()

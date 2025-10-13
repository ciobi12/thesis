import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class PathTraversalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, path_mask: np.ndarray, patch_size: int = 5,
                 start_on_path: bool = True,
                 max_steps_per_patch: int = 200,
                 max_steps_global: int = 5000,
                 target_coverage: float = 0.9, 
                 render_mode: str = "human"):
        super().__init__()
        assert path_mask.dtype == np.uint8 or path_mask.dtype == bool
        self.path_mask = (path_mask > 0).astype(np.uint8)
        # print(self.path_mask.sum())
        self.H, self.W = self.path_mask.shape
        self.patch_size = patch_size
        self.max_steps_per_patch = max_steps_per_patch
        self.max_steps_global = max_steps_global
        self.target_coverage = target_coverage

        # Path exploration state (0/1) and global exploration state
        self.explored_path = np.zeros_like(self.path_mask, dtype=np.uint8)
        self.global_path = np.zeros_like(self.path_mask, dtype=np.uint8)
        self.total_path_pixels = int(self.path_mask.sum())

        # Action space: 8 directions
        self.action_space = spaces.Discrete(8)

        # Observation: agent position within current patch (x_local, y_local)
        # self.observation_space = spaces.MultiDiscrete([patch_size, patch_size])
        self.observation_space = spaces.Box(low=0, high=patch_size-1, shape=(2,), dtype=np.int32)

        # Movement deltas
        self.moves = np.array([
            [ 0,-1],  # N
            [ 1,-1],  # NE
            [ 1, 0],  # E
            [ 1, 1],  # SE
            [ 0, 1],  # S
            [-1, 1],  # SW
            [-1, 0],  # W
            [-1,-1],  # NW
        ], dtype=np.int32)

        # Internal episode state
        self.curr_patch_idx = (0, 0)
        self.curr_patch_bounds = (0, 0, patch_size, patch_size)  # x0, y0, x1, y1 (exclusive)
        self.agent_xy = (0, 0)  # global coords
        self.last_position = (-1 ,-1)
        self.steps_in_patch = 0
        self.global_steps = 0
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            self._fig_ax = None 

        # Optional: choose a random on-path start
        if start_on_path and self.total_path_pixels > 0:
            ys, xs = np.where(self.path_mask == 1)
            idx = int(np.argmax(ys))
            self.agent_xy = (int(xs[idx]), int(ys[idx]))
            self._recenter_patch_to_include(self.agent_xy)

    def local_state_tensor(self) -> np.ndarray:
        x0, y0, x1, y1 = self.curr_patch_bounds
        patch_path = self.path_mask[y0:y1, x0:x1].astype(np.float32)
        patch_explored = self.explored_path[y0:y1, x0:x1].astype(np.float32)
        agent = np.zeros_like(patch_path, dtype=np.float32)
        ax, ay = self.agent_xy
        if y0 <= ay < y1 and x0 <= ax < x1:
            agent[ay - y0, ax - x0] = 1.0
        # shape [3, H, W]
        return np.stack([patch_path, patch_explored, agent], axis=0)
    
    def frontier_candidates(self, K: int = 8):
        px, py = self.curr_patch_idx
        cand = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: 
                    continue
                npx, npy = px + dx, py + dy
                x0, y0 = npx * self.patch_size, npy * self.patch_size
                if x0 < 0 or y0 < 0 or x0 >= self.W or y0 >= self.H:
                    continue
                x1, y1 = min(x0 + self.patch_size, self.W), min(y0 + self.patch_size, self.H)
                patch_path = self.path_mask[y0:y1, x0:x1]
                patch_explored = self.explored_path[y0:y1, x0:x1]
                has_path = int(patch_path.sum() > 0)
                unexplored = int((patch_path & (1 - patch_explored)).sum())
                cx, cy = (x0 + self.patch_size // 2, y0 + self.patch_size // 2)
                ax, ay = self.agent_xy
                dist = np.hypot(cx - ax, cy - ay)
                cand.append({
                    "idx": (npx, npy),
                    "bounds": (x0, y0, x1, y1),
                    "has_path": has_path,
                    "unexplored": unexplored,
                    "dist": dist
                })
        # Sort by dist; take top-K; build mask
        cand.sort(key=lambda c: c["dist"])
        cand = cand[:K]
        mask = np.zeros(K, dtype=np.uint8)
        for i in range(len(cand)):
            mask[i] = 1
        return cand, mask  # list of dicts + binary mask
    
    def patch_grid(self, grid_size: int = 3) -> np.ndarray:
        half = grid_size // 2
        px, py = self.curr_patch_idx
        C = 4  # path, explored_frac, frontier_flag, uncertainty (optional 0)
        grid = np.zeros((C, grid_size, grid_size), dtype=np.float32)
        for gy in range(-half, half+1):
            for gx in range(-half, half+1):
                npx, npy = px + gx, py + gy
                x0, y0 = npx * self.patch_size, npy * self.patch_size
                if x0 < 0 or y0 < 0 or x0 >= self.W or y0 >= self.H:
                    continue
                x1, y1 = min(x0 + self.patch_size, self.W), min(y0 + self.patch_size, self.H)
                patch_path = self.path_mask[y0:y1, x0:x1]
                patch_explored = self.explored_path[y0:y1, x0:x1]
                p_sum = patch_path.sum()
                e_sum = (patch_path & patch_explored).sum()
                has_path = float(p_sum > 0)
                explored_frac = float(e_sum) / max(1.0, float(p_sum))
                frontier_flag = float((p_sum > 0) and (e_sum < p_sum))
                grid[0, gy + half, gx + half] = has_path
                grid[1, gy + half, gx + half] = explored_frac
                grid[2, gy + half, gx + half] = frontier_flag
                grid[3, gy + half, gx + half] = 0.0  # placeholder for uncertainty
        return grid  # [C, H=grid_size, W=grid_size]
    
    def select_patch_candidate(self, cand):
        bounds = cand["bounds"]
        self.curr_patch_bounds = bounds
        self.curr_patch_idx = (bounds[0] // self.patch_size, bounds[1] // self.patch_size)
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + self.patch_size // 2, y0 + self.patch_size // 2)
        if self.path_mask[cy, cx] == 1:
            self.agent_xy = (cx, cy)
        else:
            ys, xs = np.where(self.path_mask[y0:y1, x0:x1] == 1)
            if len(xs) == 0:
                return False
            idx = np.random.randint(len(xs))
            self.agent_xy = (x0 + int(xs[idx]), y0 + int(ys[idx]))
        return True


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.explored_path.fill(0)
        self.global_steps = 0

        if options["start_from_seed"]:
            ys, xs = np.where(self.path_mask == 1)
            idx = int(np.argmax(ys))
            self.agent_xy = (int(xs[idx]), int(ys[idx]))
            self._recenter_patch_to_include(self.agent_xy)
            self.steps_in_patch = 0
        else:
            self.agent_xy = self.last_position
            self._recenter_patch_to_include(self.agent_xy)
            self.steps_in_patch = 0

        if options["reset_global_mask"]:
            self.global_path.fill(0)

        if self.path_mask[self.agent_xy[1], self.agent_xy[0]] == 1:
            self.explored_path[self.agent_xy[1], self.agent_xy[0]] = 1

        return self._obs(), {}

    def step(self, action: int):
        self.global_steps += 1
        self.steps_in_patch += 1

        dx, dy = self.moves[action]
        x, y = self.agent_xy
        nx, ny = x + int(dx), y + int(dy)

        # Clamp to image bounds
        nx = np.clip(nx, 0, self.W - 1)
        ny = np.clip(ny, 0, self.H - 1)

        # Enforce patch boundary: if leaving, block move
        x0, y0, x1, y1 = self.curr_patch_bounds
        if not (x0 <= nx < x1 and y0 <= ny < y1):
            nx, ny = x, y  

        self.agent_xy = (nx, ny)
        self.global_path[self.agent_xy[1], self.agent_xy[0]] = 1

        # Reward logic
        on_path = self.path_mask[ny, nx] == 1
        reward = 0.0

        if on_path:
            if self.explored_path[ny, nx] == 0:
                self.explored_path[ny, nx] = 1
                reward = 1
            else:
                reward = -0.1
        else:
            reward = -10.0
        
        patch_done = self._patch_fully_explored() or (self.steps_in_patch >= self.max_steps_per_patch)

        ep_coverage, total_coverage = self.coverage()
        global_done = (total_coverage >= self.target_coverage)

        terminated = global_done
        truncated = self.global_steps >= self.max_steps_global
        if truncated:
            self.last_position = self.agent_xy

        dead_end = False
        # If patch done and not globally done, advance to next frontier patch
        if patch_done and not global_done:
            moved = self._advance_to_frontier_patch()
            self.steps_in_patch = 0
            if not moved:
                self.last_position = self.agent_xy
                dead_end = True
        
        info = {"episode_coverage": ep_coverage, "total_coverage": total_coverage}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), patch_done, dead_end, terminated, truncated, info

    def _obs(self):
        x0, y0, x1, y1 = self.curr_patch_bounds
        x, y = self.agent_xy
        return np.array([x - x0, y - y0], dtype=np.int64)

    def _recenter_patch_to_include(self, gxy: Tuple[int, int]):
        gx, gy = gxy
        px = (gx // self.patch_size) * self.patch_size
        py = (gy // self.patch_size) * self.patch_size
        x0, y0 = px, py
        x1, y1 = min(px + self.patch_size, self.W), min(py + self.patch_size, self.H)
        self.curr_patch_bounds = (x0, y0, x1, y1)
        self.curr_patch_idx = (px // self.patch_size, py // self.patch_size)

    def _patch_fully_explored(self) -> bool:
        x0, y0, x1, y1 = self.curr_patch_bounds
        patch_path = self.path_mask[y0:y1, x0:x1]
        if patch_path.sum() == 0:
            return True
        patch_explored = self.explored_path[y0:y1, x0:x1]
        return int((patch_path & patch_explored).sum()) == int(patch_path.sum())
    
    def _advance_to_frontier_patch(self) -> bool:
        """
        Move to the nearest neighbouring patch (by center distance) that contains any path pixels.
        Does not use explored/unexplored info.
        """
        px, py = self.curr_patch_idx
        candidates = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                npx, npy = px + dx, py + dy
                x0, y0 = npx * self.patch_size, npy * self.patch_size
                if x0 >= self.W or y0 >= self.H or x0 < 0 or y0 < 0:
                    continue
                x1, y1 = min(x0 + self.patch_size, self.W), min(y0 + self.patch_size, self.H)
                patch_path = self.path_mask[y0:y1, x0:x1]
                if patch_path.sum() > 0:
                    # Compute distance from agent to patch center
                    cx, cy = (x0 + self.patch_size // 2, y0 + self.patch_size // 2)
                    ax, ay = self.agent_xy
                    dist = np.sqrt((cx - ax)**2 + (cy - ay)**2)
                    candidates.append(((npx, npy), (x0, y0, x1, y1), dist))

        if not candidates:
            return False

        # Pick closest patch
        candidates.sort(key=lambda c: c[2])
        (_, bounds, _) = candidates[0]
        self.curr_patch_bounds = bounds
        self.curr_patch_idx = (bounds[0] // self.patch_size, bounds[1] // self.patch_size)

        # Move agent to center or random path pixel in patch
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + self.patch_size // 2, y0 + self.patch_size // 2)
        if self.path_mask[cy, cx] == 1:
            self.agent_xy = (cx, cy)
        else:
            ys, xs = np.where(self.path_mask[y0:y1, x0:x1] == 1)
            if len(xs) == 0:
                return False
            idx = np.random.randint(len(xs))
            self.agent_xy = (x0 + int(xs[idx]), y0 + int(ys[idx]))

        return True

    def coverage(self) -> float:
        episode_explored_path = int((self.explored_path & self.path_mask).sum()) / max(1, self.total_path_pixels)
        total_explored_path = int((self.global_path & self.path_mask).sum()) / max(1, self.total_path_pixels)
        return episode_explored_path, total_explored_path

    def _get_frame(self) -> np.ndarray:
        """
        Image for visualization:
          - background black
          - path light gray
          - visited green
          - agent red
        """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # path_mask = (path_mask == 1)
        img[self.path_mask == 1] = (200, 200, 200)  # light gray
        # visited_mask = (self.explored_path == 1)
        global_explored = (self.global_path == 1)
        img[global_explored & ~(self.path_mask == 1)] = (65,105,225)  # royal blue
        img[global_explored & self.path_mask == 1] = (30, 180, 80) # green
        x, y = self.agent_xy
        img[y, x] = (220, 40, 40)  # agent red
        return img

    def render(self):
        ep_cov, global_cov = self.coverage()
        frame = self._get_frame()
        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
            if self._fig_ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                self._fig_ax = (fig, ax)
            fig, ax = self._fig_ax
            ax.clear()
            ax.imshow(frame, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Coverage: {ep_cov*100:.1f}%")
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not implemented.")
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
                 target_coverage: float = 0.9, 
                 render_mode: str = "human"):
        super().__init__()
        assert path_mask.dtype == np.uint8 or path_mask.dtype == bool
        self.path_mask = (path_mask > 0).astype(np.uint8)
        self.H, self.W = self.path_mask.shape
        self.patch_size = patch_size
        self.max_steps_per_patch = max_steps_per_patch
        self.target_coverage = target_coverage

        # Global exploration state (0/1) aligned with path pixels only
        self.explored_path = np.zeros_like(self.path_mask, dtype=np.uint8)
        self.global_path = np.zeros_like(self.path_mask, dtype=np.uint8)
        self.total_path_pixels = int(self.path_mask.sum())

        # Action space: 8 directions
        self.action_space = spaces.Discrete(8)

        # Observation: agent position within current patch (x_local, y_local)
        self.observation_space = spaces.MultiDiscrete([patch_size, patch_size])

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
        self.last_dead_end_position = (-1 ,-1)
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.explored_path.fill(0)
        self.global_steps = 0

        # Place agent on seed of the L-system
        if options["start_from_seed"]:
            ys, xs = np.where(self.path_mask == 1)
            idx = int(np.argmax(ys))
            self.agent_xy = (int(xs[idx]), int(ys[idx]))
            self._recenter_patch_to_include(self.agent_xy)
            self.steps_in_patch = 0
        else:
            self.agent_xy = self.last_dead_end_position
            self._recenter_patch_to_include(self.agent_xy)
            self.steps_in_patch = 0
        
        if options["reset_global_mask"]:
            self.global_path.fill(0)

        # Mark first visit if on path
        if self.path_mask[self.agent_xy[1], self.agent_xy[0]] == 1:
            self.explored_path[self.agent_xy[1], self.agent_xy[0]] = 1

        return self._obs(), {}

    def step(self, action: int):
        self.global_steps += 1
        self.steps_in_patch += 1

        # Proposed move
        dx, dy = self.moves[action]
        x, y = self.agent_xy
        nx, ny = x + int(dx), y + int(dy)

        # Clamp to image bounds
        nx = np.clip(nx, 0, self.W - 1)
        ny = np.clip(ny, 0, self.H - 1)

        # Enforce patch boundary: if leaving, block move
        x0, y0, x1, y1 = self.curr_patch_bounds
        if not (x0 <= nx < x1 and y0 <= ny < y1):
            nx, ny = x, y  # no movement if it would leave the patch

        self.agent_xy = (nx, ny)
        self.global_path[self.agent_xy[1], self.agent_xy[0]] = 1

        # Reward logic
        on_path = self.path_mask[ny, nx] == 1
        first_time = False
        reward = 0.0

        if on_path:
            if self.explored_path[ny, nx] == 0:
                self.explored_path[ny, nx] = 1
                reward = 1.0
                first_time = True
            else:
                reward = -0.5
        else:
            reward = -10.0

        # Patch done?
        patch_done = self._patch_fully_explored() or (self.steps_in_patch >= self.max_steps_per_patch)

        # Global done?
        ep_coverage, total_coverage = self.coverage()
        global_done = (total_coverage >= self.target_coverage)

        terminated = global_done
        truncated = False
        dead_end = False

        # If patch done and not globally done, advance to next frontier patch
        if patch_done and not global_done:
            moved = self._advance_to_frontier_patch()
            self.steps_in_patch = 0
            # If no frontier exists (all explored), terminate
            if not moved:
                self.last_dead_end_position = self.agent_xy
                dead_end = True

        info = {"episode_coverage": ep_coverage, "total_coverage": total_coverage}
        return self._obs(), float(reward), dead_end, terminated, truncated, info

    def _obs(self):
        # Local coordinates within current patch
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
        # Find neighboring patches with unexplored path pixels
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
                patch_explored = self.explored_path[y0:y1, x0:x1]
                unexplored_count = int(patch_path.sum()) - int((patch_path & patch_explored).sum())
                if unexplored_count > 0:
                    # Heuristic: border contact score
                    contact = self._border_contact_score((x0, y0, x1, y1))
                    candidates.append(((npx, npy), (x0, y0, x1, y1), contact, unexplored_count))

        if not candidates:
            return False

        # Pick best by contact, then by unexplored_count
        candidates.sort(key=lambda c: (c[2], c[3]), reverse=True)
        (_, _bounds, _, _) = candidates[0]
        self.curr_patch_bounds = _bounds
        self.curr_patch_idx = (self.curr_patch_bounds[0] // self.patch_size,
                               self.curr_patch_bounds[1] // self.patch_size)

        # Move agent to a path pixel inside the new patch that is adjacent to explored pixels if possible
        x0, y0, x1, y1 = self.curr_patch_bounds
        patch_path = self.path_mask[y0:y1, x0:x1]
        patch_explored = self.explored_path[y0:y1, x0:x1]
        ys, xs = np.where((patch_path == 1) & (patch_explored == 1))
        if len(xs) == 0:
            ys2, xs2 = np.where(patch_path == 1)
            if len(xs2) == 0:
                return False
            idx = np.random.randint(len(xs2))
            self.agent_xy = (x0 + int(xs2[idx]), y0 + int(ys2[idx]))
        else:
            idx = np.random.randint(len(xs))
            self.agent_xy = (x0 + int(xs[idx]), y0 + int(ys[idx]))
        return True

    def _border_contact_score(self, bounds) -> int:
        x0, y0, x1, y1 = bounds
        # Count path pixels on patch border that have been explored globally
        contact = 0
        patch_explored = self.explored_path[y0:y1, x0:x1]
        patch_path = self.path_mask[y0:y1, x0:x1]
        border = np.zeros_like(patch_path, dtype=bool)
        if (y1 - y0) > 0 and (x1 - x0) > 0:
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
        return int(((patch_path == 1) & (patch_explored == 1) & border).sum())

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
        visited_mask = (self.explored_path == 1)
        img[visited_mask] = (30, 180, 80)  # green
        global_path = (self.global_path == 1)
        img[global_path & ~visited_mask] = (65,105,225)  # royal blue
        x, y = self.agent_xy
        img[y, x] = (220, 40, 40)  # agent red
        return img

    def render(self):
        # Simple ASCII or debug prints; replace with visualization as needed
        ep_cov, global_cov = self.coverage()
        frame = self._get_frame()
        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
            # Lazy import to avoid hard dependency when not needed
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
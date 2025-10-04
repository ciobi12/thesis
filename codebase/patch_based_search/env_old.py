import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict

class PathTraversalEnv(gym.Env):
    

    def __init__(self, path_mask: np.ndarray, patch_size: int = 5,
                 start_on_path: bool = True,
                 max_steps_per_patch: int = 200,
                 target_coverage: float = 0.95):
        super().__init__()
        assert path_mask.dtype == np.uint8 or path_mask.dtype == bool
        self.path_mask = (path_mask == 1).astype(np.uint8)
        self.H, self.W = self.path_mask.shape
        self.patch_size = patch_size
        self.max_steps_per_patch = max_steps_per_patch
        self.target_coverage = target_coverage
        self.render_mode = "human"
        self._fig_ax = None # if render_mode is "human"

        # Global exploration state (0/1) aligned with path pixels only
        self.explored_global = np.zeros_like(self.path_mask, dtype=np.uint8)
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
        self.steps_in_patch = 0
        self.global_steps = 0

        # Optional: choose the lowest on-path point start
        if start_on_path and self.total_path_pixels > 0:
            ys, xs = np.where(self.path_mask == 1)
            idx = int(np.argmax(ys))
            self.agent_xy = (int(xs[idx]), int(ys[idx]))
            self._recenter_patch_to_include(self.agent_xy)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.explored_global.fill(0)
        self.global_steps = 0

        ys, xs = np.where(self.path_mask == 1)
        idx = int(np.argmax(ys))
        self.agent_xy = (int(xs[idx]), int(ys[idx]))
        print(self.agent_xy)
        self._recenter_patch_to_include(self.agent_xy)
        self.steps_in_patch = 0

        # Mark first visit if on path
        if self.path_mask[self.agent_xy[1], self.agent_xy[0]] == 1:
            self.explored_global[self.agent_xy[1], self.agent_xy[0]] = 1

        if self.render_mode == "human":
            self.render()
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

        # Reward logic
        on_path = self.path_mask[ny, nx] == 1
        reward = 0.0

        if on_path:
            if self.explored_global[ny, nx] == 0:
                self.explored_global[ny, nx] = 1
                reward = 1.0
            else:
                reward = -0.1
        else:
            reward = -1.0

        patch_done = self._patch_fully_explored() or (self.steps_in_patch >= self.max_steps_per_patch)

        # Global done?
        coverage = self.coverage()
        global_done = (coverage >= self.target_coverage)

        terminated = global_done
        truncated = False

        if patch_done and not global_done:
            moved = self._advance_to_next_patch()
            self.steps_in_patch = 0
            # If no unexplored pixel exists, terminate
            if not moved:
                print(moved)
                terminated = True

        info = {"coverage": coverage}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

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
        patch_explored = self.explored_global[y0:y1, x0:x1]
        return int((patch_path & patch_explored).sum()) == int(patch_path.sum())


    def _advance_to_next_patch(self) -> bool:
        """
        Move to the patch whose center is closest to the nearest unexplored path pixel.
        """
        # Find all unexplored path pixels
        unexplored_y, unexplored_x = np.where(
            (self.path_mask == 1) & (self.explored_global == 0)
        )
        if len(unexplored_x) == 0:
            return False  # No unexplored pixels left

        # Compute distances from agent to each unexplored pixel
        ax, ay = self.agent_xy
        distances = np.sqrt((unexplored_x - ax) ** 2 + (unexplored_y - ay) ** 2)
        nearest_idx = np.argmin(distances)
        target_pixel = (int(unexplored_x[nearest_idx]), int(unexplored_y[nearest_idx]))

        # Determine the patch containing that pixel
        tx, ty = target_pixel
        patch_x = (tx // self.patch_size) * self.patch_size
        patch_y = (ty // self.patch_size) * self.patch_size
        x0, y0 = patch_x, patch_y
        x1, y1 = min(patch_x + self.patch_size, self.W), min(patch_y + self.patch_size, self.H)

        self.curr_patch_bounds = (x0, y0, x1, y1)
        self.curr_patch_idx = (patch_x // self.patch_size, patch_y // self.patch_size)

        # Move agent to the center of this patch (or nearest on-path pixel to center)
        cx, cy = (x0 + self.patch_size // 2, y0 + self.patch_size // 2)
        if self.path_mask[cy, cx] == 1:
            self.agent_xy = (cx, cy)
        else:
            # Find nearest on-path pixel to the patch center
            py, px = np.where(self.path_mask[y0:y1, x0:x1] == 1)
            if len(px) > 0:
                dists = np.sqrt((px + x0 - cx) ** 2 + (py + y0 - cy) ** 2)
                nearest_local = np.argmin(dists)
                self.agent_xy = (x0 + int(px[nearest_local]), y0 + int(py[nearest_local]))
            else:
                return False  # Patch has no path pixels

        return True

    def _border_contact_score(self, bounds) -> int:
        x0, y0, x1, y1 = bounds
        # Count path pixels on patch border that have been explored globally
        contact = 0
        patch_explored = self.explored_global[y0:y1, x0:x1]
        patch_path = self.path_mask[y0:y1, x0:x1]
        border = np.zeros_like(patch_path, dtype=bool)
        if (y1 - y0) > 0 and (x1 - x0) > 0:
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
        return int(((patch_path == 1) & (patch_explored == 1) & border).sum())

    def coverage(self) -> float:
        explored = int((self.explored_global & self.path_mask).sum())
        return explored / max(1, self.total_path_pixels)
    
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
        visited_mask = (self.explored_global == 1)
        img[visited_mask] = (30, 180, 80)  # green
        x, y = self.agent_xy
        img[y, x] = (220, 40, 40)  # agent red
        return img

    def render(self):
        # Simple ASCII or debug prints; replace with visualization as needed
        cov = self.coverage()
        frame = self._get_frame()
        if self.render_mode == "rgb_array":
            return frame
        elif self.render_mode == "human":
            # Lazy import to avoid hard dependency when not needed
            import matplotlib.pyplot as plt
            if self._fig_ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                self._fig_ax = (fig, ax)
            fig, ax = self._fig_ax
            ax.clear()
            ax.imshow(frame, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Coverage: {self.coverage()*100:.1f}%")
            fig.canvas.draw_idle()
            plt.pause(0.0001)
        else:
            return None
        # print(f"Coverage: {cov:.3f}, Patch: {self.curr_patch_idx}, Agent: {self.agent_xy}")
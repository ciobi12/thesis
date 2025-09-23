import math
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ---------------------------
# 1) L-system → list of segments → raster binary mask
# ---------------------------

def expand_lsystem(axiom, rules, iterations):
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s

def lsystem_segments(commands, step=10.0, angle_deg=25.0, start_pos=(0.0, 0.0), start_angle=90.0):
    """Interpret commands into a list of line segments [(x0,y0,x1,y1), ...]."""
    x, y = start_pos
    heading = math.radians(start_angle)
    stack = []
    segs = []
    for ch in commands:
        if ch in ("F", "G"):  # draw forward
            nx = x + step * math.cos(heading)
            ny = y + step * math.sin(heading)
            segs.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch in ("f", "g"):  # move forward without drawing (unused by default)
            x += step * math.cos(heading)
            y += step * math.sin(heading)
        elif ch == "+":
            heading += math.radians(angle_deg)
        elif ch == "-":
            heading -= math.radians(angle_deg)
        elif ch == "[":
            stack.append((x, y, heading))
        elif ch == "]":
            x, y, heading = stack.pop()
        # Ignore symbols like X/Y used for expansion only
    return segs

def segments_to_mask(segments, canvas=(256, 256), margin=8, line_width=2):
    """
    Fit segments to the canvas with uniform scaling and padding, then rasterize.
    Returns a binary mask numpy array: 1 where path is, 0 elsewhere.
    """
    if not segments:
        return np.zeros(canvas, dtype=np.uint8)

    # Collect bounds
    xs, ys = [], []
    for x0, y0, x1, y1 in segments:
        xs.extend([x0, x1]); ys.extend([y0, y1])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Handle degenerate bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)

    W, H = canvas
    sx = (W - 2 * margin) / width
    sy = (H - 2 * margin) / height
    scale = min(sx, sy)

    def to_px(x, y):
        px = margin + (x - minx) * scale
        # Invert y for image coords (top-down)
        py = H - (margin + (y - miny) * scale)
        return px, py

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in segments:
        p0 = to_px(x0, y0)
        p1 = to_px(x1, y1)
        draw.line([p0, p1], fill=255, width=line_width)

    mask = np.array(img, dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8)
    return mask

def build_lsystem_mask(iterations=2, canvas=(256, 256)):
    """
    Example: classic plant-like L-system with 2 iterations.
    You can swap to other L-systems; the env/agent code does not change.
    """
    axiom = "X"
    rules = {
        "X": "F+[[X]-X]-F[-FX]+X",
        "F": "FF",
    }
    cmds = expand_lsystem(axiom, rules, iterations)
    segs = lsystem_segments(cmds, step=10.0, angle_deg=25.0, start_pos=(0.0, 0.0), start_angle=90.0)
    mask = segments_to_mask(segs, canvas=canvas, margin=8, line_width=2)
    with open ("mask.npy", "wb") as f:
        np.save(f, mask)
    return mask

# ---------------------------
# 2) Grid environment with 8 actions and 3x3 local observation
# ---------------------------

ACTIONS = [
    (-1,  0),  # 0: N
    ( 1,  0),  # 1: S
    ( 0,  1),  # 2: E
    ( 0, -1),  # 3: W
    (-1,  1),  # 4: NE
    (-1, -1),  # 5: NW
    ( 1,  1),  # 6: SE
    ( 1, -1),  # 7: SW
]
ACTION_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

class PathEnv:
    """
    - grid: binary path mask (H x W) where 1 = path pixel, 0 = off-path
    - visited: marks path pixels visited by the agent
    - reward:
        +1: step into an unvisited path pixel
         0: step into a visited path pixel
        -1: step into an off-path pixel (rejected; agent stays)
    - observation: 3x3 patch around agent, encoded as {0 off, 1 unvisited path, 2 visited path}
    - episode ends when all path pixels are visited
    """

    def __init__(self, path_mask):
        self.path = (path_mask.astype(np.uint8) > 0).astype(np.uint8)
        self.H, self.W = self.path.shape
        self.path_total = int(self.path.sum())
        if self.path_total == 0:
            raise ValueError("Path mask is empty.")
        self.rng = random.Random(123)

        # Compute a default start position: take a path pixel near the bottom
        ys, xs = np.where(self.path == 1)
        idx = np.argmax(ys)  # bottom-most path pixel
        
        self.start = (int(ys[idx]), int(xs[idx]))
        print()
        print(self.start)
        print()

        self.reset()

    def reset(self):
        self.visited = np.zeros_like(self.path, dtype=np.uint8)
        self.pos = self.start
        # Mark start visited immediately (we are standing on it)
        y, x = self.pos
        self.visited[y, x] = 1
        self.visited_count = int(self.visited.sum())
        return self._obs()

    def _encode_patch(self, y, x):
        """
        3x3 patch values in {0,1,2}. Off-grid treated as 0.
        Return both patch and an integer state_id in base-3 (3^9 states).
        """
        vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < self.H and 0 <= xx < self.W:
                    if self.path[yy, xx] == 0:
                        vals.append(0)
                    else:
                        vals.append(2 if self.visited[yy, xx] == 1 else 1)
                else:
                    vals.append(0)
        # Base-3 encoding
        state_id = 0
        base = 1
        for v in vals:
            state_id += v * base
            base *= 3
        return np.array(vals, dtype=np.uint8).reshape(3, 3), state_id

    def _obs(self):
        patch, sid = self._encode_patch(*self.pos)
        return sid, patch

    def step(self, action_idx):
        dy, dx = ACTIONS[action_idx]
        y, x = self.pos
        ny, nx = y + dy, x + dx

        # Off-grid → treat as off-path
        if not (0 <= ny < self.H and 0 <= nx < self.W):
            # reject move
            reward = -1.0
            # done = self.visited_count == self.path_total
            done = (self.path_total - self.visited_count)/self.path_total * 100 < 5
            sid, patch = self._obs()
            return (sid, patch), reward, done, {"moved": False, "offpath": True}

        # Off-path → penalize and reject move
        if self.path[ny, nx] == 0:
            reward = -1.0
            done = self.visited_count == self.path_total
            sid, patch = self._obs()
            return (sid, patch), reward, done, {"moved": False, "offpath": True}

        # On-path: move there
        self.pos = (ny, nx)

        # Reward depends on visited status
        if self.visited[ny, nx] == 0:
            reward = 1.0
            self.visited[ny, nx] = 1
            self.visited_count += 1
        else:
            reward = 0.0

        done = self.visited_count == self.path_total
        sid, patch = self._obs()
        return (sid, patch), reward, done, {"moved": True, "offpath": False}

    def coverage(self):
        return self.visited_count / self.path_total

    def render(self, show_agent=True, ax=None, title=None):
        """
        Visualize path (gray), visited (green), and agent (red).
        """
        img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        # Path in light gray
        img[self.path == 1] = [0.8, 0.8, 0.8]
        # Visited in green overlay
        visited_mask = (self.visited == 1)
        img[visited_mask] = [0.1, 0.7, 0.3]
        # Agent in red
        if show_agent:
            y, x = self.pos
            img[y, x] = [0.9, 0.1, 0.1]

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax

# ---------------------------
# 3) Tabular Q-learning agent over local 3x3 states
# ---------------------------

class QLearningAgent:
    def __init__(self, n_states=3**9, n_actions=8, alpha=0.3, gamma=0.99, eps_start=0.2, eps_end=0.01, eps_decay=0.999):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = n_actions

    def select_action(self, state_id):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.Q[state_id]
        # break ties randomly for stability
        maxq = np.max(q)
        candidates = np.flatnonzero(np.isclose(q, maxq))
        return int(random.choice(candidates))

    def update(self, s, a, r, s_next, done):
        qsa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - qsa)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

# ---------------------------
# 4) Train and evaluate
# ---------------------------

def train_on_mask(mask, episodes=500, max_steps=100000, seed=0, verbose_every=50):
    random.seed(seed); np.random.seed(seed)
    env = PathEnv(mask)
    agent = QLearningAgent()

    rewards_history = []
    coverage_history = []
    steps_history = []

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        total_r = 0.0
        steps = 0
        while steps < max_steps:
            a = agent.select_action(s)
            (s_next, _), r, done, info = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
            steps += 1
            if done:
                break
        agent.decay_eps()
        rewards_history.append(total_r)
        coverage_history.append(env.coverage())
        steps_history.append(steps)
        if verbose_every and ep % verbose_every == 0:
            print(f"Episode {ep:4d}: reward={total_r:.1f}, steps={steps}, coverage={coverage_history[-1]*100:.1f}%, eps={agent.eps:.3f}")

    return env, agent, {
        "rewards": np.array(rewards_history),
        "coverage": np.array(coverage_history),
        "steps": np.array(steps_history),
    }

def evaluate_greedy(env, agent, max_steps=100000):
    s, _ = env.reset()
    total_r = 0.0
    traj = [env.pos]
    for _ in range(max_steps):
        # Greedy (no exploration)
        q = agent.Q[s]
        a = int(np.argmax(q))
        (s, _), r, done, _ = env.step(a)
        total_r += r
        traj.append(env.pos)
        if done:
            break
    return total_r, traj

# ---------------------------
# 5) Demo: build L-system mask, train, and visualize
# ---------------------------

if __name__ == "__main__":
    # Build a connected L-system structure after 2 iterations (plant-like)
    mask = build_lsystem_mask(iterations=2, canvas=(256, 256))

    # Train the agent
    env, agent, logs = train_on_mask(mask, episodes=200, max_steps=1000, verbose_every=50)

    # Evaluate greedily
    total_r, traj = evaluate_greedy(env, agent)

    print(f"\nGreedy evaluation: total_reward={total_r:.1f}, coverage={env.coverage()*100:.1f}% "
          f"(visited {env.visited_count}/{env.path_total})")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(logs["rewards"])
    axes[0].set_title("Episode Reward"); axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")

    axes[1].plot(logs["coverage"] * 100)
    axes[1].set_title("Coverage (%)"); axes[1].set_xlabel("Episode"); axes[1].set_ylabel("% Path Visited")

    axes[2].plot(logs["steps"])
    axes[2].set_title("Episode Steps"); axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Steps")
    plt.tight_layout()
    plt.show()

    # Visualize final path traversal
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Before evaluation (just path)
    env.render(show_agent=False, ax=ax[0], title="Path (mask)")
    # Overlay greedy traversal
    env.render(show_agent=False, ax=ax[1], title="Visited after Greedy Run")
    plt.tight_layout()
    plt.show()

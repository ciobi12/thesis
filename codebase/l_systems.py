# lsystem_plant.py
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from typing import Dict, Tuple, List

class LSystemGenerator:
    def __init__(self, axiom: str, rules: Dict[str, str]):
        self.axiom = axiom
        self.rules = rules
        self.segments = []
        self.iterations = None

    def build_l_sys(self,
                    iterations: int,
                    step: float = 10.0, 
                    start_angle: float = 90.0,
                    angle_deg: float = 25.0,
                    start_pos = (0.0, 0.0)
                    ) -> List[Tuple[int, int, int, int]]:
        
        """Interpret commands into a list of line segments [(x0,y0,x1,y1), ...].

        Args:
            iterations (int): _description_
            step (float, optional): _description_. Defaults to 10.0.
            start_angle (float, optional): _description_. Defaults to 90.0.
            angle_deg (float, optional): _description_. Defaults to 25.0.
            start_pos (tuple, optional): _description_. Defaults to (0.0, 0.0).

        Returns:
            List[Tuple[int, int, int, int]]: _description_
        """
        self.iterations = iterations

        s = self.axiom
        for _ in range(self.iterations):
            s = "".join(self.rules.get(ch, ch) for ch in s)

        x, y = start_pos
        heading = math.radians(start_angle)
        stack = []
     
        for ch in s:
            if ch == "F":  # draw forward
                next_x = x + step * math.cos(heading)
                next_y = y + step * math.sin(heading)
                self.segments.append((x, y, next_x, next_y))
                x, y = next_x, next_y
            elif ch == "f":  # move forward without drawing
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
            # ignore other symbols like X/Y used for expansion only
        return self.segments
    
    def draw_lsystem(self, line_color="#000000FF", figsize=(8, 8), line_width = 2) -> None:
        """Draw L-system segments with matplotlib.

        Args:
            line_color (str, optional): _description_. Defaults to "#000000FF".
            figsize (tuple, optional): _description_. Defaults to (8, 8).
            line_width (int, optional): _description_. Defaults to 2.
        """
        fig, ax = plt.subplots(figsize=figsize)
        if self.segments:
            xs = []
            ys = []
            for (x0, y0, x1, y1) in self.segments:
                xs.extend([x0, x1, None])  # None breaks the polyline
                ys.extend([y0, y1, None])
            ax.plot(xs, ys, color=line_color, linewidth=line_width, solid_capstyle='round')

        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"l-system-{self.iterations}-iterations.png")
        plt.show()

    def build_mask(self, canvas_size=(256, 256), margin=8, line_width=2) -> np.ndarray:
        """
        Fit segments to the canvas with uniform scaling and padding, then rasterize.
        Returns a binary mask (H, W) where 1=path, 0=background.
        """

        xs, ys = [], []
        for x0, y0, x1, y1 in self.segments:
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)

        W, H = canvas_size
        sx = (W - 2 * margin) / width
        sy = (H - 2 * margin) / height
        scale = min(sx, sy)

        def to_px(x, y):
            px = margin + (x - minx) * scale
            # invert y for image coordinates (top-down)
            py = H - (margin + (y - miny) * scale)
            return px, py

        img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1 in self.segments:
            p0 = to_px(x0, y0)
            p1 = to_px(x1, y1)
            draw.line([p0, p1], fill=255, width=line_width)

        mask = np.array(img, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        with open("mask.npy", "wb") as f:
            np.save(f, mask)
        return mask
    
if __name__ == "__main__":
    ## Test ##
    lsys_obj = LSystemGenerator(axiom = "X",
                                rules = {"X": "F+[[X]-X]-F[-FX]+X",
                                         "F": "FF"
                                         }
                                )
    
    iterations = 3
    angle = 22.5
    step = 5

    segments = lsys_obj.build_l_sys(iterations = iterations, step = step, angle_deg = angle)
    lsys_obj.draw_lsystem()
    mask = lsys_obj.build_mask()
# lsystem_plant.py
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

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
                    step: int = 10, 
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
    
    def draw_lsystem(self, 
                     canvas_size=(256, 256), 
                     margin=8, 
                     line_width=2,
                     lsys_save_path="lsystem_ct.png", 
                     mask_save_path :str = "lsystem_mask.png", 
                     add_noise=True, 
                     add_artifacts=True) -> Tuple[np.array, np.array]:
        """
        Rasterize the L-system path into a grayscale image resembling a CT slice.
        - Path pixels: random intensities in [64, 255]
        - Background: 0
        - Optional noise and artifacts
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
            px = int(margin + (x - minx) * scale)
            py = int(H - (margin + (y - miny) * scale))
            return px, py

        # Start with black background
        pil_img = Image.new("L", (W, H), 0)
        pil_img_mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(pil_img)
        draw_mask = ImageDraw.Draw(pil_img_mask)

        # Draw path with varying intensities
        for x0, y0, x1, y1 in self.segments:
            p0 = to_px(x0, y0)
            p1 = to_px(x1, y1)
            intensity = random.randint(90, 255)
            draw.line([p0, p1], fill=intensity, width=line_width)
            draw_mask.line([p0, p1], fill=255, width=line_width)

        # Clean L-system image
        img = np.array(pil_img, dtype=np.uint8)

        # Mask as npy, also store it as .png
        mask = np.array(img, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        with open("mask.npy", "wb") as f:
            np.save(f, mask)
        cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))

        # Add noise
        if add_noise:
            # Gaussian noise
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Salt & pepper
            prob = 0.05
            rnd = np.random.rand(*img.shape)
            noisy[rnd < prob/2] = 0
            noisy[rnd > 1 - prob/2] = 255
            img = noisy

        # Add artifacts (random circles)
        if add_artifacts:
            num_circles = random.randint(5, 15)
            for _ in range(num_circles):
                cx, cy = random.randint(0, W-1), random.randint(0, H-1)
                r = random.randint(3, 9)
                val = random.randint(64, 255)
                cv2.circle(img, (cx, cy), r, val, -1)

        # Save final image
        cv2.imwrite(lsys_save_path, img)
        return img, mask

    # def build_mask(self, canvas_size=(256, 256), margin=8, line_width=2, save_path = "mask.png") -> np.ndarray:
    #     """
    #     Fit segments to the canvas with uniform scaling and padding, then rasterize.
    #     Returns a binary mask (H, W) where 1=path, 0=background.
    #     """

    #     xs, ys = [], []
    #     for x0, y0, x1, y1 in self.segments:
    #         xs.extend([x0, x1])
    #         ys.extend([y0, y1])
    #     minx, maxx = min(xs), max(xs)
    #     miny, maxy = min(ys), max(ys)

    #     width = max(maxx - minx, 1e-6)
    #     height = max(maxy - miny, 1e-6)

    #     W, H = canvas_size
    #     sx = (W - 2 * margin) / width
    #     sy = (H - 2 * margin) / height
    #     scale = min(sx, sy)

    #     def to_px(x, y):
    #         px = margin + (x - minx) * scale
    #         # invert y for image coordinates (top-down)
    #         py = H - (margin + (y - miny) * scale)
    #         return px, py

    #     img = Image.new("L", (W, H), 0)
    #     draw = ImageDraw.Draw(img)
    #     for x0, y0, x1, y1 in self.segments:
    #         p0 = to_px(x0, y0)
    #         p1 = to_px(x1, y1)
    #         draw.line([p0, p1], fill=255, width=line_width)

    #     mask = np.array(img, dtype=np.uint8)
    #     mask = (mask > 0).astype(np.uint8)
    #     with open("mask.npy", "wb") as f:
    #         np.save(f, mask)
    #     cv2.imwrite(save_path, (mask * 255).astype(np.uint8))
    #     return mask
    
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
    img, mask = lsys_obj.draw_lsystem(canvas_size=(128, 256), 
                                line_width=2,
                                lsys_save_path="lsystem_ct.png", 
                                mask_save_path = "mask.png")

    # mask = lsys_obj.build_mask()
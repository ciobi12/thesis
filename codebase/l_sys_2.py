import cv2
import random

class LSystemGenerator:
    # ... keep your existing __init__ and build_l_sys ...

    def draw_lsystem(self, canvas_size=(256, 256), margin=8, line_width=2,
                     save_path="lsystem_ct.png", add_noise=True, add_artifacts=True):
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
        img = np.zeros((H, W), dtype=np.uint8)
        draw = ImageDraw.Draw(Image.fromarray(img))

        # Draw path with varying intensities
        for x0, y0, x1, y1 in self.segments:
            p0 = to_px(x0, y0)
            p1 = to_px(x1, y1)
            intensity = random.randint(64, 255)
            draw.line([p0, p1], fill=intensity, width=line_width)

        img = np.array(draw.im, dtype=np.uint8)

        # Add noise
        if add_noise:
            # Gaussian noise
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Salt & pepper
            prob = 0.01
            rnd = np.random.rand(*img.shape)
            noisy[rnd < prob/2] = 0
            noisy[rnd > 1 - prob/2] = 255
            img = noisy

        # Add artifacts (random circles)
        if add_artifacts:
            num_circles = random.randint(5, 15)
            for _ in range(num_circles):
                cx, cy = random.randint(0, W-1), random.randint(0, H-1)
                r = random.randint(3, 15)
                val = random.randint(64, 255)
                cv2.circle(img, (cx, cy), r, val, -1)

        cv2.imwrite(save_path, img)
        return img
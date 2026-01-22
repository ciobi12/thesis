# lsystem_plant.py
import cv2
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Tuple, List

class LSystem2DGenerator:
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
        
        """Interpret commands into a list of line segments [(x0,y0,x1,y1), ...]."""
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
        return self.segments
    
    def draw_lsystem_ct_style(self, 
                          canvas_size=(512, 512), 
                          margin=40, 
                          root_width=3,
                          lsys_save_path="lsystem_ct.png", 
                          mask_save_path="lsystem_mask.png",
                          add_ct_noise=True,
                          occlude_root=False,
                          occlusion_strength=0.3,
                          ring_prob = 0.1,
                          skip_segments=False,
                          skip_probability=0.2,
                          ct_background_intensity=80,  # Darker gray like reference
                          root_intensity_range=(10, 40),  # Darker roots
                          align_top=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a CT-scan style image with root structure.
        
        Args:
            canvas_size: Output image size (W, H)
            margin: Margin around the root structure
            root_width: Thickness of root lines
            lsys_save_path: Path to save the CT-style image
            mask_save_path: Path to save the binary mask
            add_ct_noise: Whether to add CT-like noise and artifacts
            occlude_root: Whether to partially occlude the root with noise
            occlusion_strength: Strength of occlusion (0.0-1.0), higher = more hidden
            skip_segments: Whether to randomly skip segments (creates natural gaps)
            skip_probability: Probability of skipping each segment (0.0-1.0)
            ct_background_intensity: Gray level for background (0-255, ~80 for dark CT)
            root_intensity_range: (min, max) intensity for roots (darker than background)
            align_top: If True, align root to start from top edge
        
        Returns:
            Tuple of (ct_image, binary_mask)
        """
        # Calculate bounding box
        xs, ys = [], []
        for x0, y0, x1, y1 in self.segments:
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        
        if not xs:
            print("Warning: No segments to draw")
            return np.zeros(canvas_size[::-1], dtype=np.uint8), np.zeros(canvas_size[::-1], dtype=np.uint8)
        
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)

        W, H = canvas_size
        
        # Calculate scale
        sx = (W - 2 * margin) / width
        sy = (H - 2 * margin) / height
        scale = min(sx, sy)

        def to_px(x, y):
            px = int(margin + (x - minx) * scale)
            if align_top:
                # Align to top: minimum Y maps to margin (top of canvas)
                py = int(margin + (y - miny) * scale)
            else:
                # Original: flip Y coordinate
                py = int(H - (margin + (y - miny) * scale))
            return px, py

        # Create base CT background (darker gray like reference)
        img = np.full((H, W), ct_background_intensity, dtype=np.uint8)
        
        # Create mask (binary: 0 = background, 255 = root)
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Convert to PIL for drawing
        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil_img)
        draw_mask = ImageDraw.Draw(pil_mask)

        # Track skipped segments
        num_skipped = 0
        num_drawn = 0

        # Draw root structure (darker than background)
        for x0, y0, x1, y1 in self.segments:
            # Randomly skip segments to create natural gaps
            if skip_segments and random.random() < skip_probability:
                num_skipped += 1
                continue  # Skip this segment entirely
            
            p0 = to_px(x0, y0)
            p1 = to_px(x1, y1)
            
            # Roots appear darker (lower intensity) than surrounding tissue in CT
            root_intensity = random.randint(*root_intensity_range)
            
            # Draw segment
            draw.line([p0, p1], fill=root_intensity, width=root_width)
            draw_mask.line([p0, p1], fill=255, width=root_width)
            num_drawn += 1

        # Convert back to numpy
        img = np.array(pil_img, dtype=np.uint8)
        mask = np.array(pil_mask, dtype=np.uint8)

        # Add CT-style noise and artifacts
        if add_ct_noise:
            # 1. Gaussian noise (typical in CT) - stronger for darker background
            noise_std = 12  # Increased for visibility on darker background
            gaussian_noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + gaussian_noise, 0, 255).astype(np.uint8)
            
            # 2. Salt & pepper noise (sporadic artifacts)
            salt_pepper_prob = 0.003  # Slightly increased
            rnd = np.random.rand(*img.shape)
            img[rnd < salt_pepper_prob] = 255  # bright spots (adjusted for darker bg)
            img[rnd > 1 - salt_pepper_prob] = 0   # dark spots
            
            # 3. Subtle texture variation (tissue heterogeneity)
            texture = np.random.normal(0, 5, img.shape).astype(np.int16)  # Increased
            img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
            
            # 4. Reduce ring artifacts for more realistic look
            if random.random() < ring_prob:  # Reduced from 0.3
                center = (W // 2, H // 2)
                num_rings = random.randint(1, 3)
                for _ in range(num_rings):
                    radius = random.randint(50, min(W, H) // 2)
                    ring_intensity = random.randint(-8, 8)
                    y_coords, x_coords = np.ogrid[:H, :W]
                    distance = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
                    ring_mask = (distance > radius - 2) & (distance < radius + 2)
                    img[ring_mask] = np.clip(img[ring_mask].astype(np.int16) + ring_intensity, 0, 255).astype(np.uint8)
            
            # 5. Slight blur (mimicking CT resolution)
            img = cv2.GaussianBlur(img, (3, 3), 0.7)

        # 6. Occlude the root if requested
        if occlude_root:
            # Create occlusion mask - stronger noise/artifacts where root exists
            root_regions = mask > 0
            
            # Add heavy noise to root regions
            heavy_noise_std = int(25 * occlusion_strength)
            heavy_noise = np.random.normal(0, heavy_noise_std, img.shape).astype(np.int16)
            
            # Apply occlusion only where root exists
            img_occluded = img.copy().astype(np.int16)
            img_occluded[root_regions] += heavy_noise[root_regions]
            
            # Add random bright/dark patches on root (more subtle for darker background)
            occlusion_patches = np.random.rand(*img.shape) < (0.08 * occlusion_strength)
            root_occlusion = root_regions & occlusion_patches
            
            # Patches closer to background intensity
            patch_intensity_range = int(40 * occlusion_strength)
            img_occluded[root_occlusion] = np.random.randint(
                max(0, ct_background_intensity - patch_intensity_range), 
                min(255, ct_background_intensity + patch_intensity_range), 
                size=np.sum(root_occlusion)
            )
            
            # Blend with background intensity to partially hide root
            blend_factor = occlusion_strength * 0.4  # Reduced for darker background
            img_occluded[root_regions] = (
                blend_factor * ct_background_intensity + 
                (1 - blend_factor) * img_occluded[root_regions]
            ).astype(np.int16)
            
            img = np.clip(img_occluded, 0, 255).astype(np.uint8)
            
            # Add extra blur to occluded regions for realism
            occlusion_blur = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.0)
            img[root_regions] = (
                0.25 * occlusion_blur[root_regions] + 
                0.75 * img[root_regions]
            ).astype(np.uint8)

        # Save images
        cv2.imwrite(lsys_save_path, img)
        cv2.imwrite(mask_save_path, mask)
        
        # Save mask as numpy
        np.save(mask_save_path.replace('.png', '.npy'), (mask > 0).astype(np.uint8))
        
        print(f"CT-style image saved to: {lsys_save_path}")
        print(f"Mask saved to: {mask_save_path}")
        print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        print(f"Root pixels: {(mask > 0).sum()} / {mask.size} ({100 * (mask > 0).sum() / mask.size:.2f}%)")
        if skip_segments:
            print(f"Segments: {num_drawn} drawn, {num_skipped} skipped ({100*num_skipped/(num_drawn+num_skipped):.1f}% skip rate)")
        if occlude_root:
            print(f"Occlusion applied with strength: {occlusion_strength}")

        return img, mask

    def draw_lsystem(self, 
                     canvas_size=(256, 256), 
                     margin=8, 
                     line_width=2,
                     lsys_save_path="lsystem_ct.png", 
                     mask_save_path="lsystem_mask.png", 
                     add_noise=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        DEPRECATED: Use draw_lsystem_ct_style() instead.
        Legacy method for backward compatibility.
        """
        return self.draw_lsystem_ct_style(
            canvas_size=canvas_size,
            margin=margin,
            root_width=line_width,
            lsys_save_path=lsys_save_path,
            mask_save_path=mask_save_path,
            add_ct_noise=add_noise
        )


if __name__ == "__main__":
    ## Root-like L-system for CT simulation ##
    
    # Root system parameters
    lsys_obj = LSystem2DGenerator(
        axiom="F",
        rules={
            "F": "F[+F]F[-F]F",  # Branching pattern similar to roots
        }
    )
    examples_folder = "examples/simple_axiom"
    # lsys_obj = LSystem2DGenerator(
    #     axiom = "X",
    #     rules = {"X": "F-[[X]+X]+F[+FX]-X",
    #              "F": "FF"},
    # )
    # examples_folder = "examples/complex_axiom"
    
    iterations = 2
    angle = 25.0
    step = 8

    # Build the L-system (pointing downward)
    segments = lsys_obj.build_l_sys(
        iterations=iterations, 
        step=step, 
        angle_deg=angle,
        start_angle=-90  # Point downward like roots
    )

    canvas_size = (128, 192)
    root_width = 1
    
    ct_bg_intensity = 80  # Dark gray like reference, fixed value
    margin = 10
    
    print(f"Generated {len(segments)} segments")
    
    # Create different versions matching reference image style
    print("\n=== 1. Clear (no gaps, dark background) ===")
    img_clear, mask_clear = lsys_obj.draw_lsystem_ct_style(
        canvas_size = canvas_size,
        margin = margin,
        root_width = root_width,
        lsys_save_path = f"{examples_folder}/root_ct_clear.png",
        mask_save_path = f"{examples_folder}/root_mask.png",
        add_ct_noise = True,
        occlude_root = False,
        skip_segments = False,
        ct_background_intensity = ct_bg_intensity,  # Dark gray like reference
        root_intensity_range = (10, 40),
        align_top=True
    )
    
    print("\n=== 2. Extra blur ===")
    img_realistic, mask_realistic = lsys_obj.draw_lsystem_ct_style(
        canvas_size=canvas_size,
        margin=20,
        root_width=root_width,
        lsys_save_path=f"{examples_folder}/root_ct_occluded.png",
        mask_save_path=f"{examples_folder}/root_mask.png",
        add_ct_noise=True,
        occlude_root=True,
        occlusion_strength=0.5,  # Moderate occlusion
        skip_segments=False,
        ct_background_intensity=ct_bg_intensity,
        root_intensity_range=(10, 40),
        align_top=True
    )
    
    # Visualize all versions
    fig, axes = plt.subplots(1, 3, figsize=(10, 20))
    axes[0].imshow(img_clear, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Clear, no occlusion')
    axes[0].axis('off')

    axes[1].imshow(img_realistic, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Extra blur')
    axes[1].axis('off')
        
    axes[2].imshow(mask_clear, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('examples/root_realistic_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== All images saved! ===")
    print("Parameters for reference-like appearance:")
    print("- ct_background_intensity: 80 (dark gray)")
    print("- root_intensity_range: (10, 40) (darker than background)")
    print("- occlusion_strength: 0.3-0.6 (realistic noise)")
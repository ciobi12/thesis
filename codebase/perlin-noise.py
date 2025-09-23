"""
add_perlin.py

Dependencies:
  pip install pillow numpy

Usage example:
  python add_perlin.py input.jpg output.jpg --scale 6 --octaves 4 --intensity 0.45 --mode overlay --color
"""

import argparse
import math
import numpy as np
from PIL import Image
import random


# --- Perlin noise implementation (2D) ---
def lerp(a, b, t):
    return a + t * (b - a)


def fade(t):
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_gradients(grid_x, grid_y, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    angles = rng.rand(grid_y + 1, grid_x + 1) * 2 * math.pi
    gx = np.cos(angles)
    gy = np.sin(angles)
    return gx, gy


def perlin_noise_2d(width, height, scale=8.0, seed=None):
    """
    Single-octave Perlin noise in range [-1, 1].
    scale - number of grid cells across the image (~frequency).
    """
    # Number of gradient grid cells
    grid_x = int(max(1, scale))
    grid_y = int(max(1, scale * height / width))

    gx, gy = generate_gradients(grid_x, grid_y, seed)
    # Coordinates of pixels in grid space
    xs = np.linspace(0, grid_x, width, endpoint=False)
    ys = np.linspace(0, grid_y, height, endpoint=False)
    x_grid, y_grid = np.meshgrid(xs, ys)
    # integer lattice coordinates
    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    # local coords
    xf = x_grid - x0
    yf = y_grid - y0

    # wrap indices safely (gradients are sized grid+1)
    x1 = x0 + 1
    y1 = y0 + 1

    # gather gradient vectors at corners
    g00x = gx[y0, x0]
    g00y = gy[y0, x0]
    g10x = gx[y0, x1]
    g10y = gy[y0, x1]
    g01x = gx[y1, x0]
    g01y = gy[y1, x0]
    g11x = gx[y1, x1]
    g11y = gy[y1, x1]

    # vectors from corners to point
    dx00 = xf
    dy00 = yf
    dx10 = xf - 1
    dy10 = yf
    dx01 = xf
    dy01 = yf - 1
    dx11 = xf - 1
    dy11 = yf - 1

    # dot products
    dot00 = g00x * dx00 + g00y * dy00
    dot10 = g10x * dx10 + g10y * dy10
    dot01 = g01x * dx01 + g01y * dy01
    dot11 = g11x * dx11 + g11y * dy11

    # interpolate
    ux = fade(xf)
    uy = fade(yf)

    ix0 = lerp(dot00, dot10, ux)
    ix1 = lerp(dot01, dot11, ux)
    value = lerp(ix0, ix1, uy)

    # value in [-sqrt(2)/2, sqrt(2)/2], roughly -1..1 scale anyway
    return value


def octave_perlin(
    width, height, base_scale=8.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=None
):
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = base_scale
    max_ampl = 0.0
    for o in range(octaves):
        n = perlin_noise_2d(
            width,
            height,
            scale=frequency,
            seed=None if seed is None else seed + o * 131,
        )
        total += n * amplitude
        max_ampl += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    # normalize to [-1, 1]
    total = total / max_ampl
    return total


# --- blending helpers ---
def blend_add(img, noise, intensity=0.5):
    # img: float 0..1, noise: -1..1
    return np.clip(img + intensity * noise, 0.0, 1.0)


def blend_multiply(img, noise, intensity=0.5):
    # map noise to 0..2 then multiply
    factor = 1.0 + intensity * noise
    return np.clip(img * factor, 0.0, 1.0)


def blend_overlay(img, noise, intensity=0.5):
    # overlay using luminance; simple approximation:
    n = (noise + 1.0) / 2.0  # 0..1
    out = img.copy()
    mask = img <= 0.5
    out[mask] = 2 * img[mask] * (n[mask] * intensity + (1 - intensity))
    out[~mask] = 1 - 2 * (1 - img[~mask]) * (
        1 - (n[~mask] * intensity + (1 - intensity))
    )
    return np.clip(out, 0.0, 1.0)


# --- main function to apply noise to image ---
def add_perlin_to_image(
    input_path,
    output_path,
    base_scale=8.0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    intensity=0.45,
    mode="overlay",
    color_noise=False,
    seed=None,
):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img).astype(np.float32) / 255.0

    noise = octave_perlin(
        w,
        h,
        base_scale=base_scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed,
    )
    # noise is [-1,1]; if color_noise, create 3 channels with slightly different seeds
    if color_noise:
        ns = np.stack(
            [
                octave_perlin(
                    w,
                    h,
                    base_scale * 1.0,
                    octaves,
                    persistence,
                    lacunarity,
                    None if seed is None else seed + 1,
                ),
                octave_perlin(
                    w,
                    h,
                    base_scale * 1.1,
                    octaves,
                    persistence,
                    lacunarity,
                    None if seed is None else seed + 2,
                ),
                octave_perlin(
                    w,
                    h,
                    base_scale * 0.9,
                    octaves,
                    persistence,
                    lacunarity,
                    None if seed is None else seed + 3,
                ),
            ],
            axis=-1,
        )
        noise3 = ns
    else:
        noise3 = np.repeat(noise[..., None], 3, axis=2)

    if mode == "add":
        out = blend_add(arr, noise3, intensity)
    elif mode == "multiply":
        out = blend_multiply(arr, noise3, intensity)
    else:
        out = blend_overlay(arr, noise3, intensity)

    out_img = Image.fromarray((out * 255).astype(np.uint8))
    out_img.save(output_path)
    print(f"Saved: {output_path}")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Perlin noise to an image.")
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")
    parser.add_argument(
        "--scale",
        type=float,
        default=8.0,
        help="base scale / frequency (larger = bigger blobs)",
    )
    parser.add_argument("--octaves", type=int, default=4, help="number of octaves")
    parser.add_argument("--persistence", type=float, default=0.5)
    parser.add_argument("--lacunarity", type=float, default=2.0)
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.45,
        help="how strong the noise effect is (0..1)",
    )
    parser.add_argument(
        "--mode", choices=["overlay", "add", "multiply"], default="overlay"
    )
    parser.add_argument(
        "--color", action="store_true", help="use colored noise (per-channel)"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    add_perlin_to_image(
        args.input,
        args.output,
        base_scale=args.scale,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        intensity=args.intensity,
        mode=args.mode,
        color_noise=args.color,
        seed=args.seed,
    )

"""Generate a random occupancy map with rectangular and circular obstacles.

Edit the parameters in the CONFIG block below and run the script
directly by path (not via ``-m scripts.X`` — the name ``scripts``
clashes with ROS' system-level package on Linux installs)::

    python scripts/generate_random_map.py

Two grayscale PNGs are written per run, sharing the *same* obstacles
(same seed, single generation pass):

* ``random_map_<n>.png`` — occupancy map: obstacles black (0), free
  space white (255).  Matches the loader in ``core.map_io`` (threshold
  at 128).
* ``random_map_<n>_height.png`` — height map: free space is 255 and
  each obstacle carries a random integer height in ``[0, MAX_HEIGHT]``
  (the pixel value *is* the height).  Where obstacles overlap, the
  taller one wins.  This matches ``core.map_io.load_height_map``, where
  ``0`` is a passable pit, ``1..max_height`` is an obstacle of that
  height, and ``> max_height`` (here 255) is free.  In this model a
  pit and a zero-height obstacle are the same thing.

A square of ``CORNER_CLEAR_SIZE`` px is cleared to free space at each of
the four map corners in both maps, so a robot formation can always be
placed there.

Files are written to ``random_maps/`` (repo root) with auto-incremented
names; both files of a run share the same ``<n>``.
"""

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- CONFIG
MAP_SIZE = (4000, 4000)          # (height, width) in pixels
N_RECTANGLES = 250               # number of rectangular obstacles
RECT_SIDE_RANGE = (20, 60)       # min/max side length (px), inclusive
N_CIRCLES = 250                 # number of circular obstacles
CIRCLE_RADIUS_RANGE = (20, 60)   # min/max radius (px), inclusive
MAX_HEIGHT = 100                 # obstacle heights drawn from [0, MAX_HEIGHT]
CORNER_CLEAR_SIZE = 400          # side (px) of the free square carved at each corner
SEED = None                      # int for reproducibility, None for random
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "random_maps")
OUTPUT_PREFIX = "random_map_"    # files written as random_map_<n>.png
HEIGHT_SUFFIX = "_height"        # height map: random_map_<n>_height.png
# -----------------------------------------------------------------------


FREE = 255
OBSTACLE = 0
FREE_HEIGHT = 255                # height-map value for free space (> MAX_HEIGHT)


def generate_map(map_size, n_rects, rect_side_range, n_circles,
                 radius_range, max_height, rng, corner_clear=0):
    """Generate occupancy + height maps sharing the same obstacles.

    Returns ``(occ, height)`` where ``occ`` is the binary occupancy map
    (obstacle=0, free=255) and ``height`` is the height map (free=255,
    each obstacle pixel = its height in ``[0, max_height]``).  Overlapping
    obstacles take the taller height (per-pixel maximum).

    If ``corner_clear > 0``, a square of that side length is cleared to
    free space at each of the four map corners, guaranteeing room to place
    a robot formation there.
    """
    H, W = map_size
    occ = np.full((H, W), FREE, dtype=np.uint8)
    # Track the max obstacle height per pixel; -1 marks free (untouched).
    # int16 holds heights and the -1 sentinel without uint wraparound.
    height_acc = np.full((H, W), -1, dtype=np.int16)

    def stamp(mask, h):
        occ[mask] = OBSTACLE
        np.maximum(height_acc, h, out=height_acc, where=mask)

    s_lo, s_hi = rect_side_range
    for _ in range(n_rects):
        h = int(rng.integers(s_lo, s_hi + 1))
        w = int(rng.integers(s_lo, s_hi + 1))
        y = int(rng.integers(0, max(1, H - h + 1)))
        x = int(rng.integers(0, max(1, W - w + 1)))
        ob_h = int(rng.integers(0, max_height + 1))
        rect_mask = np.zeros((H, W), dtype=bool)
        rect_mask[y:y + h, x:x + w] = True
        stamp(rect_mask, ob_h)

    r_lo, r_hi = radius_range
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n_circles):
        r = int(rng.integers(r_lo, r_hi + 1))
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        ob_h = int(rng.integers(0, max_height + 1))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        stamp(mask, ob_h)

    # Carve free squares at the four corners, overriding any obstacle that
    # landed there.  Clamp so an oversized square can't wrap around.
    c = min(corner_clear, H, W)
    if c > 0:
        for ys in (slice(0, c), slice(H - c, H)):
            for xs in (slice(0, c), slice(W - c, W)):
                occ[ys, xs] = FREE
                height_acc[ys, xs] = -1

    # Compose the height map: free stays 255, obstacles get their height.
    height = np.full((H, W), FREE_HEIGHT, dtype=np.uint8)
    obs_mask = height_acc >= 0
    height[obs_mask] = height_acc[obs_mask].astype(np.uint8)

    return occ, height


def next_output_index(output_dir, prefix):
    """Lowest unused index ``<n>`` for the occupancy-map file name."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.png$")
    existing = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            m = pattern.match(name)
            if m:
                existing.append(int(m.group(1)))
    return (max(existing) + 1) if existing else 1


def main():
    rng = np.random.default_rng(SEED)
    occ, height = generate_map(
        MAP_SIZE, N_RECTANGLES, RECT_SIDE_RANGE,
        N_CIRCLES, CIRCLE_RADIUS_RANGE, MAX_HEIGHT, rng,
        corner_clear=CORNER_CLEAR_SIZE,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = next_output_index(OUTPUT_DIR, OUTPUT_PREFIX)
    occ_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{n}.png")
    height_path = os.path.join(
        OUTPUT_DIR, f"{OUTPUT_PREFIX}{n}{HEIGHT_SUFFIX}.png")

    Image.fromarray(occ, mode="L").save(occ_path)
    Image.fromarray(height, mode="L").save(height_path)

    obs_mask = occ == OBSTACLE
    obs_frac = obs_mask.mean()
    if obs_mask.any():
        h_vals = height[obs_mask]
        h_info = (f"height range [{int(h_vals.min())}, "
                  f"{int(h_vals.max())}], mean {h_vals.mean():.1f}")
    else:
        h_info = "no obstacles"
    print(f"Saved {occ_path}  ({occ.shape[1]}x{occ.shape[0]}, "
          f"{obs_frac * 100:.2f}% occupied)")
    print(f"Saved {height_path}  ({h_info})")


if __name__ == "__main__":
    main()

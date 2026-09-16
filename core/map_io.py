"""Map I/O utilities.

Loads a grayscale occupancy image and returns the pair of arrays the
planners need: the binary occupancy mask and its Euclidean distance
transform.  Also exposes a helper that produces a *signed* clearance
field (positive in free space, negative inside low obstacles), used
by the label-aware pruning step of the planner.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def load_map(path, thresh=128):
    """Load a grayscale map image → (image, occupancy, distance-transform).

    Parameters
    ----------
    path : str
        Path to a grayscale image file (PNG/JPEG/…).
    thresh : int
        Pixels with value below ``thresh`` are treated as obstacles.

    Returns
    -------
    img : ndarray (H, W) uint8 — raw grayscale pixel values.
    occ : ndarray (H, W) bool  — True where obstacle.
    dist : ndarray (H, W) float — EDT of free space (px to nearest obs).
    """
    from PIL import Image
    img = np.array(Image.open(path).convert("L"))
    occ = img < thresh
    dist = distance_transform_edt(~occ)
    return img, occ, dist


def load_height_map(path, max_height=200):
    """Load a grayscale height map for payload / cable collision checks.

    Pixel convention (matches the demos' map images):

    * ``0`` — pit / hole; passable beneath the payload at any height.
    * ``1 .. max_height`` — vertical obstacle of that height (cm).
    * ``> max_height`` — treated as free space (no vertical
      obstruction); the typical case is ``255`` in binary maps.

    Out-of-range values are clamped to ``0`` so a single ``uint8``
    comparison ``height_map[y, x] > h_payload`` is sufficient at
    runtime — no sentinel-aware logic needed.

    Parameters
    ----------
    path : str — path to a grayscale image
    max_height : int — upper bound (inclusive) of values interpreted
        as obstacle height.  Anything above is mapped to free.

    Returns
    -------
    height_map : (H, W) uint8
    """
    from PIL import Image
    img = np.array(Image.open(path).convert("L"))
    free_mask = img > max_height
    out = img.astype(np.uint8, copy=True)
    out[free_mask] = 0
    return out


def compute_signed_clearance(occ, tall_obs_mask=None, tall_sentinel=-1e9):
    """Signed Euclidean clearance field from a binary occupancy map.

    Positive in the free region (distance to the nearest obstacle),
    negative inside obstacles (negated penetration depth).  Cells
    flagged in ``tall_obs_mask`` are clamped to ``tall_sentinel``
    (very negative) so every finite pruning threshold rejects them —
    this is how *tall* obstacles, which the formation centre cannot
    overlap at any scale, are encoded.

    Parameters
    ----------
    occ : (H, W) bool ndarray — True = obstacle.
    tall_obs_mask : (H, W) bool ndarray, optional
        Subset of ``occ`` to treat as tall.  When omitted, all
        obstacles are low (the centre may overlap them).
    tall_sentinel : float — value written into tall cells.

    Returns
    -------
    signed_clearance : (H, W) float64 ndarray
    """
    free = ~occ
    dist_outside = distance_transform_edt(free)        # > 0 in free, 0 in obstacles
    dist_inside = distance_transform_edt(occ)          # > 0 in obstacles, 0 in free
    signed = (dist_outside - dist_inside).astype(np.float64)

    if tall_obs_mask is not None:
        signed[tall_obs_mask] = float(tall_sentinel)

    return signed

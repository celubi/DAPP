"""Critical Crossable Obstacle model — scene utilities for CCO_planner.

The CCO_planner (Critical Crossable Obstacle Planner) plans on the
boundary of a single elongated obstacle that the formation must
straddle.  This module builds that scene representation:

* ``Obstacle`` — loads/inflates/samples an obstacle image, extracts
  its skeleton main branch, splits the inflated contour into L/R
  chains.  Internally relies on ``find_longest_path`` and
  ``find_split_point``.
* ``find_start`` — first anchor pair on the L/R chains separated by
  approximately ``BAR`` pixels (within tolerance).  Running it on the
  reversed chains gives the far end, so start and goal configurations
  are derived automatically at the two ends of the obstacle.
* ``make_clearance_map`` — exact EDT of the wall map (free pixels are
  > 127).  Returned as ``float32`` so the kernels' specialised JIT
  signatures keep their cached compilations.
* ``get_cluster_positions`` — positions of all three robots in each
  cluster (main + 2 satellites) on the formation circle of radius
  ``|AB|/2``.
* ``_circle_free`` — Numba-jitted scalar collision check.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import networkx as nx
from numba import njit
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, Point
from skimage.morphology import skeletonize


# ═══════════════════════════════════════════════════════════
#  Skeleton helpers
# ═══════════════════════════════════════════════════════════

def find_longest_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Return the 8-connected diameter path of a binary skeleton image."""
    skel = skeleton.astype(bool)
    G = nx.Graph()
    rows, cols = np.where(skel)
    for r, c in zip(rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                    G.add_edge((r, c), (rr, cc))

    if not G.nodes:
        return []

    start = next(iter(G.nodes))
    dist1 = nx.single_source_shortest_path_length(G, start)
    far1 = max(dist1, key=dist1.get)
    dist2 = nx.single_source_shortest_path_length(G, far1)
    far2 = max(dist2, key=dist2.get)
    path_rc = nx.shortest_path(G, far1, far2)
    return [(c, r) for (r, c) in path_rc]


def find_split_point(contour: np.ndarray, point: tuple[int, int],
                     radius: float) -> tuple[int, int]:
    """Pick a stable split point on the inflated contour near a centre point."""
    circle = Point(point).buffer(radius)

    buf1, buf2 = [], []
    old_i = 0
    buf = buf1
    switched = False

    for i in range(len(contour)):
        if circle.contains(Point(contour[i])):
            if i > 0 and i > old_i + 1 and not switched:
                buf = buf2
                switched = True
            buf.append(contour[i])
            old_i = i

    buf2.extend(buf1)
    if not buf2:
        d = np.linalg.norm(contour - np.array(point), axis=1)
        return tuple(contour[int(np.argmin(d))])
    return tuple(buf2[len(buf2) // 2])


# ═══════════════════════════════════════════════════════════
#  Obstacle wrapper
# ═══════════════════════════════════════════════════════════

class Obstacle:
    """Loads an obstacle image, inflates it, samples its boundary, and
    extracts its main skeleton branch."""

    def __init__(self, path: str, inflation_radius: int = 40,
                 step_size: int = 10, scale: float = 1):
        self._path = path
        self._inflation_radius = inflation_radius
        self._step_size = step_size
        self._scale = scale

        self._image_gray: Optional[np.ndarray] = None
        self._obstacle_polygon_true: Optional[Polygon] = None
        self._obstacle_polygon: Optional[Polygon] = None
        self._sampled: Optional[np.ndarray] = None
        self._main_skeleton_branch: Optional[list[tuple[int, int]]] = None

        self._load_obstacle_from_image()
        self._sample_polygon()
        self._compute_skeleton()

    def _load_obstacle_from_image(self) -> None:
        img = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read obstacle image: {self._path}")
        self._image_gray = img

        _, thresh = cv2.threshold(self._image_gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            raise ValueError("Expected at least 2 contours (background + obstacle)")

        self._obstacle_polygon_true = Polygon(contours[1].squeeze())
        self._obstacle_polygon = self._obstacle_polygon_true.buffer(
            distance=self._inflation_radius)

    def _sample_polygon(self) -> None:
        boundary = self._obstacle_polygon.boundary
        perimeter_length = boundary.length
        num_points = max(1, int(perimeter_length // self._step_size))
        sampled = [boundary.interpolate(i * self._step_size)
                   for i in range(num_points)]
        self._sampled = np.array(
            [(int(p.x * self._scale), int(p.y * self._scale)) for p in sampled],
            np.int32)

    def _compute_skeleton(self) -> None:
        mask = np.zeros_like(self._image_gray)
        coords = np.array(list(self._obstacle_polygon.exterior.coords),
                          dtype=np.int32)
        cv2.fillPoly(mask, [coords], color=255)
        skel = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8)
        self._main_skeleton_branch = find_longest_path(skel)

    def get_skeleton(self) -> list[tuple[int, int]]:
        """Main skeleton branch as a list of (x, y) pixel tuples."""
        return list(self._main_skeleton_branch or [])

    def split_obstacle(self) -> tuple[np.ndarray, np.ndarray]:
        """Split inflated polygon samples into two ordered chains using
        skeleton endpoints as anchors."""
        center_start = self._main_skeleton_branch[0]
        center_end = self._main_skeleton_branch[-1]
        radius = self._inflation_radius + 100

        start_point = find_split_point(self._sampled, center_start, radius)
        end_point = find_split_point(self._sampled, center_end, radius)

        temp = [tuple(row) for row in self._sampled]
        start_id = temp.index(tuple(start_point))
        end_id = temp.index(tuple(end_point))

        length = len(self._sampled)
        array = np.zeros_like(self._sampled)

        array[0:length - start_id] = self._sampled[start_id:]
        array[length - start_id:] = self._sampled[0:start_id]

        offset = abs(end_id - start_id)
        array_rx = array[0:offset]
        array_lx = np.flip(array[offset:], axis=0)

        return array_lx, array_rx


# ═══════════════════════════════════════════════════════════
#  Start-pair search
# ═══════════════════════════════════════════════════════════

def find_start(array_a: np.ndarray, array_b: np.ndarray, BAR: float,
               TOL: float) -> tuple[int, Optional[int]]:
    """Find the first index pair (0, j) such that |array_a[0] − array_b[j]|
    ≈ BAR within ±TOL."""
    id_a = 0
    id_b = None
    for j in range(len(array_b)):
        distance = np.linalg.norm(array_a[id_a] - array_b[j])
        if abs(distance - BAR) < TOL:
            id_b = j
            break
    return id_a, id_b


# ═══════════════════════════════════════════════════════════
#  Clearance map & cluster geometry
# ═══════════════════════════════════════════════════════════

def make_clearance_map(binary_map: np.ndarray) -> np.ndarray:
    """Distance-transform clearance map: value at (y, x) is the Euclidean
    distance in pixels to the nearest obstacle pixel.

    Pixels > 127 are free, ≤ 127 are obstacles.  Returned as
    ``float32`` so the kernels' specialised Numba signatures keep
    their cached compilations.
    """
    if binary_map.dtype != np.uint8:
        binary_map = binary_map.astype(np.uint8)
    free = binary_map > 127
    return distance_transform_edt(free).astype(np.float32)


def get_cluster_positions(A, B, dist_deg: float):
    """Positions of all 3 robots in each cluster on the formation circle.

    ``dist_deg`` is the chord length (px) between the main robot and
    each satellite, both lying on the formation circle of radius
    ``r = |AB|/2``.  The central angle subtended by that chord is
    ``2·asin(chord / (2·r))``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = (A + B) * 0.5
    r = np.linalg.norm(B - A) * 0.5

    if r < 1e-9:
        return [A.copy(), A.copy(), A.copy()], [B.copy(), B.copy(), B.copy()]

    ratio = max(0.0, min(1.0, float(dist_deg) / (2.0 * r)))
    alpha = 2.0 * math.asin(ratio)

    ang_A = math.atan2(A[1] - C[1], A[0] - C[0])
    ang_B = ang_A + math.pi

    A1 = C + r * np.array([math.cos(ang_A + alpha), math.sin(ang_A + alpha)])
    A2 = C + r * np.array([math.cos(ang_A - alpha), math.sin(ang_A - alpha)])
    B1 = C + r * np.array([math.cos(ang_B + alpha), math.sin(ang_B + alpha)])
    B2 = C + r * np.array([math.cos(ang_B - alpha), math.sin(ang_B - alpha)])

    return [A, A1, A2], [B, B1, B2]


# ═══════════════════════════════════════════════════════════
#  Scalar Numba kernel
# ═══════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True, inline='always')
def _circle_free(cx, cy, radius, clearance):
    """True iff a robot circle of given radius centred at (cx, cy) is clear."""
    h, w = clearance.shape
    xi = int(round(cx))
    yi = int(round(cy))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    return clearance[yi, xi] >= radius

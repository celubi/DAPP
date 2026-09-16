"""CCO_planner — high-level driver (Critical Crossable Obstacle Planner).

The single assembly point of the CCO pipeline, shared by the demo and
the benchmarks:

1. :func:`prepare_cco_scene` — step-size-INDEPENDENT preprocessing:
   wall image, clearance map, optional payload height map.
2. :func:`build_cco_chains` — step-size-DEPENDENT preprocessing:
   inflate the obstacle and split its boundary into the L/R anchor
   chains.
3. :func:`find_path_cco` — the "find path" phase, timed as one unit:
   prefilter → parallel kernel batch → reshape → start/goal
   resolution at the obstacle's two ends → mini A*.

Steps 1+2 are untimed preprocessing; ``result.t_search`` covers step 3
only — the same convention as
``da_astar.find_path_da_from_map(split_timing=True)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cco_obstacle import Obstacle, find_start, make_clearance_map
from .map_io import load_height_map
from .cco_kernels import optimise_anchors_batch, batch_to_arrays
from .cco_astar import astar_small_jit, resolve_node_id


# ═══════════════════════════════════════════════════════════
#  Scene preparation (untimed by the benchmarks)
# ═══════════════════════════════════════════════════════════

@dataclass
class CCOScene:
    """Step-size-independent scene data.

    ``height_map`` is a shape-correct dummy and ``do_payload == 0``
    when the payload / cable check is disabled, so the kernel keeps a
    single JIT signature either way.  ``timings`` holds the wall-clock
    seconds of each preparation sub-step (for display only).
    """
    wall_img: np.ndarray
    clearance: np.ndarray          # float32 EDT of the wall map
    height_map: np.ndarray
    do_payload: int
    timings: Dict[str, float] = field(default_factory=dict)


def prepare_cco_scene(wall_path, height_map_path=None, height_max=200):
    """Load the wall map, build its clearance map, optionally load the
    payload height map.

    This is the preprocessing that does NOT depend on the boundary
    step size, so per-map-set callers build it once and reuse it
    across step sizes.
    """
    t = time.perf_counter()
    wall_img = cv2.imread(str(wall_path), cv2.IMREAD_GRAYSCALE)
    if wall_img is None:
        raise FileNotFoundError(f"Cannot read wall map: {wall_path}")
    t_wall = time.perf_counter() - t

    t = time.perf_counter()
    clearance = make_clearance_map(wall_img).astype(np.float32)
    t_clear = time.perf_counter() - t

    t = time.perf_counter()
    do_payload = 0
    height_map = np.zeros((1, 1), dtype=np.uint8)   # dummy when disabled
    if height_map_path:
        height_map = load_height_map(height_map_path, max_height=height_max)
        if height_map.shape != wall_img.shape:
            raise ValueError(
                f"height map shape {height_map.shape} differs from wall "
                f"map shape {wall_img.shape}; both must share the grid.")
        do_payload = 1
    t_height = time.perf_counter() - t

    return CCOScene(wall_img=wall_img, clearance=clearance,
                    height_map=height_map, do_payload=do_payload,
                    timings=dict(wall_imread=t_wall, clearance_map=t_clear,
                                 height_load=t_height))


def build_cco_chains(obstacle_map_path, r_infl, step_size):
    """Inflate the obstacle and split its boundary into L/R chains.

    Step-size-dependent preprocessing (redone per step size in the
    benchmarks, untimed).  Returns ``(array_a, array_b, timings)``
    with the chains as float64 ``(N, 2)`` arrays.
    """
    t = time.perf_counter()
    obstacle = Obstacle(str(obstacle_map_path), inflation_radius=r_infl,
                        step_size=step_size)
    t_infl = time.perf_counter() - t

    t = time.perf_counter()
    array_a, array_b = obstacle.split_obstacle()
    array_a = np.asarray(array_a, dtype=np.float64)
    array_b = np.asarray(array_b, dtype=np.float64)
    t_split = time.perf_counter() - t

    return array_a, array_b, dict(obstacle_inflate=t_infl,
                                  split_chains=t_split)


# ═══════════════════════════════════════════════════════════
#  Find path (the timed phase)
# ═══════════════════════════════════════════════════════════

def _prefilter_anchors(array_a, array_b, d_max_keep):
    """(ii, jj) of anchor pairs with ‖r_j − l_i‖ ≤ d_max_keep."""
    aa = np.asarray(array_a, dtype=np.float32)
    bb = np.asarray(array_b, dtype=np.float32)
    dist = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2)
    ii, jj = np.where(dist <= d_max_keep)
    return ii.astype(np.int64), jj.astype(np.int64)


@dataclass
class CCOResult:
    """Everything :func:`find_path_cco` produced besides the path.

    ``t_search`` is the whole find-path phase; ``t_kernel`` and
    ``t_astar`` are sub-spans of it.  Slot-population statistics are
    computed AFTER the timer stops, from the kernel output.
    """
    t_search: float
    t_kernel: float
    t_astar: float
    n_pairs: int                   # len_a * len_b before the prefilter
    n_kept: int                    # anchor pairs surviving the prefilter
    n_valid: int                   # populated A* nodes
    n_left: int                    # pairs with a LEFT slot
    n_right: int                   # pairs with a RIGHT slot
    n_any: int                     # pairs with at least one slot
    start_ij: Tuple[int, int]      # (id_a, id_b) at the near obstacle end
    goal_ij: Tuple[int, int]       # (id_a, id_b) at the far obstacle end
    start_id: int
    goal_id: int


def find_path_cco(array_a, array_b, scene, *,
                  bar, tol_frac, r_infl,
                  robot_r, clearance_margin, intra_robot_dist,
                  L_pole=0.0, L_rope=0.0, cable_sample_step_px=1.0,
                  ) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]],
                             CCOResult]:
    """Run the CCO find-path phase on prepared chains + scene.

    Start and goal are derived automatically at the obstacle's two
    ends: ``find_start`` walks the chains from one end, and again on
    the reversed chains from the other.

    Returns ``(path_AB, result)``: ``path_AB`` is the list of
    ``(m_L, m_R)`` cluster-main pairs (``None`` when the mini A* finds
    no path), ``result`` a :class:`CCOResult`.  ``result.t_search``
    covers prefilter → kernel batch → reshape → start/goal resolution
    → mini A*, and nothing else — the quantity the benchmarks report.

    Raises ``RuntimeError`` (from ``resolve_node_id``) when no
    populated node exists near an obstacle end — e.g. when the wall
    map encloses the obstacle so tightly that the boundary anchors
    have no clearance.
    """
    len_a, len_b = len(array_a), len(array_b)
    min_dist = bar * (1.0 - tol_frac)
    max_dist = bar * (1.0 + tol_frac)
    r_check = float(robot_r) + float(clearance_margin)

    # ── Timed span: everything from the raw chains to the path. ──
    t0 = time.perf_counter()
    ii, jj = _prefilter_anchors(array_a, array_b,
                                max_dist + 2.0 * float(r_infl))

    t = time.perf_counter()
    out = optimise_anchors_batch(
        ii, jj, array_a, array_b,
        float(min_dist), float(max_dist), float(r_infl),
        r_check, float(intra_robot_dist), scene.clearance,
        float(L_pole), float(L_rope), float(cable_sample_step_px),
        scene.height_map, int(scene.do_payload))
    t_kernel = time.perf_counter() - t

    is_valid, mL_xy, mR_xy = batch_to_arrays(
        ii, jj, out, array_a, array_b, len_a, len_b)

    tol_px = tol_frac * bar
    id_a_s, id_b_s = find_start(array_a, array_b, bar, tol_px)
    id_a_e_i, id_b_e_i = find_start(array_a[::-1], array_b[::-1],
                                    bar, tol_px)
    id_a_e = len_a - 1 - id_a_e_i
    id_b_e = len_b - 1 - id_b_e_i
    start_id = resolve_node_id(id_a_s, id_b_s, is_valid, len_b)
    goal_id = resolve_node_id(id_a_e, id_b_e, is_valid, len_b)

    t = time.perf_counter()
    path_ids = astar_small_jit(start_id, goal_id, len_a, len_b,
                               is_valid, mL_xy, mR_xy)
    t_astar = time.perf_counter() - t
    t_search = time.perf_counter() - t0
    # ── End of the timed span. ──

    L_pop = int((out[:, 0] > 0.5).sum())
    R_pop = int((out[:, 3] > 0.5).sum())
    any_pop = int(((out[:, 0] > 0.5) | (out[:, 3] > 0.5)).sum())
    result = CCOResult(
        t_search=t_search, t_kernel=t_kernel, t_astar=t_astar,
        n_pairs=len_a * len_b, n_kept=len(ii),
        n_valid=int(is_valid.sum()),
        n_left=L_pop, n_right=R_pop, n_any=any_pop,
        start_ij=(int(id_a_s), int(id_b_s)),
        goal_ij=(int(id_a_e), int(id_b_e)),
        start_id=int(start_id), goal_id=int(goal_id))

    if len(path_ids) == 0:
        return None, result
    path_AB = [(mL_xy[nid].copy(), mR_xy[nid].copy()) for nid in path_ids]
    return path_AB, result

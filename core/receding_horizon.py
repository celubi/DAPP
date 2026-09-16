"""
Receding-Horizon Formation Planning
====================================

Online (progressively-revealed) formation planning on top of the
``find_path_da`` A* core (DA_astar).  The planner runs in an outer
loop that:

1. Maintains an accumulating known map of observed cells.
2. Treats unknown cells as free when planning (freespace optimism).
3. Plans to the goal on the optimistic map.
4. Commits only the longest path prefix that lies entirely inside
   the already-observed region — guaranteed collision-free under the
   true map.
5. Senses from the new position and repeats.

External dependencies: numpy, scipy (distance_transform_edt), numba.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numba
from scipy.ndimage import distance_transform_edt

from .formations import (
    parse_formations,
    compute_reconfig_costs,
    precompute_offsets,
    compute_h_payload,
    precompute_cable_offsets,
)
from .da_astar import (
    find_path_da,
    _offsets_to_array,
    _precompute_clearance_grid,
    _shortest_reconfig,
    _precompute_arc_angles,
)


State = Tuple[int, int, int, int, int]


# ═══════════════════════════════════════════════════════════
#  Sensing
# ═══════════════════════════════════════════════════════════

def _disc_offset_indices(radius):
    """Precompute (dy, dx) int64 offsets of a filled disc."""
    r = int(math.ceil(radius))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    mask = ys * ys + xs * xs <= radius * radius
    return ys[mask].astype(np.int64), xs[mask].astype(np.int64)


def sense_disc(known_mask, known_occ, true_occ, center_xy, radius,
               disc_offsets=None, known_height=None, true_height=None):
    """In-place reveal of a disc around ``center_xy`` from ``true_occ``.

    Parameters
    ----------
    known_mask : ndarray (H, W) bool — updated in place.
    known_occ  : ndarray (H, W) bool — updated in place.  Must be
        ``False`` on every pixel where ``known_mask`` is ``False``
        (this function preserves that invariant).
    true_occ   : ndarray (H, W) bool — ground-truth occupancy.
    center_xy  : (cx, cy) in pixel coordinates.
    radius     : disc radius in pixels.
    disc_offsets : optional tuple of precomputed ``(dy, dx)`` int
        arrays from ``_disc_offset_indices(radius)``.  Pass it when
        calling repeatedly with the same radius.
    known_height : ndarray (H, W) uint8 or None — updated in place.
        The accumulating height map.  Pass together with
        ``true_height`` to reveal obstacle heights with the same disc
        that reveals occupancy.  ``load_height_map`` encodes ``0`` as
        "passable at any height", so zeros-initialised unseen cells
        are already optimistic — no sentinel needed.
    true_height : ndarray (H, W) uint8 or None — ground-truth height.
    """
    H, W = true_occ.shape
    cx = int(center_xy[0])
    cy = int(center_xy[1])
    if disc_offsets is None:
        dy, dx = _disc_offset_indices(radius)
    else:
        dy, dx = disc_offsets
    ys = cy + dy
    xs = cx + dx
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys = ys[valid]
    xs = xs[valid]
    known_mask[ys, xs] = True
    known_occ[ys, xs] = true_occ[ys, xs]
    if known_height is not None and true_height is not None:
        known_height[ys, xs] = true_height[ys, xs]


def rebuild_maps(known_mask, known_occ):
    """Return ``(dist_map, known_dist)`` for the current known state.

    ``dist_map`` is the EDT under the freespace assumption (unknown
    cells treated as free).  ``known_dist`` is the EDT of
    ``known_mask`` — per-pixel distance to the nearest unknown cell,
    used by ``truncate_to_known`` to check whether a formation state
    lies safely inside the observed region.
    """
    # known_occ is maintained to be False on unknown cells (see
    # sense_disc's invariant), so it already represents the optimistic
    # occupancy map.
    dist_map = distance_transform_edt(~known_occ).astype(np.float64)
    known_dist = distance_transform_edt(known_mask).astype(np.float64)
    return dist_map, known_dist


# ═══════════════════════════════════════════════════════════
#  Horizon truncation
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _is_inside_known(cx, cy, offsets_arr, ic, it, js, known_dist, rb):
    """True when every robot disc centre has ``known_dist >= rb``."""
    H, W = known_dist.shape
    n_robots = offsets_arr.shape[3]
    for k in range(n_robots):
        rx = cx + offsets_arr[ic, it, js, k, 0]
        ry = cy + offsets_arr[ic, it, js, k, 1]
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return False
        if known_dist[ry, rx] < rb:
            return False
    return True


def truncate_to_known(path, known_dist, offsets_arr, rb, xy_step):
    """Longest prefix of ``path`` whose every state is safely inside
    the known region.

    A state is "safely inside" iff every robot disc centre satisfies
    ``known_dist[ry, rx] >= rb`` — the robot is at least ``rb`` pixels
    away from the nearest unknown cell.  Returns a list of states.
    """
    kd = np.ascontiguousarray(known_dist, dtype=np.float64)
    oa = np.ascontiguousarray(offsets_arr, dtype=np.int32)
    rb_f = float(rb)
    step = int(xy_step)
    prefix = []
    for state in path:
        ix, iy, it, js, ic = state
        if _is_inside_known(int(ix) * step, int(iy) * step,
                            oa, int(ic), int(it), int(js), kd, rb_f):
            prefix.append(state)
        else:
            break
    return prefix


# ═══════════════════════════════════════════════════════════
#  Outer loop
# ═══════════════════════════════════════════════════════════

@dataclass
class RecedingResult:
    """Result of a receding-horizon run.

    ``status`` is one of:
      - ``"reached"``       — current state matches a goal state.
      - ``"unreachable"``   — optimistic A* returned no path; no
                              future observation can change this.
      - ``"stuck"``         — path exists but the first step leaves
                              the known-free region, and a retry did
                              not help.
      - ``"max_iterations"``— ran out of outer-loop iterations.
    """
    trajectory: List[State]
    costs: List[float]
    expansions: List[int]
    known_mask_final: np.ndarray
    known_occ_final: np.ndarray
    status: str
    known_height_final: np.ndarray = None
    # One entry per planning call, aligned with ``expansions``:
    # ``plan_times[k]`` is how long iteration k's A* took (seconds) and
    # ``commit_ends[k]`` is the index in ``trajectory`` its committed
    # prefix ends at (unchanged from the previous entry when the
    # iteration committed nothing).
    plan_times: List[float] = None
    commit_ends: List[int] = None


def plan_receding_horizon(
    true_occ, start, goal, *,
    sensor_radius,
    rb=8.0, rf=40.0,
    formations_deg,
    xy_step=10, n_theta=72,
    s_min=0.6, s_max=1.4, n_s=20,
    w_move=1.0, w_rot=1.0, w_scale=1.0, w_config=1.0,
    c_deform=None,
    reconfig_check="none", n_arc_samples=8,
    use_symmetry=True,
    free_theta=False, free_s=False, free_config=False,
    true_height=None, L_pole=None, L_rope=None,
    cable_sample_step_px=1.5,
    cost_limit=None,
    max_iterations=200,
    verbose=True,
):
    """Run the full receding-horizon loop against a ground-truth map.

    Parameters mirror ``find_path_da_from_map`` plus:

    true_occ : ndarray (H, W) bool — ground-truth occupancy.
    sensor_radius : float — radius (pixels) of the disc revealed on
        each sensor update.  Must satisfy ``sensor_radius >= 2·rf·s_max``
        (otherwise the formation at max scale may not fit inside the
        sensed disc and horizon truncation can fail at the start).
    true_height : ndarray (H, W) uint8 or None — ground-truth height
        map (from ``map_io.load_height_map``), same grid as
        ``true_occ``.  When given, the payload / cable height check is
        enabled and heights are revealed progressively by the same
        sensor disc that reveals occupancy (unseen cells stay ``0`` =
        passable, i.e. optimistic).  Requires ``L_pole`` and
        ``L_rope``.  *None* → height check disabled (default).
    L_pole, L_rope : float — pole height and rope length, in the same
        units as the height-map values.  Required with ``true_height``.
    cable_sample_step_px : float — spacing of the cable samples used
        by the height check.
    max_iterations : int — upper bound on outer-loop iterations.

    Returns
    -------
    RecedingResult
    """
    true_occ = np.asarray(true_occ, dtype=bool)
    H, W = true_occ.shape

    if true_height is not None:
        if L_pole is None or L_rope is None:
            raise ValueError(
                "true_height requires both L_pole and L_rope to be "
                "specified (pole height and rope length in the same "
                "units as the height map values).")
        true_height = np.ascontiguousarray(true_height, dtype=np.uint8)
        if true_height.shape != true_occ.shape:
            raise ValueError(
                f"true_height shape {true_height.shape} differs from "
                f"true_occ shape {true_occ.shape}; both must be the "
                f"same image grid.")

    safe_radius = 2.0 * rf * s_max
    if sensor_radius < safe_radius:
        raise ValueError(
            f"sensor_radius={sensor_radius:.1f} is below the safe "
            f"threshold 2·rf·s_max={safe_radius:.1f}. At max scale the "
            f"formation may not fit inside the sensed disc."
        )

    # ── Build offsets / reconfig structures ──────────────────────────
    s_values = np.linspace(s_min, s_max, n_s)
    formations_rad, clusters, cfg_sym_orders = parse_formations(formations_deg)
    n_config = len(formations_rad)
    n_robots_actual = len(formations_rad[0])

    sym_orders = cfg_sym_orders if use_symmetry else [1] * n_config
    periods = [n_theta // k for k in sym_orders]
    for ic, k in enumerate(sym_orders):
        if n_theta % k != 0:
            raise ValueError(
                f"n_theta={n_theta} not divisible by symmetry order "
                f"{k} of config {ic}")
    periods_arr = np.array(periods, dtype=np.int64)

    off_dict = precompute_offsets(formations_rad, rf, n_theta,
                                  s_values, periods, clusters)
    offsets_arr = _offsets_to_array(off_dict, n_config, n_theta,
                                    n_s, n_robots_actual)

    arc_angles = None
    if n_config > 1:
        rc_costs, rc_assignments = compute_reconfig_costs(formations_rad)
        rc_min = _shortest_reconfig(rc_costs)
        if reconfig_check == "sampling":
            arc_angles = _precompute_arc_angles(
                formations_rad, rc_assignments, n_arc_samples)
    else:
        rc_costs = np.zeros((1, 1))
        rc_min = np.zeros((1, 1))

    # ── Payload / cable height structures (optional) ────────────────
    h_payload = None
    cable_offsets = None
    cable_counts = None
    js_admissible = None
    if true_height is not None:
        h_payload, js_admissible = compute_h_payload(
            s_values, rf, L_pole, L_rope)
        cable_offsets, cable_counts = precompute_cable_offsets(
            formations_rad, rf, n_theta, s_values, periods,
            clusters, sample_step_px=cable_sample_step_px)
        if verbose:
            n_adm = int(js_admissible.sum())
            print(f"  Payload check: ON  | L_pole={L_pole}, "
                  f"L_rope={L_rope} | js admissible {n_adm}/{n_s}")

    # ── Initial map state ────────────────────────────────────────────
    known_mask = np.zeros((H, W), dtype=bool)
    known_occ = np.zeros((H, W), dtype=bool)
    # Unseen height stays 0 = passable (optimistic, see sense_disc).
    known_height = (np.zeros((H, W), dtype=np.uint8)
                    if true_height is not None else None)
    disc_offsets = _disc_offset_indices(sensor_radius)

    start_t: State = tuple(int(x) for x in start)
    if len(start_t) == 4:
        start_t = start_t + (0,)
    goal_t = tuple(int(x) for x in goal)
    if len(goal_t) == 4:
        goal_t = goal_t + (0,)

    def pixel_center(state):
        return state[0] * xy_step, state[1] * xy_step

    sense_disc(known_mask, known_occ, true_occ,
               pixel_center(start_t), sensor_radius, disc_offsets,
               known_height=known_height, true_height=true_height)

    trajectory: List[State] = [start_t]
    costs: List[float] = []
    expansions: List[int] = []
    plan_times: List[float] = []
    commit_ends: List[int] = []
    current = start_t
    status = "max_iterations"
    stuck_retries = 0

    for it_num in range(max_iterations):
        dist_map, known_dist = rebuild_maps(known_mask, known_occ)
        clr_grid = (_precompute_clearance_grid(dist_map, xy_step)
                    if c_deform is not None else None)

        t_iter = time.perf_counter()
        path, cost, n_exp = find_path_da(
            dist_map, offsets_arr, s_values, periods_arr,
            formations_rad, rc_costs, rc_min,
            rb, xy_step, n_theta, n_s,
            w_move, w_rot, w_scale, w_config,
            current, goal_t,
            free_theta=free_theta, free_s=free_s, free_config=free_config,
            clearance_grid=clr_grid, c_deform=c_deform,
            reconfig_check=reconfig_check, rf=rf, arc_angles=arc_angles,
            cost_limit=cost_limit,
            # known_height, NOT true_height: the planner may only see
            # what the sensor has actually measured.
            height_map=known_height, h_payload=h_payload,
            cable_offsets=cable_offsets, cable_counts=cable_counts,
            js_admissible=js_admissible,
            verbose=verbose,
        )
        plan_times.append(time.perf_counter() - t_iter)
        expansions.append(n_exp)
        # Provisional: overwritten below when this iteration commits.
        commit_ends.append(len(trajectory) - 1)

        if path is None:
            status = "unreachable"
            if verbose:
                print(f"  Iter {it_num}: optimistic planner found no path. "
                      f"Terminating.")
            break

        costs.append(cost)

        if len(path) == 1:
            status = "reached"
            if verbose:
                print(f"  Iter {it_num}: goal reached at {current}.")
            break

        prefix = truncate_to_known(path, known_dist, offsets_arr, rb, xy_step)

        if len(prefix) <= 1:
            stuck_retries += 1
            if verbose:
                print(f"  Iter {it_num}: stuck at frontier "
                      f"(retry {stuck_retries}).")
            if stuck_retries >= 2:
                status = "stuck"
                break
            # Best-effort re-sense (a no-op on a static map).
            sense_disc(known_mask, known_occ, true_occ,
                       pixel_center(current), sensor_radius, disc_offsets,
                       known_height=known_height, true_height=true_height)
            continue
        stuck_retries = 0

        trajectory.extend(prefix[1:])
        commit_ends[-1] = len(trajectory) - 1
        current = prefix[-1]
        sense_disc(known_mask, known_occ, true_occ,
                   pixel_center(current), sensor_radius, disc_offsets,
                   known_height=known_height, true_height=true_height)

        if verbose:
            print(f"  Iter {it_num}: plan cost {cost:.2f}, committed "
                  f"{len(prefix)}/{len(path)} states, n_exp {n_exp:,}.")

    return RecedingResult(
        trajectory=trajectory,
        costs=costs,
        expansions=expansions,
        known_mask_final=known_mask,
        known_occ_final=known_occ,
        status=status,
        known_height_final=known_height,
        plan_times=plan_times,
        commit_ends=commit_ends,
    )

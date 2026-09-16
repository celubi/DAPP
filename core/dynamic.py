"""Dynamic A* — time-aware re-planning around moving obstacles.

Takes a *spatial* 5-tuple path ``(ix, iy, iθ, is, ic)`` and re-plans
the *timing* plus bounded θ/scale offsets within a windowed sub-path
that intersects a set of moving obstacles.  The cluster config ``ic``
of each waypoint is held fixed during the dynamic search.

Main entry point: :func:`dynamic_astar_windowed`.
"""

import math
import time as time_module

import numpy as np

from .dynamic_jit import dynamic_astar_jit as _dynamic_astar_jit


# ═══════════════════════════════════════════════════════════
#  Waypoint extraction
# ═══════════════════════════════════════════════════════════

def extract_unique_waypoints(path, xy_step, deduplicate=True):
    """Extract ``(x, y)`` waypoints from a 5-tuple A* path.

    Each entry of ``path`` is ``(ix, iy, iθ, is, ic)``.

    When ``deduplicate=True`` (default), consecutive states sharing the
    same pixel (in-place rotation / scale / reconfiguration) are
    collapsed into a single waypoint carrying the *latest* config.
    With ``deduplicate=False`` every state becomes its own waypoint —
    the dynamic planner needs this so one waypoint = one action and
    in-place rotations/scales keep their time cost in the baseline.

    Returns
    -------
    waypoints_px : ndarray (N_wp, 2) float64
    wp_configs   : list of (iθ, is, ic)
    wp_map       : list[int]  — wp_map[orig_step] = waypoint index
    """
    pts = [(s[0] * xy_step, s[1] * xy_step) for s in path]
    configs = [(int(s[2]), int(s[3]), int(s[4])) for s in path]

    if not deduplicate:
        waypoints = list(pts)
        wp_configs_out = list(configs)
        wp_map = list(range(len(pts)))
        return (np.array(waypoints, dtype=np.float64),
                wp_configs_out, wp_map)

    waypoints = [pts[0]]
    wp_configs = [configs[0]]
    wp_map = [0]
    for i in range(1, len(pts)):
        if pts[i] != waypoints[-1]:
            waypoints.append(pts[i])
            wp_configs.append(configs[i])
        else:
            wp_configs[-1] = configs[i]
        wp_map.append(len(waypoints) - 1)
    return np.array(waypoints, dtype=np.float64), wp_configs, wp_map


# ═══════════════════════════════════════════════════════════
#  Static collision LUT along waypoints
# ═══════════════════════════════════════════════════════════

def compute_path_static_free(waypoints_px, wp_configs, offsets_arr,
                             rb, dist_map, n_theta, n_s, periods,
                             height_map=None, h_payload=None,
                             cable_offsets=None, cable_counts=None,
                             js_admissible=None):
    """Per-waypoint static-collision lookup table.

    For each waypoint ``p`` (with fixed cluster config ``ic_p``), build
    ``static_free[p, iθ, is]`` — true when every robot of the formation
    at ``(ic_p, iθ, is)`` is at least ``rb`` from any static obstacle.

    The θ axis is folded through symmetry: ``it >= periods[ic]`` is
    looked up as ``it % periods[ic]``.

    When the optional ``height_map`` / ``h_payload`` / ``cable_offsets``
    triple is supplied, samples along the robot-to-centre cables are
    additionally checked against the height map, and scales flagged as
    inadmissible in ``js_admissible`` are forced to ``False``.

    Parameters
    ----------
    waypoints_px : (N, 2) ndarray
    wp_configs   : list of (iθ, is, ic)
    offsets_arr  : (n_config, n_theta, n_s, n_robots, 2) int — fapp offsets
                   (only entries with ``it < periods[ic]`` need be filled)
    rb           : float — robot body radius (px)
    dist_map     : (H, W) float — Euclidean distance to nearest obstacle
    n_theta, n_s : int
    periods      : sequence of int, length ``n_config``
    height_map   : (H, W) uint8, optional — obstacle height per pixel
    h_payload    : (n_s,) float64, optional — payload height per scale
    cable_offsets : (n_config, n_theta, n_s, K_max, 2) int32, optional
    cable_counts : (n_s,) int32, optional — valid sample count per scale
    js_admissible : (n_s,) bool, optional — physical feasibility mask

    Returns
    -------
    static_free : (N, n_theta, n_s) bool
    """
    N = len(waypoints_px)
    H, W = dist_map.shape
    free = np.zeros((N, n_theta, n_s), dtype=bool)
    do_payload = height_map is not None

    for p in range(N):
        ic = int(wp_configs[p][2])
        per = int(periods[ic])
        cx = int(round(waypoints_px[p, 0]))
        cy = int(round(waypoints_px[p, 1]))
        for it in range(n_theta):
            it_eff = it % per
            for js in range(n_s):
                if do_payload and not js_admissible[js]:
                    continue
                off = offsets_arr[ic, it_eff, js]
                ok = True
                for k in range(off.shape[0]):
                    rx = cx + int(off[k, 0])
                    ry = cy + int(off[k, 1])
                    if rx < 0 or rx >= W or ry < 0 or ry >= H:
                        ok = False
                        break
                    if dist_map[ry, rx] < rb:
                        ok = False
                        break
                if ok and do_payload:
                    h_thresh = h_payload[js]
                    cab = cable_offsets[ic, it_eff, js]
                    K = int(cable_counts[js])
                    for k in range(K):
                        rx = cx + int(cab[k, 0])
                        ry = cy + int(cab[k, 1])
                        if rx < 0 or rx >= W or ry < 0 or ry >= H:
                            ok = False
                            break
                        if height_map[ry, rx] > h_thresh:
                            ok = False
                            break
                free[p, it, js] = ok
    return free


# ═══════════════════════════════════════════════════════════
#  Cumulative distance & obstacle table
# ═══════════════════════════════════════════════════════════

def compute_wp_dt_offsets(waypoints_px, wp_configs,
                          dt_move=1, dt_rot=1, dt_scale=1):
    """Cumulative dt offset of each waypoint under the spatial plan.

    ``wp_dt_offsets[k]`` is the number of planner time steps elapsed
    between waypoint 0 and waypoint k: each transition
    ``wp[k-1] → wp[k]`` is classified as a move / rotation / scale and
    charged the matching dt.  Lets :func:`find_collision_window`, the
    JIT solver and :func:`stitch_plan` agree on when each waypoint is
    reached when the per-axis dt's are not all equal.

    Parameters
    ----------
    waypoints_px : (N, 2) ndarray
    wp_configs   : list of (iθ, is, ic)
    dt_move, dt_rot, dt_scale : int
        Time-step cost of each primitive (defaults match the demo
        with all-equal dt's = 1).

    Returns
    -------
    wp_dt_offsets : (N,) int ndarray
    """
    N = len(waypoints_px)
    out = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        same_px = (waypoints_px[k, 0] == waypoints_px[k - 1, 0]
                   and waypoints_px[k, 1] == waypoints_px[k - 1, 1])
        if same_px:
            it_prev = int(wp_configs[k - 1][0])
            it_curr = int(wp_configs[k][0])
            js_prev = int(wp_configs[k - 1][1])
            js_curr = int(wp_configs[k][1])
            if it_prev != it_curr:
                dt = dt_rot
            elif js_prev != js_curr:
                dt = dt_scale
            else:
                # Same pose and same pixel — degenerate; charge a
                # move dt as a neutral default.
                dt = dt_move
        else:
            dt = dt_move
        out[k] = out[k - 1] + dt
    return out


def compute_cumulative_dist(waypoints_px):
    """Cumulative Euclidean distance from waypoint 0 to each waypoint."""
    N = len(waypoints_px)
    cum = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        cum[i] = cum[i - 1] + np.linalg.norm(
            waypoints_px[i] - waypoints_px[i - 1])
    return cum


def compute_wp_cost_offsets(waypoints_px, wp_configs, rf, s_values,
                            n_theta, w_path=1.0, w_rot=1.0, w_scale=1.0):
    """Cumulative *spatial cost* of the baseline up to each waypoint.

    The spatial twin of :func:`compute_wp_dt_offsets`: that one
    accumulates the duration of each baseline transition, this one the
    weighted displacement it produces — a translation's Euclidean
    length, the arc ``rf · s · Δθ`` of an in-place rotation, or the
    radial travel ``rf · |Δs|`` of an in-place scaling.

    Rotations use the *nominal* scale of the waypoint: the deviation
    ``δs`` is unknown offline, and a fixed radius keeps the baseline
    cost a constant of the window — identical for every candidate plan,
    which is what makes charging it harmless.

    Feeding this to the solver makes the dynamic cost model the exact
    restriction of the global planner's cost to the baseline corridor,
    and turns the residual path term of the heuristic into the exact
    remaining cost rather than a lower bound.

    Returns
    -------
    wp_cost_offsets : (N,) float64 ndarray — weights already applied.
    """
    N = len(waypoints_px)
    out = np.zeros(N, dtype=np.float64)
    if N == 0:
        return out
    th_step = 2.0 * math.pi / n_theta
    s_arr = np.asarray(s_values, dtype=np.float64)
    s_step = ((s_arr[-1] - s_arr[0]) / max(len(s_arr) - 1, 1)
              if len(s_arr) > 1 else 0.0)
    half = n_theta // 2
    for k in range(1, N):
        dx = float(waypoints_px[k, 0] - waypoints_px[k - 1, 0])
        dy = float(waypoints_px[k, 1] - waypoints_px[k - 1, 1])
        if dx != 0.0 or dy != 0.0:
            c = math.hypot(dx, dy) * w_path
        else:
            it_prev = int(wp_configs[k - 1][0])
            it_curr = int(wp_configs[k][0])
            js_prev = int(wp_configs[k - 1][1])
            js_curr = int(wp_configs[k][1])
            if it_prev != it_curr:
                d_it = abs(((it_curr - it_prev + half) % n_theta) - half)
                c = d_it * th_step * rf * s_arr[js_curr] * w_rot
            elif js_prev != js_curr:
                c = abs(js_curr - js_prev) * s_step * rf * w_scale
            else:
                c = 0.0
        out[k] = out[k - 1] + c
    return out


def prepare_obstacle_table(obstacles, t_max, dt):
    """Resample obstacle trajectories to discrete planner time steps.

    Parameters
    ----------
    obstacles : list of dict
        Each ``{'positions': (M, 2), 'times': (M,), 'radius': float}``.
    t_max : int — number of planner time steps.
    dt    : float — seconds per time step.

    Returns
    -------
    obs_table : (n_obs, t_max, 2) float64 — NaN where the obstacle is
                                            outside its active window.
    obs_radii : (n_obs,) float64
    """
    if not obstacles:
        return (np.empty((0, t_max, 2), dtype=np.float64),
                np.empty(0, dtype=np.float64))

    n_obs = len(obstacles)
    table = np.full((n_obs, t_max, 2), np.nan, dtype=np.float64)
    radii = np.zeros(n_obs, dtype=np.float64)
    planner_times = np.arange(t_max) * dt

    for i, obs in enumerate(obstacles):
        radii[i] = obs['radius']
        obs_t = obs['times']
        for dim in range(2):
            table[i, :, dim] = np.interp(
                planner_times, obs_t, obs['positions'][:, dim],
                left=np.nan, right=np.nan)
    return table, radii


# ═══════════════════════════════════════════════════════════
#  Offsets array helper
# ═══════════════════════════════════════════════════════════

def offsets_to_array(offsets, n_config, n_theta, n_s, n_robots, periods):
    """Normalise offsets into a dense float64 array for the JIT solver.

    ``precompute_offsets`` only emits entries for ``it < periods[ic]``;
    this helper folds the θ axis through symmetry so that every one of
    the ``n_theta`` slots is populated with the correct offsets.

    Parameters
    ----------
    offsets : dict or ndarray
        ``{(ic, iθ, is): (n_robots, 2)}`` (the ``precompute_offsets``
        output) or a precomputed 5-D array.  Arrays are passed through
        unchanged — the caller is responsible for having folded them.
    periods : sequence of int, length ``n_config``
        Rotation period (= ``n_theta // sym_order``) of each cluster
        config.  Used for symmetry folding.

    Returns ``(n_config, n_theta, n_s, n_robots, 2)`` float64.
    """
    if isinstance(offsets, np.ndarray):
        return np.ascontiguousarray(offsets, dtype=np.float64)
    arr = np.zeros((n_config, n_theta, n_s, n_robots, 2),
                   dtype=np.float64)
    for (ic, it, js), off in offsets.items():
        arr[ic, it, js] = off
    # Fold θ through symmetry: arr[ic, it, js] := arr[ic, it % periods[ic], js]
    # for every it in [periods[ic], n_theta).
    for ic in range(n_config):
        per = int(periods[ic])
        if per >= n_theta:
            continue
        for it in range(per, n_theta):
            arr[ic, it] = arr[ic, it % per]
    return arr


# ═══════════════════════════════════════════════════════════
#  Dynamic collision & window finder
# ═══════════════════════════════════════════════════════════

def dynamic_collision(cx, cy, offs, rb, obs_table, obs_radii, t):
    """True if ANY robot disc overlaps ANY dynamic obstacle at time ``t``.

    ``offs`` is the ``(n_robots, 2)`` offset table for the formation
    pose ``(ic, iθ, is)`` at the queried waypoint.
    """
    n_obs = len(obs_radii)
    if n_obs == 0 or t < 0 or t >= obs_table.shape[1]:
        return False
    for i in range(n_obs):
        ox = obs_table[i, t, 0]
        oy = obs_table[i, t, 1]
        if math.isnan(ox):
            continue
        thr = rb + obs_radii[i]
        thr2 = thr * thr
        for k in range(offs.shape[0]):
            dx = cx + offs[k, 0] - ox
            dy = cy + offs[k, 1] - oy
            if dx * dx + dy * dy < thr2:
                return True
    return False


def find_collision_window(waypoints_px, wp_configs, offsets_arr, rb,
                          obs_table, obs_radii, margin=10,
                          margin_before=None, margin_after=None,
                          wp_dt_offsets=None,
                          verbose=True):
    """Detect the naïve-pass collision window.

    The "naïve" pass moves the formation through the spatial plan
    one waypoint after the other and queries the obstacle table at
    each waypoint's *physical* time.  When ``wp_dt_offsets`` is
    supplied, waypoint k is sampled at planner step
    ``wp_dt_offsets[k]``; otherwise the historical "1 wp = 1 dt"
    convention is used (waypoint k sampled at step k).

    Parameters
    ----------
    margin : int
        Symmetric padding around the detected collision range,
        applied when ``margin_before`` / ``margin_after`` are not
        supplied.
    margin_before, margin_after : int, optional
        Asymmetric padding (waypoints added before / after the
        detected collision range).  When either is None it falls back
        to ``margin``.
    wp_dt_offsets : (N,) array_like, optional
        Per-waypoint time offset (in planner steps) under the spatial
        plan.  Required for coherent collision detection when the
        per-axis dt's are not all 1.

    Returns
    -------
    (p_start, p_end) : tuple[int, int]  — inclusive bounds, or
    None — when no collision is found.
    """
    if margin_before is None:
        margin_before = margin
    if margin_after is None:
        margin_after = margin
    n_wp = len(waypoints_px)
    n_obs = len(obs_radii)
    t_max_obs = obs_table.shape[1] if n_obs > 0 else 0

    col_steps = []
    min_dist = float('inf')
    min_dist_t = -1
    # p indexes the waypoint; t is the planner step it is reached at.
    for p in range(n_wp):
        if wp_dt_offsets is None:
            t = p
        else:
            t = int(wp_dt_offsets[p])
        if t >= t_max_obs:
            break
        it, js, ic = wp_configs[p]
        cx, cy = waypoints_px[p]
        offs = offsets_arr[ic, it, js]
        if dynamic_collision(cx, cy, offs, rb, obs_table, obs_radii, t):
            col_steps.append(p)
        for oi in range(n_obs):
            ox, oy = obs_table[oi, t, 0], obs_table[oi, t, 1]
            if math.isnan(ox):
                continue
            for k in range(offs.shape[0]):
                d = math.hypot(cx + offs[k, 0] - ox,
                               cy + offs[k, 1] - oy)
                if d < min_dist:
                    min_dist = d
                    min_dist_t = t

    if not col_steps:
        if verbose and min_dist_t >= 0:
            thr = rb + (obs_radii[0] if n_obs > 0 else 0.0)
            print(f"  No naive-path collision detected.")
            print(f"  Closest approach: {min_dist:.1f} px at t={min_dist_t}"
                  f"  (threshold = {thr:.1f} = rb + obs_r)")
        return None

    p_start = max(0, min(col_steps) - margin_before)
    p_end = min(n_wp - 1, max(col_steps) + margin_after)
    if verbose:
        print(f"  Collision detected at {len(col_steps)} steps "
              f"(t={min(col_steps)}-{max(col_steps)})")
    return p_start, p_end


# ═══════════════════════════════════════════════════════════
#  Plan stitching
# ═══════════════════════════════════════════════════════════

def stitch_plan(wp_theta, wp_scale, wp_config, n_wp,
                p_start, p_end, window_plan,
                wp_dt_offsets=None):
    """Concatenate naïve pre-window, the dynamic window plan, and naïve
    post-window into a single full-length plan.

    Each entry of the returned list is ``(p, iθ, is, ic, t)``.

    When ``wp_dt_offsets`` is supplied, the pre-window and post-window
    segments are timed against the spatial-plan dt offsets — every
    waypoint k receives ``t = wp_dt_offsets[k]`` shifted so that
    ``p_end`` aligns with the dynamic plan's last ``t``.  Without it,
    the "1 wp = 1 dt" convention is used.
    """
    full = []

    if wp_dt_offsets is None:
        for t in range(p_start):
            full.append((t, int(wp_theta[t]), int(wp_scale[t]),
                         int(wp_config[t]), t))
    else:
        for p in range(p_start):
            t = int(wp_dt_offsets[p])
            full.append((p, int(wp_theta[p]), int(wp_scale[p]),
                         int(wp_config[p]), t))

    for entry in window_plan:
        full.append(entry)

    if window_plan:
        t_last = window_plan[-1][4]
    else:
        if wp_dt_offsets is None:
            t_last = p_end
        else:
            t_last = int(wp_dt_offsets[p_end])

    if wp_dt_offsets is None:
        for i in range(1, n_wp - p_end):
            p = p_end + i
            t = t_last + i
            full.append((p, int(wp_theta[p]), int(wp_scale[p]),
                         int(wp_config[p]), t))
    else:
        base = int(wp_dt_offsets[p_end])
        for i in range(1, n_wp - p_end):
            p = p_end + i
            t = t_last + (int(wp_dt_offsets[p]) - base)
            full.append((p, int(wp_theta[p]), int(wp_scale[p]),
                         int(wp_config[p]), t))

    return full


def densify_plan(plan):
    """Fill temporal gaps so that each consecutive entry differs by
    exactly one time step.

    Gap frames must hold the **destination** state, not the source:
    the JIT collision check of an action verifies the destination pose
    at every intermediate global time (the formation arrives early and
    waits), so that is the only pose the search has checked.
    """
    if not plan:
        return plan
    dense = []
    for entry in plan:
        p, ith, isc, ic, t = entry
        if dense:
            prev_t = dense[-1][4]
            for fill_t in range(prev_t + 1, t):
                dense.append((p, ith, isc, ic, fill_t))
        dense.append((p, ith, isc, ic, t))
    return dense



# ═══════════════════════════════════════════════════════════
#  JIT warmup (one-time compilation cost)
# ═══════════════════════════════════════════════════════════

_jit_warmed_up = False


def _warmup_jit(waypoints_px, wp_theta, wp_scale, wp_config,
                cum_base_cost, cum_base_dt, static_free, offsets_flat,
                obs_table, obs_radii, rb,
                n_theta, n_s, th_step, s_step,
                obs_heights, cable_offsets, cable_counts, h_payload,
                decoupled_actions, dt_move, dt_rot, dt_scale, dt_wait,
                rf, s_values):
    global _jit_warmed_up
    if _jit_warmed_up:
        return
    t0 = time_module.time()
    _dynamic_astar_jit(
        waypoints_px[:2], wp_theta[:2], wp_scale[:2], wp_config[:2],
        cum_base_cost[:2], cum_base_dt[:2], static_free[:2], offsets_flat,
        obs_table, obs_radii, rb,
        n_theta, n_s, th_step, s_step,
        0, 1, 0, 0,
        1.0, 1.0, 1.0, 1.0,
        False, 2, 10,
        1.0, 0, 0,
        obs_heights, cable_offsets, cable_counts, h_payload,
        decoupled_actions, dt_move, dt_rot, dt_scale, dt_wait,
        0,
        rf, s_values,
    )
    dt = time_module.time() - t0
    _jit_warmed_up = True
    if dt > 0.05:
        print(f"  JIT warmup: {dt:.2f} s (one-time compilation)")


# ═══════════════════════════════════════════════════════════
#  Unified wrapper — dynamic_astar_windowed
# ═══════════════════════════════════════════════════════════

def dynamic_astar_windowed(
    waypoints_px, wp_configs, offsets, cumulative_dist,
    static_free, obs_table, obs_radii,
    rb, n_config, n_theta, n_s, s_values, periods,
    rf=40.0,
    w_path=1.0, w_rot=2.0, w_scale=2.0, w_time=0.5,
    max_dtheta_deg=90.0, max_ds_steps=2,
    window_margin=15, auto_window=True,
    window_margin_before=None, window_margin_after=None,
    allow_backward=True,
    time_budget=20.0, time_step=0.5,
    time_budget_factor=None, time_budget_extra=0.0,
    max_exp=2_000_000,
    wa_epsilon=1.0,
    dt_backward_extra=0,
    dt_turn_extra=0,
    obs_heights=None,
    cable_offsets=None, cable_counts=None,
    h_payload=None,
    decoupled_actions=False,
    dt_move=1, dt_rot=1, dt_scale=1, dt_wait=1,
    wp_dt_offsets=None,
    verbose=True,
):
    """Dynamic A* with auto-windowed collision detection and bounded
    θ/scale offsets relative to a fixed spatial baseline.

    Parameters
    ----------
    waypoints_px : (N, 2) ndarray
        Unique waypoints from :func:`extract_unique_waypoints`.
    wp_configs : list of (iθ, is, ic)
        Spatial-plan configuration at each waypoint.  ``ic`` is held
        fixed during the dynamic search.
    offsets : dict or ndarray
        Either ``{(ic, iθ, is): (n_robots, 2) int}`` (the fapp
        ``precompute_offsets`` output) or a 5-D array
        ``(n_config, n_theta, n_s, n_robots, 2)``.
    cumulative_dist : (N,) ndarray — from :func:`compute_cumulative_dist`.
        Retained for API compatibility; the search uses the cumulative
        *cost* of the baseline, built internally by
        :func:`compute_wp_cost_offsets`, which also charges the robot
        travel of in-place rotations and scalings.
    static_free : (N, n_theta, n_s) bool — from
        :func:`compute_path_static_free`.
    obs_table : (n_obs, t_max, 2) — from :func:`prepare_obstacle_table`.
    obs_radii : (n_obs,)
    rb : float — robot body radius.
    n_config, n_theta, n_s : int
    s_values : (n_s,) ndarray
    periods : sequence of int, length ``n_config``
        Rotation period (= ``n_theta // sym_order``) per cluster config;
        used to fold the θ axis of ``offsets`` through symmetry so every
        ``n_theta`` slot is correctly populated.
    w_path, w_rot, w_scale, w_time : float — A* edge cost weights.
    max_dtheta_deg : float — bound on θ-offset magnitude.
    max_ds_steps : int   — bound on scale-offset magnitude.
    window_margin : int  — symmetric padding around the detected
        collision range (used when the asymmetric overrides below are
        None).
    window_margin_before, window_margin_after : int, optional
        Asymmetric padding (waypoints added before / after the
        detected collision range).  Either falls back to
        ``window_margin`` when None.
    auto_window : bool   — when False the entire path is searched.
    allow_backward : bool — allow ``dp = -1`` moves.
    time_budget : float
        Extra seconds beyond naïve traversal.  Used as an absolute
        budget when ``time_budget_factor`` is None.
    time_budget_factor : float, optional
        When set, the horizon is ``naive_secs * factor +
        time_budget_extra``, where ``naive_secs`` is the naïve
        traversal time of the spatial window — the temporal budget
        then scales with the window size.
    time_budget_extra : float
        Absolute seconds added on top of the proportional budget.
        Ignored when ``time_budget_factor`` is None.
    time_step : float    — seconds per planner time step.
    max_exp : int        — A* expansion limit.
    wa_epsilon : float   — Weighted-A* heuristic inflation (≥ 1).
    dt_backward_extra, dt_turn_extra : int
        Extra time steps charged for backward / turning actions in
        ``coupled`` mode (variable-duration compound moves).
    decoupled_actions : bool
        If ``True`` the search uses one-axis-at-a-time actions (2 move
        + 2 rotate + 2 scale, branching ≤ 7) with per-axis dt's
        ``dt_move`` / ``dt_rot`` / ``dt_scale``; if ``False``, coupled
        compound moves (branching ≤ 26) with unit dt plus
        ``dt_turn_extra`` / ``dt_backward_extra``.
    dt_move, dt_rot, dt_scale : int
        Time-step cost of a single decoupled move / rotate / scale
        action.  Ignored when ``decoupled_actions=False``.
    wp_dt_offsets : (N,) array_like of int, optional
        Cumulative dt offset of each waypoint under the spatial plan
        (output of :func:`compute_wp_dt_offsets`).  Required when the
        per-axis dt's are not all 1; when None, the "1 wp = 1 dt"
        convention is used.

    Returns
    -------
    plan : list of (p, iθ, is, ic, t) or None
    n_expanded : int
    window : (p_start, p_end) or None
    """
    n_wp = len(waypoints_px)
    if n_wp == 0:
        return [], 0, None

    wp_theta = np.array([c[0] for c in wp_configs], dtype=np.int32)
    wp_scale = np.array([c[1] for c in wp_configs], dtype=np.int32)
    wp_config = np.array([c[2] for c in wp_configs], dtype=np.int32)

    n_robots = None
    if isinstance(offsets, np.ndarray):
        n_robots = offsets.shape[3]
    else:
        for v in offsets.values():
            n_robots = v.shape[0]
            break
    if n_robots is None:
        raise ValueError("offsets is empty — cannot infer n_robots")

    offsets_flat = offsets_to_array(
        offsets, n_config, n_theta, n_s, n_robots, periods)

    # ── Payload / cable check on dynamic obstacles ────────
    # obs_heights=None disables the check via a -inf sentinel height;
    # placeholder arrays keep the JIT signature uniform.
    n_obs = obs_radii.shape[0]
    if obs_heights is None:
        obs_heights_c = np.full(max(n_obs, 1), -np.inf, dtype=np.float64)
    else:
        obs_heights_c = np.ascontiguousarray(obs_heights, dtype=np.float64)
        if obs_heights_c.shape != (n_obs,):
            raise ValueError(
                f"obs_heights must have length {n_obs}, got "
                f"{obs_heights_c.shape}")
    if cable_offsets is None:
        cable_offsets_c = np.zeros(
            (n_config, n_theta, n_s, 1, 2), dtype=np.int32)
        cable_counts_c = np.zeros(n_s, dtype=np.int32)
    else:
        cable_offsets_c = np.ascontiguousarray(cable_offsets, dtype=np.int32)
        cable_counts_c = np.ascontiguousarray(cable_counts, dtype=np.int32)
    if h_payload is None:
        h_payload_c = np.full(n_s, np.inf, dtype=np.float64)
    else:
        h_payload_c = np.ascontiguousarray(h_payload, dtype=np.float64)

    # ── window detection ──────────────────────────────────
    if auto_window:
        window = find_collision_window(
            waypoints_px, wp_configs, offsets_flat, rb,
            obs_table, obs_radii, margin=window_margin,
            margin_before=window_margin_before,
            margin_after=window_margin_after,
            wp_dt_offsets=wp_dt_offsets,
            verbose=verbose)
        if window is None:
            if wp_dt_offsets is None:
                plan = [(t, int(wp_theta[t]), int(wp_scale[t]),
                         int(wp_config[t]), t)
                        for t in range(n_wp)]
            else:
                plan = [(p, int(wp_theta[p]), int(wp_scale[p]),
                         int(wp_config[p]), int(wp_dt_offsets[p]))
                        for p in range(n_wp)]
            return plan, 0, None
        p_start, p_end = window
    else:
        p_start, p_end = 0, n_wp - 1
        window = (p_start, p_end)

    # ── θ/scale offset bounds ─────────────────────────────
    th_step_deg = 360.0 / n_theta
    max_dth = int(round(max_dtheta_deg / th_step_deg))
    max_ds = max_ds_steps

    th_step = 2.0 * math.pi / n_theta
    s_step = ((s_values[-1] - s_values[0]) / max(n_s - 1, 1)
              if n_s > 1 else 0.0)
    s_values_c = np.ascontiguousarray(s_values, dtype=np.float64)
    rf_v = float(rf)

    n_local = p_end - p_start + 1
    # Naïve window traversal time: the dt span between p_start and
    # p_end, or "1 wp = 1 dt" when wp_dt_offsets is absent.
    if wp_dt_offsets is None:
        n_local_dt = n_local
        t_offset_global = p_start
    else:
        n_local_dt = (int(wp_dt_offsets[p_end])
                      - int(wp_dt_offsets[p_start]) + 1)
        t_offset_global = int(wp_dt_offsets[p_start])
    naive_secs = n_local_dt * time_step
    # Horizon: absolute (naive + time_budget) or proportional
    # (naive * factor + extra) when time_budget_factor is set.
    if time_budget_factor is None:
        horizon_s = naive_secs + time_budget
    else:
        horizon_s = naive_secs * time_budget_factor + time_budget_extra
    t_max_local = int(math.ceil(horizon_s / time_step))
    n_dth = 2 * max_dth + 1
    n_ds = 2 * max_ds + 1
    state_space = n_local * n_dth * n_ds * t_max_local

    if verbose:
        print(f"  Window: [{p_start}, {p_end}] ({n_local} waypoints)")
        print(f"  θ step: {th_step_deg:.1f}°  |  s step: {s_step:.4f}")
        print(f"  Bounds: max_dθ = ±{max_dth} steps "
              f"(±{max_dth * th_step_deg:.0f}°), "
              f"max_ds = ±{max_ds} steps "
              f"(±{max_ds * s_step:.3f})")
        if time_budget_factor is None:
            print(f"  Time:   naive = {naive_secs:.1f} s + budget = "
                  f"{time_budget:.1f} s → horizon = {horizon_s:.1f} s "
                  f"(dt = {time_step:.2f} s, t_max = {t_max_local} steps)")
        else:
            print(f"  Time:   naive = {naive_secs:.1f} s × "
                  f"{time_budget_factor:.2f} + extra = "
                  f"{time_budget_extra:.1f} s → horizon = "
                  f"{horizon_s:.1f} s "
                  f"(dt = {time_step:.2f} s, t_max = {t_max_local} steps)")
        print(f"  State space: {n_local}×{n_dth}×{n_ds}×{t_max_local}"
              f" = {state_space:,}")
        if decoupled_actions:
            print(f"  Actions: decoupled (branching ≤ 7) "
                  f"dt_move={dt_move}, dt_rot={dt_rot}, "
                  f"dt_scale={dt_scale}, dt_wait={dt_wait}")
        else:
            print(f"  Actions: coupled (branching ≤ 26) "
                  f"dt_turn_extra={dt_turn_extra}, "
                  f"dt_backward_extra={dt_backward_extra}")

    # ── baseline cumulative cost / schedule ───────────────
    # Two arrays indexed by waypoint: the weighted spatial cost of the
    # baseline up to each waypoint, and its arrival time.  Together they
    # make both residual terms of the heuristic exact rather than lower
    # bounds, and let the edge cost charge every baseline primitive.
    cum_base_cost = compute_wp_cost_offsets(
        waypoints_px, wp_configs, rf_v, s_values_c, n_theta,
        w_path=w_path, w_rot=w_rot, w_scale=w_scale)
    if wp_dt_offsets is None:
        cum_base_dt = np.arange(n_wp, dtype=np.float64)
    else:
        cum_base_dt = np.asarray(wp_dt_offsets, dtype=np.float64)

    # ── run A* ────────────────────────────────────────────
    _warmup_jit(waypoints_px, wp_theta, wp_scale, wp_config,
                cum_base_cost, cum_base_dt, static_free, offsets_flat,
                obs_table, obs_radii, rb,
                n_theta, n_s, th_step, s_step,
                obs_heights_c, cable_offsets_c, cable_counts_c,
                h_payload_c,
                decoupled_actions, dt_move, dt_rot, dt_scale, dt_wait,
                rf_v, s_values_c)

    t_search = time_module.time()
    parent_arr, goal_flat, n_exp = _dynamic_astar_jit(
        waypoints_px, wp_theta, wp_scale, wp_config,
        cum_base_cost, cum_base_dt, static_free, offsets_flat,
        obs_table, obs_radii, rb,
        n_theta, n_s, th_step, s_step,
        p_start, p_end, max_dth, max_ds,
        w_path, w_rot, w_scale, w_time,
        allow_backward, t_max_local, max_exp,
        wa_epsilon, dt_backward_extra, dt_turn_extra,
        obs_heights_c, cable_offsets_c, cable_counts_c, h_payload_c,
        decoupled_actions, dt_move, dt_rot, dt_scale, dt_wait,
        t_offset_global,
        rf_v, s_values_c,
    )
    t_search = time_module.time() - t_search
    if verbose:
        print(f"  A* search: {t_search:.4f} s  ({n_exp} expansions)")
    if goal_flat < 0:
        return None, n_exp, window

    # reconstruct windowed plan
    n_dth_ = 2 * max_dth + 1
    n_ds_ = 2 * max_ds + 1
    path_flat = []
    cur = int(goal_flat)
    while cur >= 0:
        path_flat.append(cur)
        cur = int(parent_arr[cur])
    path_flat.reverse()

    window_plan = []
    for pf in path_flat:
        tmp = pf
        pl = tmp % n_local; tmp //= n_local
        dth_idx = tmp % n_dth_; tmp //= n_dth_
        ds_idx = tmp % n_ds_; tmp //= n_ds_
        t_local = tmp

        dth = dth_idx - max_dth
        ds_off = ds_idx - max_ds
        p_g = pl + p_start
        t_g = t_local + t_offset_global
        a_ith = (int(wp_theta[p_g]) + dth) % n_theta
        a_is = int(wp_scale[p_g]) + ds_off
        window_plan.append(
            (int(p_g), int(a_ith), int(a_is),
             int(wp_config[p_g]), int(t_g)))

    full_plan = stitch_plan(wp_theta, wp_scale, wp_config, n_wp,
                            p_start, p_end, window_plan,
                            wp_dt_offsets=wp_dt_offsets)
    full_plan = densify_plan(full_plan)
    return full_plan, n_exp, window

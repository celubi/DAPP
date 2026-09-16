"""Dynamic A* — Numba JIT core.

Kept in its own file so editing parameters or other Python code does
**not** invalidate the Numba on-disk cache (``cache=True`` keys off the
source file's modification timestamp).

State space
-----------
Each search node is ``(p_local, dθ_offset, ds_offset, t_local)``:

* ``p_local`` — index along the *windowed* sub-path
  (waypoint = ``p_local + p_start``).
* ``dθ_offset`` — bounded θ-offset *from* the spatial baseline
  ``wp_theta[p]`` (in θ-grid steps, range ``[-max_dth, +max_dth]``).
* ``ds_offset`` — bounded scale offset *from* ``wp_scale[p]``
  (range ``[-max_ds, +max_ds]``).
* ``t_local`` — discrete time step within the windowed search horizon.

The cluster config ``ic`` is **fixed per waypoint** (from the spatial
plan) — the dynamic search does not change it.  Offsets are looked up
as ``offsets_flat[wp_config[p], actual_ith, actual_is, k, dim]``.
"""

import math
import numpy as np

from numba import njit


# ═══════════════════════════════════════════════════════════
#  Min-heap (flat-array, Numba-friendly)
# ═══════════════════════════════════════════════════════════

@njit(cache=True)
def _dyn_heap_push(heap_f, heap_idx, heap_size, f_val, flat_idx):
    pos = heap_size[0]
    heap_f[pos] = f_val
    heap_idx[pos] = flat_idx
    while pos > 0:
        par = (pos - 1) >> 1
        if heap_f[par] > heap_f[pos]:
            heap_f[par], heap_f[pos] = heap_f[pos], heap_f[par]
            heap_idx[par], heap_idx[pos] = heap_idx[pos], heap_idx[par]
            pos = par
        else:
            break
    heap_size[0] += 1


@njit(cache=True)
def _dyn_heap_pop(heap_f, heap_idx, heap_size):
    top_f = heap_f[0]
    top_i = heap_idx[0]
    n = heap_size[0] - 1
    heap_size[0] = n
    heap_f[0] = heap_f[n]
    heap_idx[0] = heap_idx[n]
    pos = 0
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos
        if left < n and heap_f[left] < heap_f[smallest]:
            smallest = left
        if right < n and heap_f[right] < heap_f[smallest]:
            smallest = right
        if smallest != pos:
            heap_f[pos], heap_f[smallest] = heap_f[smallest], heap_f[pos]
            heap_idx[pos], heap_idx[smallest] = heap_idx[smallest], heap_idx[pos]
            pos = smallest
        else:
            break
    return top_f, top_i


# ═══════════════════════════════════════════════════════════
#  Dynamic A* — windowed, bounded θ/s offsets, fixed ic per waypoint
# ═══════════════════════════════════════════════════════════

@njit(cache=True)
def dynamic_astar_jit(
    waypoints_px, wp_theta, wp_scale, wp_config,
    cum_base_cost, cum_base_dt,
    static_free, offsets_flat,
    obs_table, obs_radii, rb,
    n_theta, n_s, th_step, s_step,
    p_start, p_end,
    max_dth, max_ds,
    w_path, w_rot, w_scale_w, w_time,
    allow_backward,
    t_max, max_exp,
    wa_epsilon,
    dt_backward_extra,
    dt_turn_extra,
    obs_heights, cable_offsets, cable_counts, h_payload,
    decoupled_actions, dt_move, dt_rot_act, dt_scale_act, dt_wait,
    t_offset_global,
    rf, s_values,
):
    """Numba JIT dynamic A* over ``(p_local, dθ, ds, t_local)``.

    Returns
    -------
    parent : int64 array — back-pointers (flat indices)
    goal_flat : int64 — flat index of goal state (``-1`` if not found)
    n_expanded : int  — number of nodes expanded
    """
    INF = 1e30
    n_local = p_end - p_start + 1
    n_dth = 2 * max_dth + 1
    n_ds = 2 * max_ds + 1
    n_robots = offsets_flat.shape[3]
    n_obs = obs_radii.shape[0]
    obs_t_max = 0
    if n_obs > 0:
        obs_t_max = obs_table.shape[1]

    total = n_local * n_dth * n_ds * t_max

    s_pl = 1
    s_dth = n_local
    s_ds = n_local * n_dth
    s_t = n_local * n_dth * n_ds

    g_cost = np.full(total, INF, dtype=np.float64)
    parent = np.full(total, -2, dtype=np.int64)
    closed = np.zeros(total, dtype=np.bool_)

    # Global-time offset of the window start: wp_dt_offsets[p_start],
    # or p_start when all per-axis dt's are 1.
    t_offset = t_offset_global
    goal_pl = n_local - 1

    start_flat = 0 + max_dth * s_dth + max_ds * s_ds
    g_cost[start_flat] = 0.0
    parent[start_flat] = -1

    rem_cost = cum_base_cost[p_end] - cum_base_cost[p_start]
    if rem_cost < 0.0:
        rem_cost = 0.0
    if decoupled_actions:
        h0_time = (cum_base_dt[p_end] - cum_base_dt[p_start]) * w_time
    else:
        h0_time = goal_pl * w_time
    h0 = (rem_cost + h0_time) * wa_epsilon

    heap_cap = min(total, max_exp * 2)
    if heap_cap < 1:
        heap_cap = 1
    heap_f = np.empty(heap_cap, dtype=np.float64)
    heap_idx = np.empty(heap_cap, dtype=np.int64)
    heap_size = np.zeros(1, dtype=np.int64)

    _dyn_heap_push(heap_f, heap_idx, heap_size, h0, start_flat)

    n_expanded = 0
    goal_flat_result = np.int64(-1)

    while heap_size[0] > 0:
        _, cur_flat = _dyn_heap_pop(heap_f, heap_idx, heap_size)

        if closed[cur_flat]:
            continue
        closed[cur_flat] = True
        n_expanded += 1

        tmp = cur_flat
        c_pl = tmp % n_local; tmp //= n_local
        c_dth_idx = tmp % n_dth; tmp //= n_dth
        c_ds_idx = tmp % n_ds; tmp //= n_ds
        c_t = tmp

        c_dth = c_dth_idx - max_dth
        c_ds = c_ds_idx - max_ds

        if c_pl == goal_pl and c_dth == 0 and c_ds == 0:
            goal_flat_result = cur_flat
            break

        if n_expanded >= max_exp:
            break

        cg = g_cost[cur_flat]
        c_p_global = c_pl + p_start

        # Enumerate successor (dp, ddth, dds) triplets.  Coupled mode:
        # every combination of {-1,0,+1}^3 minus (0,0,0) is one
        # compound action.  Decoupled mode: exactly one axis changes
        # per action.
        for action_id in range(27):
            dp = (action_id % 3) - 1
            ddth = ((action_id // 3) % 3) - 1
            dds = ((action_id // 9) % 3) - 1
            if dp == -1 and not allow_backward:
                continue
            if decoupled_actions:
                # Single-axis actions plus the wait action (all zero:
                # hold pose for one dt); compound actions forbidden.
                nonzero = 0
                if dp != 0:
                    nonzero += 1
                if ddth != 0:
                    nonzero += 1
                if dds != 0:
                    nonzero += 1
                if nonzero > 1:
                    continue
            else:
                # Coupled mode: (0,0,0) is not an action.
                if dp == 0 and ddth == 0 and dds == 0:
                    continue

            npl = c_pl + dp
            ndth = c_dth + ddth
            nds = c_ds + dds

            if npl < 0 or npl >= n_local:
                continue
            if ndth < -max_dth or ndth > max_dth:
                continue
            if nds < -max_ds or nds > max_ds:
                continue

            n_p_global = npl + p_start
            actual_ith = (wp_theta[n_p_global] + ndth) % n_theta
            actual_is = wp_scale[n_p_global] + nds
            if actual_is < 0 or actual_is >= n_s:
                continue

            if not static_free[n_p_global, actual_ith, actual_is]:
                continue

            if decoupled_actions:
                if dp != 0:
                    # A dp step through an in-place waypoint (same
                    # pixel, different θ or s) is semantically a
                    # rotation or scale and is billed as such.
                    same_px = (waypoints_px[n_p_global, 0]
                               == waypoints_px[c_p_global, 0]
                               and waypoints_px[n_p_global, 1]
                               == waypoints_px[c_p_global, 1])
                    if same_px:
                        if wp_theta[n_p_global] != wp_theta[c_p_global]:
                            dt_action = dt_rot_act
                        elif wp_scale[n_p_global] != wp_scale[c_p_global]:
                            dt_action = dt_scale_act
                        else:
                            dt_action = dt_move
                    else:
                        dt_action = dt_move
                    if dp == -1:
                        dt_action += dt_backward_extra
                elif ddth != 0:
                    dt_action = dt_rot_act
                elif dds != 0:
                    dt_action = dt_scale_act
                else:
                    # Wait action: hold pose for dt_wait time steps.
                    dt_action = dt_wait
            else:
                dt_action = 1
                if dp == -1:
                    dt_action += dt_backward_extra
                if ddth != 0:
                    dt_action += dt_turn_extra
            nt = c_t + dt_action
            if nt >= t_max:
                continue

            ndth_idx = ndth + max_dth
            nds_idx = nds + max_ds
            nb_flat = (npl * s_pl + ndth_idx * s_dth
                       + nds_idx * s_ds + nt * s_t)

            if closed[nb_flat]:
                continue

            ic_n = wp_config[n_p_global]
            cx = waypoints_px[n_p_global, 0]
            cy = waypoints_px[n_p_global, 1]
            h_thresh = h_payload[actual_is]
            K = cable_counts[actual_is]
            collision = False
            for t_check in range(c_t + 1, nt + 1):
                global_t = t_check + t_offset
                if n_obs > 0 and 0 <= global_t < obs_t_max:
                    for oi in range(n_obs):
                        ox = obs_table[oi, global_t, 0]
                        oy = obs_table[oi, global_t, 1]
                        if ox != ox:  # NaN
                            continue
                        thr2 = (rb + obs_radii[oi]) ** 2
                        for k in range(n_robots):
                            dx = (cx + offsets_flat[ic_n, actual_ith,
                                                    actual_is, k, 0] - ox)
                            dy = (cy + offsets_flat[ic_n, actual_ith,
                                                    actual_is, k, 1] - oy)
                            if dx * dx + dy * dy < thr2:
                                collision = True
                                break
                        if collision:
                            break
                        # Payload / cable sample check: a sample inside
                        # the obstacle disc collides only when the
                        # obstacle is tall enough to block the payload.
                        thr2_o = obs_radii[oi] * obs_radii[oi]
                        obs_tall = obs_heights[oi] > h_thresh
                        for k in range(K):
                            dx = (cx + cable_offsets[ic_n, actual_ith,
                                                     actual_is, k, 0]
                                  - ox)
                            dy = (cy + cable_offsets[ic_n, actual_ith,
                                                     actual_is, k, 1]
                                  - oy)
                            if dx * dx + dy * dy < thr2_o and obs_tall:
                                collision = True
                                break
                        if collision:
                            break
                if collision:
                    break
            if collision:
                continue

            mc = w_time * dt_action
            if dp != 0:
                # Spatial cost of the baseline primitive crossed here:
                # a translation's length, or the robot travel of an
                # in-place rotation/scaling.  Same either direction;
                # weights are folded into cum_base_cost.
                lo_p = n_p_global if dp < 0 else c_p_global
                mc += cum_base_cost[lo_p + 1] - cum_base_cost[lo_p]
            # Rotation / scale costs as real pixel travel of the
            # outermost robot (radius rf·s), matching da_astar._expand:
            # an arc rf·s·Δθ for rotation, a radial rf·|Δs| for scale.
            mc += abs(ddth) * th_step * rf * s_values[actual_is] * w_rot
            mc += abs(dds) * s_step * rf * w_scale_w
            ng = cg + mc

            if ng < g_cost[nb_flat]:
                g_cost[nb_flat] = ng
                parent[nb_flat] = cur_flat

                rem = (cum_base_cost[p_end]
                       - cum_base_cost[n_p_global])
                if rem < 0.0:
                    rem = 0.0
                h_path = rem
                # Admissible: rotate at the smallest reachable radius
                # (rf·s_min) — doing it at any larger scale costs more.
                h_rot = abs(ndth) * th_step * rf * s_values[0] * w_rot
                h_sc = abs(nds) * s_step * rf * w_scale_w
                # Admissible time lower bound.  Coupled: one compound
                # action advances all axes per dt → max(steps_p,
                # |ndth|, |nds|) plus dt_turn_extra per turn.
                # Decoupled: one axis per action → the exact baseline
                # residual plus the offset corrections.
                steps_p = goal_pl - npl
                abs_ndth = ndth if ndth >= 0 else -ndth
                abs_nds = nds if nds >= 0 else -nds
                if decoupled_actions:
                    # steps_p * dt_move would price every remaining
                    # transition at the cheapest primitive, missing
                    # (dt_scale - dt_move) per in-place scaling ahead.
                    rem_dt = (cum_base_dt[p_end]
                              - cum_base_dt[n_p_global])
                    if rem_dt < 0.0:
                        rem_dt = 0.0
                    h_time = (rem_dt
                              + abs_ndth * dt_rot_act
                              + abs_nds * dt_scale_act) * w_time
                else:
                    min_moves = steps_p
                    if abs_ndth > min_moves:
                        min_moves = abs_ndth
                    if abs_nds > min_moves:
                        min_moves = abs_nds
                    h_time = (min_moves
                              + abs_ndth * dt_turn_extra) * w_time
                fv = ng + (h_path + h_rot + h_sc + h_time) * wa_epsilon

                if heap_size[0] < heap_cap:
                    _dyn_heap_push(heap_f, heap_idx,
                                   heap_size, fv, nb_flat)

    return parent, goal_flat_result, n_expanded

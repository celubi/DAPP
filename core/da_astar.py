"""
DA_astar — Deformation-Aware A* with Numba JIT.

Lazy A* over the 5-D state (x, y, θ, s, c): neighbours are generated
on demand during expansion, so memory scales with expanded nodes, not
with the grid.  Optional pruning: loose-space (``c_deform``, with
``prune_rot_in_free`` and ``c_admit_per_config``) and symmetry
(``use_symmetry``: θ folded to each template's fundamental period,
reconfiguration branching over the distinct orbit representatives).

Entry points: ``find_path_da``, ``find_path_da_from_map``.
"""

import math
import time

import numpy as np
import numba

from .map_io import load_map, compute_signed_clearance
from .formations import (
    parse_formations,
    compute_reconfig_costs,
    precompute_offsets,
)


# ═══════════════════════════════════════════════════════════
#  Numba JIT — collision check
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _check_free(cx, cy, offsets_arr, ic, it, js, dist_map, rb):
    """True when every robot disc at centre (cx, cy) clears obstacles.

    Parameters
    ----------
    cx, cy : int — pixel coordinates of formation centre
    offsets_arr : (n_config, n_theta, n_s, n_robots, 2) int32
    ic, it, js : int — config, theta-index, scale-index
    dist_map : (H, W) float64 — Euclidean distance transform
    rb : float — robot body radius
    """
    H, W = dist_map.shape
    n_robots = offsets_arr.shape[3]
    for k in range(n_robots):
        rx = cx + offsets_arr[ic, it, js, k, 0]
        ry = cy + offsets_arr[ic, it, js, k, 1]
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return False
        if dist_map[ry, rx] < rb:
            return False
    return True


# ═══════════════════════════════════════════════════════════
#  Numba JIT — arc collision check for reconfiguration
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _check_arc_free(cx, cy, ic_src, ic_dst, it, js,
                    arc_angles, n_arc_pts,
                    rf, s_values, th_step,
                    dist_map, rb, rep_ofs):
    """Check that intermediate robot positions along arcs are clear.

    During a reconfiguration transition each robot slides along an
    arc on the formation circle from its source slot to its
    destination slot.  This function samples those intermediate
    positions and verifies that every sample clears obstacles.

    Parameters
    ----------
    cx, cy : int — pixel coordinates of formation centre
    ic_src, ic_dst : int — source / destination config index
    it : int — current θ-index (source config)
    js : int — current scale index
    arc_angles : (n_config, n_config, n_arc_pts) float64
        Precomputed intermediate base-angles (relative to formation
        centre, without orientation θ).
    n_arc_pts : int — total sample count (n_robots × n_arc_samples)
    rf : float — formation radius
    s_values : (n_s,) float64 — scale values
    th_step : float — angular step (2π / n_theta)
    dist_map : (H, W) float64 — Euclidean distance transform
    rb : float — robot body radius
    rep_ofs : int — orbit-representative offset in θ-steps
        (``m·period_src``).  A branching reconfig edge is the identity
        reassignment sweep rigidly rotated by ``rep_ofs`` θ-steps, so
        the precomputed arcs are reused with this constant offset
        added; ``rep_ofs == 0`` is the identity sweep.
    """
    H, W = dist_map.shape
    theta = (it + rep_ofs) * th_step
    r = rf * s_values[js]
    for idx in range(n_arc_pts):
        a_mid = arc_angles[ic_src, ic_dst, idx]
        angle = theta + a_mid
        rx = int(round(r * math.cos(angle))) + cx
        ry = int(round(r * math.sin(angle))) + cy
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return False
        if dist_map[ry, rx] < rb:
            return False
    return True


# ═══════════════════════════════════════════════════════════
#  Numba JIT — payload / cable height collision check
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _check_payload_free(cx, cy, cable_offsets, cable_counts,
                        ic, it, js, height_map, h_payload,
                        do_payload_check):
    """True when no cable sample lies under a tall enough obstacle.

    Samples along each robot-to-centre cable are checked against the
    height map: a pixel with ``height > h_payload[js]`` blocks the
    formation centred at ``(cx, cy)`` in state ``(ic, it, js)``.

    When ``do_payload_check == 0`` returns ``True`` immediately so
    the caller pays at most one comparison.

    Parameters
    ----------
    cx, cy : int — pixel coordinates of formation centre
    cable_offsets : (n_config, n_theta, n_s, K_max, 2) int32
    cable_counts : (n_s,) int32 — valid sample count per scale
    ic, it, js : int
    height_map : (H, W) uint8 — obstacle height per pixel (0 = free)
    h_payload : (n_s,) float64 — payload height at each scale
    do_payload_check : int — 0 disables the check, any other value
        enables it (kept as a plain int so the JIT signature stays
        simple)
    """
    if do_payload_check == 0:
        return True
    H, W = height_map.shape
    K = cable_counts[js]
    h_thresh = h_payload[js]
    for k in range(K):
        rx = cx + cable_offsets[ic, it, js, k, 0]
        ry = cy + cable_offsets[ic, it, js, k, 1]
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return False
        if height_map[ry, rx] > h_thresh:
            return False
    return True


# ═══════════════════════════════════════════════════════════
#  Numba JIT — neighbor expansion with adaptive branching
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _expand(ix, iy, it, js, ic,
            dist_map, offsets_arr, xy_step,
            n_ix, n_iy, n_s, n_config,
            periods, rb,
            w_move, w_rot, w_scale, w_config,
            th_step, s_step,
            reconfig_costs,
            clearance_grid, c_deform,
            c_admit_per_config,
            reconfig_mode, arc_angles, n_arc_pts, rf, s_values,
            check_counts,
            cable_offsets, cable_counts, height_map, h_payload,
            js_admissible, do_payload_check,
            prune_rot_in_free, gx, gy,
            out_states, out_costs):
    """Generate valid neighbors for state (ix, iy, it, js, ic).

    Writes to pre-allocated ``out_states`` (N, 5) and ``out_costs``
    (N,).  Returns the number of neighbors written.

    Pruning inputs:

    * ``c_deform``: above this centre-clearance threshold, scale and
      reconfiguration edges are suppressed.
    * ``c_admit_per_config[ic]``: minimum clearance for config ``ic``
      — spatial moves to cells below the current config's threshold
      and reconfigurations to configs whose threshold the current
      cell fails are skipped.  All-zero entries disable this.
    * ``prune_rot_in_free``: also suppress rotation edges in open
      space, except on the goal cell ``(gx, gy)`` so a θ-constrained
      goal stays reachable (pass ``gx = gy = -1`` to disable the
      exception, e.g. multi-goal callers).
    """
    SQRT2 = 1.4142135623730951
    period = periods[ic]
    # n_theta is the full θ resolution; offsets_arr is shaped
    # (n_config, n_theta, n_s, n_robots, 2), so axis 1 gives it without
    # threading an extra parameter through every call site.
    n_theta = offsets_arr.shape[1]
    ptr = 0

    # Adaptive branching: deform only near obstacles
    deform = (c_deform < 0.0) or (clearance_grid[iy, ix] <= c_deform)

    # ── 8 spatial moves ──────────────────────────────────
    for dmx in (-1, 0, 1):
        for dmy in (-1, 0, 1):
            if dmx == 0 and dmy == 0:
                continue
            nix = ix + dmx
            niy = iy + dmy
            if nix < 0 or nix >= n_ix or niy < 0 or niy >= n_iy:
                continue
            # Label-aware admissibility: skip if dest cell clearance
            # is below the current config's required threshold.
            if clearance_grid[niy, nix] < c_admit_per_config[ic]:
                continue
            check_counts[ic] += 1
            if not _check_free(nix * xy_step, niy * xy_step,
                               offsets_arr, ic, it, js, dist_map, rb):
                continue
            if not _check_payload_free(nix * xy_step, niy * xy_step,
                                       cable_offsets, cable_counts,
                                       ic, it, js, height_map, h_payload,
                                       do_payload_check):
                continue
            if abs(dmx) + abs(dmy) == 2:
                mc = SQRT2 * xy_step * w_move
            else:
                mc = float(xy_step) * w_move
            out_states[ptr, 0] = nix
            out_states[ptr, 1] = niy
            out_states[ptr, 2] = it
            out_states[ptr, 3] = js
            out_states[ptr, 4] = ic
            out_costs[ptr] = mc
            ptr += 1

    cx = ix * xy_step
    cy = iy * xy_step

    # ── 2 rotation moves ────────────────────────────────
    rotate = deform or (not prune_rot_in_free) or (ix == gx and iy == gy)
    if rotate:
        for dth in (-1, 1):
            nit = (it + dth) % period
            check_counts[ic] += 1
            if not _check_free(cx, cy, offsets_arr, ic, nit, js,
                               dist_map, rb):
                continue
            if not _check_payload_free(cx, cy,
                                       cable_offsets, cable_counts,
                                       ic, nit, js, height_map, h_payload,
                                       do_payload_check):
                continue
            out_states[ptr, 0] = ix
            out_states[ptr, 1] = iy
            out_states[ptr, 2] = nit
            out_states[ptr, 3] = js
            out_states[ptr, 4] = ic
            # Arc length swept by the outermost robot (radius rf·s) when
            # the formation rotates one θ-step → cost in pixels of real
            # travel.
            out_costs[ptr] = rf * s_values[js] * th_step * w_rot
            ptr += 1

    # ── Scale + reconfig (only in deformation zone) ─────
    if deform:
        for dsc in (-1, 1):
            njs = js + dsc
            if njs < 0 or njs >= n_s:
                continue
            if not js_admissible[njs]:
                continue
            check_counts[ic] += 1
            if not _check_free(cx, cy, offsets_arr, ic, it, njs,
                               dist_map, rb):
                continue
            if not _check_payload_free(cx, cy,
                                       cable_offsets, cable_counts,
                                       ic, it, njs, height_map, h_payload,
                                       do_payload_check):
                continue
            out_states[ptr, 0] = ix
            out_states[ptr, 1] = iy
            out_states[ptr, 2] = it
            out_states[ptr, 3] = njs
            out_states[ptr, 4] = ic
            # Radial travel of each robot for a one-step scale change:
            # the radius rf·s shifts by rf·|Δs| = rf·s_step pixels.
            out_costs[ptr] = rf * s_step * w_scale
            ptr += 1

        for ic_dst in range(n_config):
            if ic_dst == ic:
                continue
            # Label-aware admissibility: skip if current cell clearance
            # is below the target config's required threshold.
            if clearance_grid[iy, ix] < c_admit_per_config[ic_dst]:
                continue
            p_dst = periods[ic_dst]
            # Symmetry-aware reconfiguration: one edge per DISTINCT
            # orbit representative — landing orientations
            # (it + m*period) % p_dst, deduped on the fly.  With
            # symmetry OFF (period == n_theta) this reduces to the
            # single edge nit = it.
            n_reps = n_theta // period

            for m in range(n_reps):
                nit = (it + m * period) % p_dst
                # Dedup: skip if this nit was already produced for this
                # ic_dst in the current expansion.
                dup = False
                if m > 0:
                    for mm in range(m):
                        if (it + mm * period) % p_dst == nit:
                            dup = True
                            break
                if dup:
                    continue

                check_counts[ic_dst] += 1
                if not _check_free(cx, cy, offsets_arr, ic_dst, nit, js,
                                   dist_map, rb):
                    continue
                if not _check_payload_free(cx, cy,
                                           cable_offsets, cable_counts,
                                           ic_dst, nit, js,
                                           height_map, h_payload,
                                           do_payload_check):
                    continue
                # Transition collision check
                if reconfig_mode == 1:          # clearance
                    if dist_map[cy, cx] < s_values[js] * rf + rb:
                        continue
                elif reconfig_mode == 2:        # arc sampling
                    # The physical sweep is the identity sweep rigidly
                    # rotated by m*period θ-steps — sample the arcs
                    # around that rotated representative.
                    if not _check_arc_free(cx, cy, ic, ic_dst, it, js,
                                           arc_angles, n_arc_pts,
                                           rf, s_values, th_step,
                                           dist_map, rb, m * period):
                        continue
                out_states[ptr, 0] = ix
                out_states[ptr, 1] = iy
                out_states[ptr, 2] = nit
                out_states[ptr, 3] = js
                out_states[ptr, 4] = ic_dst
                # Reconfig cost: worst-case angular travel (radians) of
                # the makespan-optimal reassignment, scaled by rf·s to
                # pixels of real arc travel.
                out_costs[ptr] = (reconfig_costs[ic, ic_dst]
                                  * rf * s_values[js] * w_config)
                ptr += 1

    return ptr


# ═══════════════════════════════════════════════════════════
#  Numba JIT — heuristic
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _sym_rot_dist(it_n, ic_n, it_g, ic_g, n_theta, periods):
    """Min rotation steps accounting for per-config symmetry."""
    p_n = periods[ic_n]
    p_g = periods[ic_g]
    k_n = n_theta // p_n
    k_g = n_theta // p_g
    best = n_theta
    for m in range(k_n):
        tn = (it_n + m * p_n) % n_theta
        for mp in range(k_g):
            tg = (it_g + mp * p_g) % n_theta
            d = abs(tn - tg)
            if n_theta - d < d:
                d = n_theta - d
            if d < best:
                best = d
    return best


@numba.njit(cache=True)
def _heuristic(ix, iy, it, js, ic,
               gx, gy, gt, gs, gc,
               xy_step, n_theta, th_step, s_step,
               w_move, w_rot, w_scale, w_config,
               periods, reconfig_min,
               free_theta, free_s, free_config,
               rf, s_min):
    """Admissible A* heuristic for the 5-D formation state.

    Rotation and scale costs are arc/radial lengths in pixels
    (``rf · s · Δθ`` and ``rf · |Δs|``), matching :func:`_expand`.  To
    stay admissible the rotation term uses the *smallest* radius the
    path can reach (``rf · s_min``): rotating at a larger scale only
    ever costs more, so this never overestimates.
    """
    dx = float(ix - gx) * xy_step
    dy = float(iy - gy) * xy_step
    val = math.sqrt(dx * dx + dy * dy) * w_move
    if not free_theta:
        rd = _sym_rot_dist(it, ic, gt, gc, n_theta, periods)
        val += rd * th_step * rf * s_min * w_rot
    if not free_s:
        val += abs(js - gs) * s_step * rf * w_scale
    if not free_config:
        # Same rf·s scaling as the reconfig edge cost; rf·s_min keeps the
        # bound admissible (reconfiguring at a larger scale costs more).
        val += reconfig_min[ic, gc] * rf * s_min * w_config
    return val


# ═══════════════════════════════════════════════════════════
#  Setup helpers
# ═══════════════════════════════════════════════════════════

def _dummy_payload_arrays(n_config, n_theta, n_s):
    """Shape-correct placeholder arrays for the payload-check parameters.

    Returned alongside ``do_payload_check = 0``, these let ``_expand``
    accept a single JIT signature whether or not the caller wants
    payload checking — the check short-circuits on the flag, so
    pixel contents are never read.
    """
    hmap = np.zeros((1, 1), dtype=np.uint8)
    hpay = np.zeros(n_s, dtype=np.float64)
    cab_off = np.zeros((n_config, n_theta, n_s, 1, 2), dtype=np.int32)
    cab_cnt = np.zeros(n_s, dtype=np.int32)
    js_adm = np.ones(n_s, dtype=np.bool_)
    return hmap, hpay, cab_off, cab_cnt, js_adm


def _offsets_to_array(offsets_dict, n_config, n_theta, n_s, n_robots):
    """Convert ``{(ic, it, js): (n_robots, 2)}`` → 5-D int32 array.

    Shape: ``(n_config, n_theta, n_s, n_robots, 2)``.
    Only entries with ``it < periods[ic]`` contain valid data;
    the rest are zero-filled and never accessed.
    """
    arr = np.zeros((n_config, n_theta, n_s, n_robots, 2), dtype=np.int32)
    for (ic, it, js), off in offsets_dict.items():
        arr[ic, it, js] = off
    return arr


def _precompute_clearance_grid(dist_map, xy_step):
    """Distance-transform sampled at each grid centre → (n_iy, n_ix)."""
    H, W = dist_map.shape
    n_iy = (H - 1) // xy_step + 1
    n_ix = (W - 1) // xy_step + 1
    ys = np.clip(np.arange(n_iy) * xy_step, 0, H - 1)
    xs = np.clip(np.arange(n_ix) * xy_step, 0, W - 1)
    return dist_map[np.ix_(ys, xs)].astype(np.float64, copy=True)


def _shortest_reconfig(raw_costs):
    """Floyd–Warshall all-pairs shortest reconfig costs."""
    d = raw_costs.astype(np.float64).copy()
    n = d.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via = d[i, k] + d[k, j]
                if via < d[i, j]:
                    d[i, j] = via
    return d


def _precompute_arc_angles(formations_rad, assignments, n_arc_samples):
    """Precompute intermediate arc angles for reconfiguration transitions.

    For each config pair (src, dst) and each robot, samples
    ``n_arc_samples`` intermediate angles along the shorter arc
    from the source slot to the assigned destination slot.

    Returns ndarray (n_config, n_config, n_robots * n_arc_samples).
    """
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])
    n_pts = n_robots * n_arc_samples
    t_values = np.linspace(0, 1, n_arc_samples + 2)[1:-1]  # interior only

    arc_angles = np.zeros((n_config, n_config, n_pts), dtype=np.float64)

    for ic_src in range(n_config):
        for ic_dst in range(n_config):
            if ic_src == ic_dst:
                continue
            perm = assignments[(ic_src, ic_dst)]
            idx = 0
            for k in range(n_robots):
                a_start = float(formations_rad[ic_src][k])
                a_end = float(formations_rad[ic_dst][perm[k]])
                diff = (a_end - a_start) % (2 * math.pi)
                if diff > math.pi:
                    diff -= 2 * math.pi
                for t in t_values:
                    arc_angles[ic_src, ic_dst, idx] = a_start + diff * t
                    idx += 1

    return arc_angles


def _resolve_goals(goal, n_s, n_config, periods,
                   free_theta, free_s, free_config,
                   offsets_arr, dist_map, rb, xy_step):
    """Expand goal spec → set of collision-free 5-tuples."""
    ix, iy, it, js = goal[:4]
    ic = goal[4] if len(goal) >= 5 else 0
    goals = set()
    for c in (range(n_config) if free_config else (ic,)):
        p = periods[c]
        for t in (range(p) if free_theta else (it % p,)):
            for s in (range(n_s) if free_s else (js,)):
                if _check_free(int(ix) * xy_step, int(iy) * xy_step,
                               offsets_arr, int(c), int(t), int(s),
                               dist_map, rb):
                    goals.add((int(ix), int(iy), int(t), int(s), int(c)))
    return goals


# ═══════════════════════════════════════════════════════════
#  Numba JIT — A* core kernel
# ═══════════════════════════════════════════════════════════
#
#  The whole search loop runs in nopython mode.  State 5-tuples are
#  packed into a single int64 key so g_cost / closed / parent are
#  ``numba.typed.Dict`` instances keyed by int64; the open set is a
#  binary min-heap on three parallel preallocated arrays (f-value,
#  insertion counter, key) with hand-written sift-up / sift-down
#  (``heapq`` is unavailable under njit).

@numba.njit(cache=True, inline='always')
def _pack_state(ix, iy, it, js, ic, n_iy, n_theta, n_s, n_config):
    """(ix, iy, it, js, ic) → single int64 key (positional encoding)."""
    k = ix
    k = k * n_iy + iy
    k = k * n_theta + it
    k = k * n_s + js
    k = k * n_config + ic
    return k


@numba.njit(cache=True, inline='always')
def _unpack_state(k, n_iy, n_theta, n_s, n_config):
    """int64 key → (ix, iy, it, js, ic)."""
    ic = k % n_config
    k //= n_config
    js = k % n_s
    k //= n_s
    it = k % n_theta
    k //= n_theta
    iy = k % n_iy
    k //= n_iy
    ix = k
    return ix, iy, it, js, ic


@numba.njit(cache=True)
def _heap_push(hf, hc, hk, size, f, cnt, key):
    """Push (f, cnt, key) onto the binary min-heap; return new size.

    Ordering matches Python's ``heapq`` on the (f, cnt) tuple: ``cnt``
    is a strictly increasing insertion counter so ties on ``f`` break
    deterministically (FIFO), reproducing the reference behaviour.
    """
    i = size
    hf[i] = f
    hc[i] = cnt
    hk[i] = key
    # sift up
    while i > 0:
        parent = (i - 1) >> 1
        if hf[parent] < hf[i] or (hf[parent] == hf[i] and hc[parent] <= hc[i]):
            break
        # swap i, parent
        hf[i], hf[parent] = hf[parent], hf[i]
        hc[i], hc[parent] = hc[parent], hc[i]
        hk[i], hk[parent] = hk[parent], hk[i]
        i = parent
    return size + 1


@numba.njit(cache=True)
def _heap_pop(hf, hc, hk, size):
    """Pop the min element; return (f, cnt, key, new_size)."""
    f0 = hf[0]
    c0 = hc[0]
    k0 = hk[0]
    size -= 1
    # move last to root
    hf[0] = hf[size]
    hc[0] = hc[size]
    hk[0] = hk[size]
    # sift down
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        smallest = i
        if left < size and (
                hf[left] < hf[smallest]
                or (hf[left] == hf[smallest] and hc[left] < hc[smallest])):
            smallest = left
        if right < size and (
                hf[right] < hf[smallest]
                or (hf[right] == hf[smallest] and hc[right] < hc[smallest])):
            smallest = right
        if smallest == i:
            break
        hf[i], hf[smallest] = hf[smallest], hf[i]
        hc[i], hc[smallest] = hc[smallest], hc[i]
        hk[i], hk[smallest] = hk[smallest], hk[i]
        i = smallest
    return f0, c0, k0, size


@numba.njit(cache=True)
def _astar_kernel(
        s0_key, goal_keys,
        gx, gy, gt, gs, gc,
        dist_map, offsets_arr, xy_step,
        n_ix, n_iy, n_s, n_config, n_theta,
        periods, rb,
        w_move, w_rot, w_scale, w_config,
        th_step, s_step,
        reconfig_costs, reconfig_min,
        clearance_grid, c_deform,
        c_admit_per_config,
        reconfig_mode, arc_angles, n_arc_pts, rf, s_values,
        check_counts,
        cable_offsets, cable_counts, height_map, h_payload,
        js_admissible, do_payload_check,
        free_theta, free_s, free_config,
        prune_rot_in_free,
        cost_lim, heap_cap):
    """Run A* entirely in nopython mode.

    Returns ``(found, goal_key, cost, n_expanded, parent_dict)``.
    ``parent_dict`` maps each settled state-key to its predecessor key
    (the start maps to ``-1``); the Python wrapper walks it to rebuild
    the path.  When ``found == 0`` the search exhausted the open set
    (or hit ``cost_lim``) without reaching a goal.
    """
    # Typed dicts keyed by packed int64 state.
    g_cost = numba.typed.Dict.empty(numba.int64, numba.float64)
    parent = numba.typed.Dict.empty(numba.int64, numba.int64)
    closed = numba.typed.Dict.empty(numba.int64, numba.boolean)

    # Goal membership as a typed set-like dict.
    goal_set = numba.typed.Dict.empty(numba.int64, numba.boolean)
    for gi in range(goal_keys.shape[0]):
        goal_set[goal_keys[gi]] = True

    # Binary min-heap on parallel arrays.
    hf = np.empty(heap_cap, dtype=np.float64)
    hc = np.empty(heap_cap, dtype=np.int64)
    hk = np.empty(heap_cap, dtype=np.int64)
    hsize = 0

    # Per-expansion neighbour scratch: 8 spatial + 2 rot + 2 scale +
    # up to (n_theta // min_period) branching reconfig edges per
    # destination config.
    min_period = n_theta
    for _ic in range(n_config):
        if periods[_ic] < min_period:
            min_period = periods[_ic]
    max_reconfig = (n_config - 1) * (n_theta // min_period) if n_config > 1 else 0
    max_nb = 12 + max_reconfig
    out_states = np.empty((max_nb, 5), dtype=np.int64)
    out_costs = np.empty(max_nb, dtype=np.float64)

    g_cost[s0_key] = 0.0
    parent[s0_key] = -1

    ix0, iy0, it0, js0, ic0 = _unpack_state(
        s0_key, n_iy, n_theta, n_s, n_config)
    h0 = _heuristic(ix0, iy0, it0, js0, ic0,
                    gx, gy, gt, gs, gc,
                    xy_step, n_theta, th_step, s_step,
                    w_move, w_rot, w_scale, w_config,
                    periods, reconfig_min,
                    free_theta, free_s, free_config,
                    rf, s_values[0])
    hsize = _heap_push(hf, hc, hk, hsize, h0, 0, s0_key)
    cnt = 1
    n_exp = 0

    while hsize > 0:
        f_val, _, u_key, hsize = _heap_pop(hf, hc, hk, hsize)
        if f_val > cost_lim:
            break
        if u_key in closed:
            continue
        closed[u_key] = True
        n_exp += 1

        if u_key in goal_set:
            return 1, u_key, g_cost[u_key], n_exp, parent

        ug = g_cost[u_key]
        uix, uiy, uit, ujs, uic = _unpack_state(
            u_key, n_iy, n_theta, n_s, n_config)

        n_nb = _expand(uix, uiy, uit, ujs, uic,
                       dist_map, offsets_arr, xy_step,
                       n_ix, n_iy, n_s, n_config,
                       periods, rb,
                       w_move, w_rot, w_scale, w_config,
                       th_step, s_step,
                       reconfig_costs, clearance_grid, c_deform,
                       c_admit_per_config,
                       reconfig_mode, arc_angles, n_arc_pts, rf, s_values,
                       check_counts,
                       cable_offsets, cable_counts, height_map, h_payload,
                       js_admissible, do_payload_check,
                       prune_rot_in_free, gx, gy,
                       out_states, out_costs)

        for i in range(n_nb):
            vix = out_states[i, 0]
            viy = out_states[i, 1]
            vit = out_states[i, 2]
            vjs = out_states[i, 3]
            vic = out_states[i, 4]
            v_key = _pack_state(vix, viy, vit, vjs, vic,
                                n_iy, n_theta, n_s, n_config)
            if v_key in closed:
                continue
            ng = ug + out_costs[i]
            # g_cost.get(v, inf) — typed dict has no .get with default
            old = g_cost.get(v_key, np.inf)
            if ng < old:
                g_cost[v_key] = ng
                parent[v_key] = u_key
                hv = _heuristic(vix, viy, vit, vjs, vic,
                                gx, gy, gt, gs, gc,
                                xy_step, n_theta, th_step, s_step,
                                w_move, w_rot, w_scale, w_config,
                                periods, reconfig_min,
                                free_theta, free_s, free_config,
                                rf, s_values[0])
                hsize = _heap_push(hf, hc, hk, hsize, ng + hv, cnt, v_key)
                cnt += 1

    return 0, np.int64(-1), np.inf, n_exp, parent


# ═══════════════════════════════════════════════════════════
#  DA A*
# ═══════════════════════════════════════════════════════════

def find_path_da(
    dist_map, offsets_arr, s_values, periods,
    formations_rad, reconfig_costs, reconfig_min,
    rb, xy_step, n_theta, n_s,
    w_move, w_rot, w_scale, w_config,
    start, goal,
    free_theta=False, free_s=False, free_config=False,
    clearance_grid=None, c_deform=None,
    c_admit_per_config=None,
    occ=None, tall_obs_mask=None,
    reconfig_check="none", rf=0.0, arc_angles=None,
    cost_limit=None,
    check_counts=None,
    height_map=None, h_payload=None,
    cable_offsets=None, cable_counts=None,
    js_admissible=None,
    prune_rot_in_free=False,
    verbose=True,
    timing=None,
):
    """A* search without pre-built graph.

    Parameters
    ----------
    dist_map       : ndarray (H, W) float64 — distance transform
    offsets_arr    : ndarray (n_cfg, n_theta, n_s, n_robots, 2) int32
    s_values       : ndarray (n_s,) float64
    periods        : ndarray (n_cfg,) int64
    formations_rad : list of ndarray (n_robots,)
    reconfig_costs : ndarray (n_cfg, n_cfg) float64
    reconfig_min   : ndarray (n_cfg, n_cfg) float64 — Floyd–Warshall
    rb             : float — robot body radius
    start, goal    : (ix, iy, iθ, is, ic) tuples
    clearance_grid : ndarray (n_iy, n_ix) float64 or None
    c_deform       : float or None
        Clearance threshold.  Where centre clearance ≤ c_deform,
        scale + reconfig edges are active.  *None* → always active.
    c_admit_per_config : sequence of float, length n_config, or None
        Per-config minimum *signed* clearance threshold.  Config
        ``ic`` is inadmissible at any cell whose signed clearance is
        below ``c_admit_per_config[ic]``; spatial moves to such cells
        and reconfigurations to such target configs are skipped.
        *None* → label-aware pruning disabled.
    occ : (H, W) bool ndarray, optional
        Occupancy mask used to build the signed clearance grid when
        ``clearance_grid`` is not supplied.  Required whenever
        ``c_admit_per_config`` is not ``None``.
    tall_obs_mask : (H, W) bool ndarray, optional
        Subset of ``occ`` treated as *tall* (forbidden for the centre
        at every scale, via a sentinel that fails every threshold).
        When omitted, every obstacle is treated as low.
    reconfig_check : str
        Collision checking for reconfiguration transitions:
        ``"none"`` (destination state only, default), ``"clearance"``
        (radius check at centre), ``"sampling"`` (arc sampling).
    rf : float
        Formation radius (pixels).  Required for ``"clearance"``
        and ``"sampling"`` modes.
    arc_angles : ndarray (n_cfg, n_cfg, n_pts) or None
        From ``_precompute_arc_angles``.  Required for ``"sampling"``.
    cost_limit : float or None
        Stop the search once the best f-value exceeds this bound.
    timing : dict or None
        When given, ``timing["t_search"]`` receives the wall-clock
        seconds of the A* kernel alone (open-list loop + expansions),
        excluding input conversion, start/goal resolution and path
        reconstruction.

    Returns
    -------
    path : list[tuple] | None — sequence of (ix, iy, iθ, is, ic)
    cost : float
    n_expanded : int
    """
    H, W = dist_map.shape
    n_ix = (W - 1) // xy_step + 1
    n_iy = (H - 1) // xy_step + 1
    n_config = len(formations_rad)
    th_step = 2.0 * math.pi / n_theta
    s_step = ((s_values[-1] - s_values[0]) / max(n_s - 1, 1)
              if n_s > 1 else 0.0)

    # Ensure correct dtypes for Numba
    dist_map_c = np.ascontiguousarray(dist_map, dtype=np.float64)
    offsets_c = np.ascontiguousarray(offsets_arr, dtype=np.int32)
    periods_c = np.ascontiguousarray(periods, dtype=np.int64)
    rc_costs_c = np.ascontiguousarray(reconfig_costs, dtype=np.float64)
    rc_min_c = np.ascontiguousarray(reconfig_min, dtype=np.float64)

    # ── Build the cell-wise clearance grid ────────────────
    # Label-aware pruning needs the *signed* clearance grid (negative
    # inside low obstacles); plain c_deform works on the unsigned one.
    if c_admit_per_config is not None and clearance_grid is None:
        if occ is None:
            raise ValueError(
                "c_admit_per_config requires a signed clearance grid; "
                "pass either ``clearance_grid`` directly or the "
                "binary ``occ`` mask so it can be built internally.")
        signed_dist = compute_signed_clearance(occ, tall_obs_mask)
        clearance_grid = _precompute_clearance_grid(signed_dist, xy_step)

    if clearance_grid is None:
        # ``c_deform_v = -1`` is the sentinel the JIT kernel reads as
        # "deformations always active".
        c_deform_v = -1.0
        clr_grid = np.zeros((n_iy, n_ix), dtype=np.float64)
    else:
        clr_grid = np.ascontiguousarray(clearance_grid, dtype=np.float64)
        c_deform_v = -1.0 if c_deform is None else float(c_deform)

    # ── Label-aware admissibility thresholds ──────────────
    if c_admit_per_config is None:
        c_admit_arr = np.zeros(n_config, dtype=np.float64)
    else:
        c_admit_arr = np.ascontiguousarray(c_admit_per_config,
                                           dtype=np.float64)
        if c_admit_arr.shape != (n_config,):
            raise ValueError(
                f"c_admit_per_config must have length {n_config}, "
                f"got {c_admit_arr.shape}")

    # ── Reconfig collision checking ───────────────────────
    _rc_modes = {"none": 0, "clearance": 1, "sampling": 2}
    rc_mode = _rc_modes.get(reconfig_check, 0)
    rf_v = float(rf)
    s_values_c = np.ascontiguousarray(s_values, dtype=np.float64)

    if rc_mode == 2 and arc_angles is not None:
        arc_c = np.ascontiguousarray(arc_angles, dtype=np.float64)
        n_arc_pts = arc_c.shape[2]
    else:
        arc_c = np.zeros((n_config, n_config, 1), dtype=np.float64)
        n_arc_pts = 0
        if rc_mode == 2:
            rc_mode = 0            # fall back if no arc data

    # ── Payload / cable height check setup ────────────────
    if height_map is not None:
        do_payload_check = 1
        hmap_c = np.ascontiguousarray(height_map, dtype=np.uint8)
        hpay_c = np.ascontiguousarray(h_payload, dtype=np.float64)
        cab_off_c = np.ascontiguousarray(cable_offsets, dtype=np.int32)
        cab_cnt_c = np.ascontiguousarray(cable_counts, dtype=np.int32)
        if js_admissible is None:
            js_adm_c = np.ones(n_s, dtype=np.bool_)
        else:
            js_adm_c = np.ascontiguousarray(js_admissible, dtype=np.bool_)
    else:
        do_payload_check = 0
        hmap_c, hpay_c, cab_off_c, cab_cnt_c, js_adm_c = \
            _dummy_payload_arrays(n_config, n_theta, n_s)

    # ── Resolve start ─────────────────────────────────────
    s0 = tuple(int(x) for x in start)
    if len(s0) == 4:
        s0 = s0 + (0,)
    ix0, iy0, it0, js0, ic0 = s0
    it0 = int(it0 % periods_c[ic0])
    s0 = (ix0, iy0, it0, js0, ic0)

    if not _check_free(ix0 * xy_step, iy0 * xy_step,
                       offsets_c, ic0, it0, js0, dist_map_c, rb):
        raise ValueError(f"Start state {s0} is blocked")
    if do_payload_check and not js_adm_c[js0]:
        raise ValueError(
            f"Start state {s0}: scale index js={js0} is physically "
            f"infeasible (rope cannot reach across the formation "
            f"circle at this scale)")

    # ── Resolve goals ─────────────────────────────────────
    goal_t = tuple(int(x) for x in goal)
    if len(goal_t) == 4:
        goal_t = goal_t + (0,)
    goal_set = _resolve_goals(goal_t, n_s, n_config,
                              periods_c.tolist(),
                              free_theta, free_s, free_config,
                              offsets_c, dist_map_c, rb, xy_step)
    if do_payload_check:
        goal_set = {g for g in goal_set if js_adm_c[g[3]]}
    if not goal_set:
        raise ValueError(f"No free goal state for {goal}")

    # Representative goal for heuristic (admissible with free-dim drops)
    gx, gy, gt, gs, gc = min(goal_set)

    cost_lim = float('inf') if cost_limit is None else float(cost_limit)

    # Per-config check counters (shared with caller when supplied).
    if check_counts is None:
        cc_arr = np.zeros(n_config, dtype=np.int64)
    else:
        cc_arr = np.ascontiguousarray(check_counts, dtype=np.int64)

    # ── Pack start + goals into int64 keys for the JIT kernel ──
    s0_key = (((ix0 * n_iy + iy0) * n_theta + it0) * n_s + js0) \
        * n_config + ic0
    goal_keys = np.array(
        [(((g[0] * n_iy + g[1]) * n_theta + g[2]) * n_s + g[3])
         * n_config + g[4] for g in goal_set],
        dtype=np.int64)

    # Heap capacity: capped at the full state count so pushes never
    # overflow; in practice the heap stays far smaller.
    n_states = n_ix * n_iy * n_theta * n_s * n_config
    heap_cap = int(n_states) + 16

    t_kernel0 = time.perf_counter()
    found, u_key, cost, n_exp, parent_d = _astar_kernel(
        np.int64(s0_key), goal_keys,
        gx, gy, gt, gs, gc,
        dist_map_c, offsets_c, xy_step,
        n_ix, n_iy, n_s, n_config, n_theta,
        periods_c, rb,
        w_move, w_rot, w_scale, w_config,
        th_step, s_step,
        rc_costs_c, rc_min_c,
        clr_grid, c_deform_v,
        c_admit_arr,
        rc_mode, arc_c, n_arc_pts, rf_v, s_values_c,
        cc_arr,
        cab_off_c, cab_cnt_c, hmap_c, hpay_c,
        js_adm_c, do_payload_check,
        free_theta, free_s, free_config,
        bool(prune_rot_in_free),
        cost_lim, heap_cap)
    if timing is not None:
        timing["t_search"] = time.perf_counter() - t_kernel0

    if check_counts is not None:
        check_counts[:] = cc_arr

    if not found:
        if verbose:
            print(f"  DA A*: no path ({n_exp:,} expanded)")
        return None, float('inf'), n_exp

    # ── Reconstruct path from the typed parent dict ───────────
    path = []
    cur = int(u_key)
    while cur != -1:
        ic = cur % n_config
        rest = cur // n_config
        js = rest % n_s
        rest //= n_s
        it = rest % n_theta
        rest //= n_theta
        iy = rest % n_iy
        ix = rest // n_iy
        path.append((int(ix), int(iy), int(it), int(js), int(ic)))
        cur = int(parent_d[cur])
    path.reverse()
    if verbose:
        print(f"  DA A*: {n_exp:,} expanded, "
              f"cost {cost:.2f}, path len {len(path)}")
    return path, float(cost), n_exp


# ═══════════════════════════════════════════════════════════
#  Convenience wrapper — from map image
# ═══════════════════════════════════════════════════════════

def find_path_da_from_map(
    map_path, start, goal, *,
    obs_thresh=128,
    rb=8.0, rf=40.0,
    formations_deg,
    xy_step=10, n_theta=72,
    s_min=0.6, s_max=1.4, n_s=20,
    w_move=1.0, w_rot=1.0, w_scale=1.0, w_config=1.0,
    c_deform=None,
    c_admit_per_config=None,
    reconfig_check="none", n_arc_samples=8,
    use_symmetry=True,
    free_theta=False, free_s=False, free_config=False,
    cost_limit=None,
    height_map_path=None, L_pole=None, L_rope=None,
    cable_sample_step_px=1.5, height_max=200,
    prune_rot_in_free=False,
    verbose=True,
    split_timing=False,
):
    """One-call DA A* from a map image file.

    Parameters
    ----------
    map_path : str — path to grayscale occupancy map image
    start, goal : (ix, iy, iθ, is, ic) tuples
    formations_deg : list of formations, each a list of clusters,
        each cluster a list of robot angles in degrees, e.g.
        ``[[[0], [90], [180], [270]], [[-10, 10], [170, 190]]]``.
        See :func:`core.formations.parse_formations`; every formation
        must be rotationally symmetric.
    c_deform : float or None
        Clearance threshold (pixels) for loose-space pruning.
        *None* → deformations everywhere.
    c_admit_per_config : sequence of float, length n_config, or None
        Per-config minimum clearance threshold (see
        :func:`find_path_da`).  *None* → disabled.
    reconfig_check : str
        ``"none"`` | ``"clearance"`` | ``"sampling"`` — see
        :func:`find_path_da`.  Sampling uses per-robot arc
        interpolation and does not preserve cluster rigidity
        mid-transition.
    n_arc_samples : int
        Intermediate samples per robot arc (``"sampling"`` only).
    cost_limit : float or None
        Stop the search once the best f-value exceeds this bound.
    split_timing : bool
        When True, also return preprocessing and search wall-clock
        seconds separately.  ``t_search`` covers the A* kernel only
        (see ``timing`` in :func:`find_path_da`).

    Returns
    -------
    path : list[tuple] | None
    cost : float
    n_expanded : int
    t_prep, t_search : float, float — only when ``split_timing=True``
    """
    t_prep0 = time.perf_counter()
    _, occ, dist_map = load_map(map_path, obs_thresh)
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
            if verbose:
                print(f"  Reconfig check: arc sampling "
                      f"({n_arc_samples} samples)")
        elif reconfig_check == "clearance" and verbose:
            print(f"  Reconfig check: clearance (s·Rf + rb)")
    else:
        rc_costs = np.zeros((1, 1))
        rc_min = np.zeros((1, 1))

    clr_grid = None
    if c_admit_per_config is not None:
        # Label-aware pruning needs a signed clearance grid.
        signed_dist = compute_signed_clearance(occ)
        clr_grid = _precompute_clearance_grid(signed_dist, xy_step)
    elif c_deform is not None:
        # Plain c_deform pruning is fine on the unsigned distance.
        clr_grid = _precompute_clearance_grid(dist_map, xy_step)

    # ── Payload / cable height check (optional) ───────────
    height_map = None
    h_payload = None
    cable_offsets = None
    cable_counts = None
    js_admissible = None
    if height_map_path is not None:
        if L_pole is None or L_rope is None:
            raise ValueError(
                "height_map_path requires both L_pole and L_rope to be "
                "specified (pole height and rope length in the same "
                "units as the height map values).")
        from .map_io import load_height_map
        from .formations import (compute_h_payload,
                                 precompute_cable_offsets)
        height_map = load_height_map(height_map_path, max_height=height_max)
        if height_map.shape != dist_map.shape:
            raise ValueError(
                f"height_map shape {height_map.shape} differs from "
                f"obstacle map shape {dist_map.shape}; both must be "
                f"the same image grid.")
        h_payload, js_admissible = compute_h_payload(
            s_values, rf, L_pole, L_rope)
        cable_offsets, cable_counts = precompute_cable_offsets(
            formations_rad, rf, n_theta, s_values, periods,
            clusters, sample_step_px=cable_sample_step_px)
        if verbose:
            n_adm = int(js_admissible.sum())
            print(f"  Payload check: ON  | L_pole={L_pole}, "
                  f"L_rope={L_rope} | js admissible {n_adm}/{n_s} | "
                  f"cable samples per scale {cable_counts.tolist()}")

    if verbose:
        tag_d = f"c_deform={c_deform}" if c_deform is not None else "full"
        tag_a = ("c_admit=" + str(list(c_admit_per_config))
                 if c_admit_per_config is not None
                 else "no-label-pruning")
        signed_tag = "signed-clearance" if c_admit_per_config is not None else "unsigned"
        print(f"  DA A* setup: {n_config} config(s), {tag_d}, "
              f"{tag_a} [{signed_tag}], sym_orders={sym_orders}")

    t_prep = time.perf_counter() - t_prep0

    timing = {}
    path, cost, n_exp = find_path_da(
        dist_map, offsets_arr, s_values, periods_arr,
        formations_rad, rc_costs, rc_min,
        rb, xy_step, n_theta, n_s,
        w_move, w_rot, w_scale, w_config,
        start, goal,
        free_theta=free_theta, free_s=free_s, free_config=free_config,
        clearance_grid=clr_grid, c_deform=c_deform,
        c_admit_per_config=c_admit_per_config,
        occ=occ,
        reconfig_check=reconfig_check, rf=rf, arc_angles=arc_angles,
        cost_limit=cost_limit,
        height_map=height_map, h_payload=h_payload,
        cable_offsets=cable_offsets, cable_counts=cable_counts,
        js_admissible=js_admissible,
        prune_rot_in_free=prune_rot_in_free,
        verbose=verbose,
        timing=timing,
    )
    t_search = timing["t_search"]

    if split_timing:
        return path, cost, n_exp, t_prep, t_search
    return path, cost, n_exp


def warmup(*, formations_deg=None, rb=8.0, rf=40.0,
           n_theta=72, n_s=20, s_min=0.6, s_max=1.4,
           xy_step=10, use_symmetry=True):
    """Trigger JIT compilation of the DA A* kernels.

    Runs one tiny real search on a synthetic free map so that
    ``_astar_kernel`` and every ``_expand`` / ``_check_free`` helper it
    calls are compiled before the timed query.  Parameters should match
    the demo's config (formation geometry, n_theta, n_s, symmetry) so
    the compiled specialisation is the one the real search reuses.
    """
    if formations_deg is None:
        formations_deg = [[[0], [90], [180], [270]]]

    s_values = np.linspace(s_min, s_max, n_s)
    formations_rad, clusters, cfg_sym_orders = parse_formations(formations_deg)
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])

    sym_orders = cfg_sym_orders if use_symmetry else [1] * n_config
    periods = [n_theta // k for k in sym_orders]
    periods_arr = np.array(periods, dtype=np.int64)

    off_dict = precompute_offsets(formations_rad, rf, n_theta,
                                  s_values, periods, clusters)
    offsets_arr = _offsets_to_array(off_dict, n_config, n_theta,
                                    n_s, n_robots)

    if n_config > 1:
        rc_costs, _ = compute_reconfig_costs(formations_rad)
        rc_min = _shortest_reconfig(rc_costs)
    else:
        rc_costs = np.zeros((1, 1))
        rc_min = np.zeros((1, 1))

    # Small all-free map (large clearance everywhere) so start/goal
    # resolve and the search runs a handful of expansions.
    margin = int(math.ceil(rf * s_max + rb)) + xy_step
    side = margin * 4
    dist_map = np.full((side, side), 1e4, dtype=np.float64)
    n_ix = (side - 1) // xy_step + 1
    n_iy = (side - 1) // xy_step + 1
    lo = margin // xy_step
    start = (lo, lo, 0, n_s // 2, 0)
    goal = (min(lo + 2, n_ix - 1), min(lo + 2, n_iy - 1), 0, n_s // 2, 0)

    find_path_da(
        dist_map, offsets_arr, s_values, periods_arr,
        formations_rad, rc_costs, rc_min,
        rb, xy_step, n_theta, n_s,
        1.0, 1.0, 1.0, 1.0,
        start, goal,
        verbose=False,
    )

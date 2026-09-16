"""Baseline planner — graph-based, after Liu et al., arXiv:2210.03340.

Reproduces the reference paper's technique for benchmarking against
DA_astar: **Mapping** (Algorithm 1) validity-checks every discretised
configuration into a dense ``C_free`` grid, then **Planning**
(Algorithm 2) materialises the entire graph G(V, E) over ``C_free``
in CSR form, gates feasibility with BFS, and runs Dijkstra (no
heuristic) on the stored graph.  Edges and costs come from the shared
:func:`core.da_astar._expand` kernel, so Dijkstra here and
``find_path_da`` minimise the same cost function.

Entry point: :func:`plan_baseline_from_map`.
"""

import math

import numpy as np
import numba

from .map_io import load_map, compute_signed_clearance
from .formations import (parse_formations, precompute_offsets,
                         compute_reconfig_costs)
from .da_astar import (_check_free, _expand, _offsets_to_array,
                       _precompute_clearance_grid, _shortest_reconfig,
                       _precompute_arc_angles, _dummy_payload_arrays,
                       _resolve_goals)


# ═══════════════════════════════════════════════════════════
#  Phase 1 — full mapping (Algorithm 1)
# ═══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def map_configurations(valid, n_ix, n_iy, n_s, n_config,
                       periods, offsets_arr, dist_map, rb, xy_step,
                       cable_offsets, cable_counts, height_map, h_payload,
                       js_admissible, do_payload_check):
    """Fill ``valid[ic, ix, iy, it, js]`` with the validity of every
    configuration over the whole map (the paper's ``C_free``).

    Only ``it < periods[ic]`` is swept; the rest of the θ axis is the
    symmetric image of that fundamental period and is left ``False``
    (the search never visits it).

    Returns the number of valid configurations found.
    """
    n_valid = 0
    for ic in range(n_config):
        period = periods[ic]
        for ix in range(n_ix):
            cx = ix * xy_step
            for iy in range(n_iy):
                cy = iy * xy_step
                for js in range(n_s):
                    if do_payload_check != 0 and not js_admissible[js]:
                        continue
                    for it in range(period):
                        if not _check_free(cx, cy, offsets_arr,
                                           ic, it, js, dist_map, rb):
                            continue
                        # Cable / payload height check (no-op when
                        # disabled).
                        ok = True
                        if do_payload_check != 0:
                            H, W = height_map.shape
                            K = cable_counts[js]
                            h_thresh = h_payload[js]
                            for k in range(K):
                                rx = cx + cable_offsets[ic, it, js, k, 0]
                                ry = cy + cable_offsets[ic, it, js, k, 1]
                                if rx < 0 or rx >= W or ry < 0 or ry >= H:
                                    ok = False
                                    break
                                if height_map[ry, rx] > h_thresh:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        valid[ic, ix, iy, it, js] = True
                        n_valid += 1
    return n_valid


# ═══════════════════════════════════════════════════════════
#  Phase 2 — Algorithm 2: materialise the whole graph, then search
# ═══════════════════════════════════════════════════════════
#
#  The entire graph G(V, E) over C_free is built and stored before any
#  search (lines 2-9), then BFS tests feasibility (line 10) and
#  Dijkstra finds the optimum (lines 11-13).  The graph is stored in
#  CSR form keyed by a compact node index (0..n_valid-1):
#  ``node_id[packed_key] -> compact index or -1``.
#
#  CSR layout:
#    edge_start : (n_valid + 1,) int64  — edge_start[v]..edge_start[v+1]
#    edge_to    : (n_edges,)     int32  — neighbour compact index
#    edge_cost  : (n_edges,)     float64

@numba.njit(cache=True)
def _pack(ix, iy, it, js, ic, n_iy, n_theta, n_s, n_config):
    k = ix
    k = k * n_iy + iy
    k = k * n_theta + it
    k = k * n_s + js
    k = k * n_config + ic
    return k


@numba.njit(cache=True)
def assign_node_ids(valid, node_id, key_of_node,
                    n_ix, n_iy, n_s, n_config, n_theta, periods):
    """Lines 2-4: give every valid configuration a compact vertex id.

    ``node_id`` (flat, size = total packed states) is filled with the
    compact index of each valid cell (``-1`` elsewhere).
    ``key_of_node`` (size n_valid) maps compact index -> packed key.
    Returns ``n_valid``.
    """
    nv = 0
    for ic in range(n_config):
        period = periods[ic]
        for ix in range(n_ix):
            for iy in range(n_iy):
                for it in range(period):
                    for js in range(n_s):
                        if valid[ic, ix, iy, it, js]:
                            key = _pack(ix, iy, it, js, ic,
                                        n_iy, n_theta, n_s, n_config)
                            node_id[key] = nv
                            key_of_node[nv] = key
                            nv += 1
    return nv


@numba.njit(cache=True)
def build_edges(node_id, key_of_node, n_valid,
                edge_start, edge_to, edge_cost, count_only,
                dist_map, offsets_arr, xy_step,
                n_ix, n_iy, n_s, n_config, n_theta,
                periods, rb,
                w_move, w_rot, w_scale, w_config,
                th_step, s_step,
                reconfig_costs, clearance_grid, c_deform,
                c_admit_per_config,
                reconfig_mode, arc_angles, n_arc_pts, rf, s_values,
                check_counts,
                cable_offsets, cable_counts, height_map, h_payload,
                js_admissible, do_payload_check):
    """Lines 5-9: enumerate ConnectionDetect edges via ``_expand``.

    Two-pass CSR build.  ``count_only=1`` only counts the degree of
    each node into ``edge_start[v+1]`` (caller then prefix-sums and
    allocates).  ``count_only=0`` fills ``edge_to`` / ``edge_cost``
    using a per-node write cursor seeded from ``edge_start``.

    Returns the total number of (directed) edges.
    """
    min_period = n_theta
    for _ic in range(n_config):
        if periods[_ic] < min_period:
            min_period = periods[_ic]
    max_nb = 12 + ((n_config - 1) * (n_theta // min_period)
                   if n_config > 1 else 0)
    out_states = np.empty((max_nb, 5), dtype=np.int64)
    out_costs = np.empty(max_nb, dtype=np.float64)

    # Write cursor (only used in fill pass).
    cursor = np.empty(n_valid, dtype=np.int64)
    if count_only == 0:
        for v in range(n_valid):
            cursor[v] = edge_start[v]

    total = 0
    for v in range(n_valid):
        key = key_of_node[v]
        tmp = key
        ic = tmp % n_config; tmp //= n_config
        js = tmp % n_s; tmp //= n_s
        it = tmp % n_theta; tmp //= n_theta
        iy = tmp % n_iy; tmp //= n_iy
        ix = tmp

        n_nb = _expand(ix, iy, it, js, ic,
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
                       False, -1, -1,
                       out_states, out_costs)
        deg = 0
        for i in range(n_nb):
            nb_key = _pack(out_states[i, 0], out_states[i, 1],
                           out_states[i, 2], out_states[i, 3],
                           out_states[i, 4], n_iy, n_theta, n_s, n_config)
            w = node_id[nb_key]
            if w < 0:
                continue   # neighbour not in C_free (shouldn't happen)
            if count_only == 0:
                pos = cursor[v]
                edge_to[pos] = w
                edge_cost[pos] = out_costs[i]
                cursor[v] = pos + 1
            deg += 1
        if count_only != 0:
            edge_start[v + 1] = deg
        total += deg
    return total


@numba.njit(cache=True)
def bfs_reachable(edge_start, edge_to, n_valid, src, goal_mask):
    """Line 10: BFS from ``src``; True iff any goal vertex is reached.

    ``goal_mask`` (size n_valid) marks goal vertices.  Returns
    ``(found, n_visited)``.
    """
    visited = np.zeros(n_valid, dtype=np.bool_)
    queue = np.empty(n_valid, dtype=np.int64)
    head = 0
    tail = 0
    queue[tail] = src; tail += 1
    visited[src] = True
    n_visited = 0
    found = False
    while head < tail:
        u = queue[head]; head += 1
        n_visited += 1
        if goal_mask[u]:
            found = True
            break
        for e in range(edge_start[u], edge_start[u + 1]):
            w = edge_to[e]
            if not visited[w]:
                visited[w] = True
                queue[tail] = w; tail += 1
    return found, n_visited


@numba.njit(cache=True)
def _dij_push(hf, hk, size, f, key):
    i = size
    hf[i] = f
    hk[i] = key
    while i > 0:
        parent = (i - 1) >> 1
        if hf[parent] <= hf[i]:
            break
        hf[i], hf[parent] = hf[parent], hf[i]
        hk[i], hk[parent] = hk[parent], hk[i]
        i = parent
    return size + 1


@numba.njit(cache=True)
def _dij_pop(hf, hk, size):
    f0 = hf[0]
    k0 = hk[0]
    size -= 1
    hf[0] = hf[size]
    hk[0] = hk[size]
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        small = i
        if left < size and hf[left] < hf[small]:
            small = left
        if right < size and hf[right] < hf[small]:
            small = right
        if small == i:
            break
        hf[i], hf[small] = hf[small], hf[i]
        hk[i], hk[small] = hk[small], hk[i]
        i = small
    return f0, k0, size


@numba.njit(cache=True)
def dijkstra_csr(edge_start, edge_to, edge_cost, n_valid,
                 src, goal_mask, heap_cap):
    """Lines 12-13: Dijkstra on the materialised CSR graph.

    Operates purely on compact node indices and the stored edge
    arrays — no ``_expand`` calls, no re-derivation of neighbours.
    Returns ``(found, goal_node, cost, n_settled, parent)`` where
    ``parent`` is an int64 array of compact-index back-pointers
    (``-1`` at the source, ``-2`` unvisited) and ``n_settled`` is the
    number of vertices popped/closed.
    """
    g_cost = np.full(n_valid, np.inf, dtype=np.float64)
    parent = np.full(n_valid, -2, dtype=np.int64)
    closed = np.zeros(n_valid, dtype=np.bool_)

    hf = np.empty(heap_cap, dtype=np.float64)
    hk = np.empty(heap_cap, dtype=np.int64)
    hsize = 0

    g_cost[src] = 0.0
    parent[src] = -1
    hsize = _dij_push(hf, hk, hsize, 0.0, src)
    n_settled = 0

    while hsize > 0:
        f_val, u, hsize = _dij_pop(hf, hk, hsize)
        if closed[u]:
            continue
        closed[u] = True
        n_settled += 1

        if goal_mask[u]:
            return 1, u, g_cost[u], n_settled, parent

        ug = g_cost[u]
        for e in range(edge_start[u], edge_start[u + 1]):
            w = edge_to[e]
            if closed[w]:
                continue
            ng = ug + edge_cost[e]
            if ng < g_cost[w]:
                g_cost[w] = ng
                parent[w] = u
                hsize = _dij_push(hf, hk, hsize, ng, w)

    return 0, np.int64(-1), np.inf, n_settled, parent


# ═══════════════════════════════════════════════════════════
#  One-call driver — mirrors find_path_da_from_map's signature
# ═══════════════════════════════════════════════════════════

def plan_baseline_from_map(
    map_path, start, goal, *,
    obs_thresh=128,
    rb=8.0, rf=40.0,
    formations_deg,
    xy_step=10, n_theta=72,
    s_min=0.6, s_max=1.4, n_s=20,
    w_move=1.0, w_rot=1.0, w_scale=1.0, w_config=1.0,
    c_deform=None,                       # ignored — see note below
    c_admit_per_config=None,             # ignored — see note below
    reconfig_check="none", n_arc_samples=8,
    use_symmetry=True,                   # ignored — see note below
    free_theta=False, free_s=False, free_config=False,
    height_map_path=None, L_pole=None, L_rope=None,
    cable_sample_step_px=1.5, height_max=200,
    return_timings=False,
    verbose=True,
):
    """Baseline planner: full mapping + uniform-cost (Dijkstra) search.

    Same signature as :func:`core.da_astar.find_path_da_from_map` so
    the two can be swapped in a benchmark — but the baseline always
    runs on the **full** configuration space: ``c_deform``,
    ``c_admit_per_config`` and ``use_symmetry`` are accepted and
    ignored (no pruning, no symmetry folding).  Costs still come from
    the shared ``_expand`` kernel, so this and a DA A* run with
    ``c_deform=None, use_symmetry=False`` return paths of identical
    cost.

    Returns ``(path, cost, n_settled)`` — plus, when
    ``return_timings=True``, a dict with the Mapping / Planning
    timing split.
    """
    import time

    # ── Baseline: no C_DEFORM pruning, no symmetry folding ──
    c_deform = None
    c_admit_per_config = None
    use_symmetry = False

    _, occ, dist_map = load_map(map_path, obs_thresh)
    s_values = np.linspace(s_min, s_max, n_s)

    formations_rad, clusters, cfg_sym = parse_formations(formations_deg)
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])
    sym_orders = [1] * n_config          # full θ axis, no folding
    periods = [n_theta] * n_config
    periods_arr = np.array(periods, dtype=np.int64)

    off_dict = precompute_offsets(formations_rad, rf, n_theta,
                                  s_values, periods, clusters)
    offsets_arr = _offsets_to_array(off_dict, n_config, n_theta,
                                    n_s, n_robots)

    arc_angles = None
    if n_config > 1:
        rc_costs, rc_assign = compute_reconfig_costs(formations_rad)
        rc_min = _shortest_reconfig(rc_costs)
        if reconfig_check == "sampling":
            arc_angles = _precompute_arc_angles(
                formations_rad, rc_assign, n_arc_samples)
    else:
        rc_costs = np.zeros((1, 1))
        rc_min = np.zeros((1, 1))

    clr_grid = None
    if c_admit_per_config is not None:
        clr_grid = _precompute_clearance_grid(
            compute_signed_clearance(occ), xy_step)
    elif c_deform is not None:
        clr_grid = _precompute_clearance_grid(dist_map, xy_step)

    # ── Payload / cable setup ─────────────────────────────
    height_map = None
    h_payload = None
    cable_offsets = None
    cable_counts = None
    js_admissible = None
    if height_map_path is not None:
        if L_pole is None or L_rope is None:
            raise ValueError("height_map_path requires L_pole and L_rope")
        from .map_io import load_height_map
        from .formations import compute_h_payload, precompute_cable_offsets
        height_map = load_height_map(height_map_path, max_height=height_max)
        h_payload, js_admissible = compute_h_payload(
            s_values, rf, L_pole, L_rope)
        cable_offsets, cable_counts = precompute_cable_offsets(
            formations_rad, rf, n_theta, s_values, periods,
            clusters, sample_step_px=cable_sample_step_px)

    # ── Dtype normalisation (matches find_path_da) ──────
    H, W = dist_map.shape
    n_ix = (W - 1) // xy_step + 1
    n_iy = (H - 1) // xy_step + 1
    th_step = 2.0 * math.pi / n_theta
    s_step = ((s_values[-1] - s_values[0]) / max(n_s - 1, 1)
              if n_s > 1 else 0.0)

    dist_c = np.ascontiguousarray(dist_map, dtype=np.float64)
    off_c = np.ascontiguousarray(offsets_arr, dtype=np.int32)
    rc_costs_c = np.ascontiguousarray(rc_costs, dtype=np.float64)
    s_values_c = np.ascontiguousarray(s_values, dtype=np.float64)
    c_admit_arr = (np.zeros(n_config, dtype=np.float64)
                   if c_admit_per_config is None
                   else np.ascontiguousarray(c_admit_per_config,
                                             dtype=np.float64))
    _rc_modes = {"none": 0, "clearance": 1, "sampling": 2}
    rc_mode = _rc_modes.get(reconfig_check, 0)
    if rc_mode == 2 and arc_angles is not None:
        arc_c = np.ascontiguousarray(arc_angles, dtype=np.float64)
        n_arc_pts = arc_c.shape[2]
    else:
        arc_c = np.zeros((n_config, n_config, 1), dtype=np.float64)
        n_arc_pts = 0
        if rc_mode == 2:
            rc_mode = 0

    if height_map is not None:
        do_payload = 1
        hmap_c = np.ascontiguousarray(height_map, dtype=np.uint8)
        hpay_c = np.ascontiguousarray(h_payload, dtype=np.float64)
        cab_off_c = np.ascontiguousarray(cable_offsets, dtype=np.int32)
        cab_cnt_c = np.ascontiguousarray(cable_counts, dtype=np.int32)
        js_adm_c = (np.ones(n_s, dtype=np.bool_) if js_admissible is None
                    else np.ascontiguousarray(js_admissible, dtype=np.bool_))
    else:
        do_payload = 0
        hmap_c, hpay_c, cab_off_c, cab_cnt_c, js_adm_c = \
            _dummy_payload_arrays(n_config, n_theta, n_s)

    if clr_grid is None:
        c_deform_v = -1.0
        clr_c = np.zeros((n_iy, n_ix), dtype=np.float64)
    else:
        clr_c = np.ascontiguousarray(clr_grid, dtype=np.float64)
        c_deform_v = -1.0 if c_deform is None else float(c_deform)

    # ── Phase 1: full mapping (Algorithm 1) ───────────────
    valid = np.zeros((n_config, n_ix, n_iy, n_theta, n_s), dtype=np.bool_)
    t0 = time.perf_counter()
    n_valid = map_configurations(
        valid, n_ix, n_iy, n_s, n_config, periods_arr,
        off_c, dist_c, rb, xy_step,
        cab_off_c, cab_cnt_c, hmap_c, hpay_c, js_adm_c, do_payload)
    map_time = time.perf_counter() - t0

    if verbose:
        print(f"  Baseline mapping: {n_valid:,} valid configs "
              f"({map_time:.3f}s)")

    # ── Resolve start / goals ─────────────────────────────
    s0 = tuple(int(x) for x in start)
    if len(s0) == 4:
        s0 = s0 + (0,)
    ix0, iy0, it0, js0, ic0 = s0
    it0 = int(it0 % periods_arr[ic0])
    if not valid[ic0, ix0, iy0, it0, js0]:
        raise ValueError(f"Start state {(ix0, iy0, it0, js0, ic0)} "
                         f"is not in C_free")

    goal_t = tuple(int(x) for x in goal)
    if len(goal_t) == 4:
        goal_t = goal_t + (0,)
    goal_set = _resolve_goals(goal_t, n_s, n_config, periods_arr.tolist(),
                              free_theta, free_s, free_config,
                              off_c, dist_c, rb, xy_step)
    if do_payload:
        goal_set = {g for g in goal_set if js_adm_c[g[3]]}
    if not goal_set:
        raise ValueError(f"No free goal state for {goal}")

    s0_key = _pack(ix0, iy0, it0, js0, ic0, n_iy, n_theta, n_s, n_config)
    goal_keys = [_pack(g[0], g[1], g[2], g[3], g[4],
                       n_iy, n_theta, n_s, n_config) for g in goal_set]

    cc = np.zeros(n_config, dtype=np.int64)
    n_states = n_ix * n_iy * n_theta * n_s * n_config

    # ── Phase 2a: assign vertex ids (Algorithm 2, lines 2-4) ──
    node_id = np.full(n_states, -1, dtype=np.int64)
    key_of_node = np.empty(n_valid, dtype=np.int64)
    assign_node_ids(valid, node_id, key_of_node,
                    n_ix, n_iy, n_s, n_config, n_theta, periods_arr)

    # ── Phase 2b: materialise edges into CSR (lines 5-9) ──────
    edge_start = np.zeros(n_valid + 1, dtype=np.int64)
    t0 = time.perf_counter()
    dummy_to = np.empty(0, dtype=np.int32)
    dummy_cost = np.empty(0, dtype=np.float64)
    # Pass 1: count degrees into edge_start[v+1].
    n_edges = build_edges(
        node_id, key_of_node, n_valid,
        edge_start, dummy_to, dummy_cost, 1,
        dist_c, off_c, xy_step,
        n_ix, n_iy, n_s, n_config, n_theta,
        periods_arr, rb,
        w_move, w_rot, w_scale, w_config, th_step, s_step,
        rc_costs_c, clr_c, c_deform_v, c_admit_arr,
        rc_mode, arc_c, n_arc_pts, float(rf), s_values_c, cc,
        cab_off_c, cab_cnt_c, hmap_c, hpay_c, js_adm_c, do_payload)
    # Prefix-sum to turn degrees into CSR offsets.
    for v in range(n_valid):
        edge_start[v + 1] += edge_start[v]
    edge_to = np.empty(n_edges, dtype=np.int32)
    edge_cost = np.empty(n_edges, dtype=np.float64)
    # Pass 2: fill.
    build_edges(
        node_id, key_of_node, n_valid,
        edge_start, edge_to, edge_cost, 0,
        dist_c, off_c, xy_step,
        n_ix, n_iy, n_s, n_config, n_theta,
        periods_arr, rb,
        w_move, w_rot, w_scale, w_config, th_step, s_step,
        rc_costs_c, clr_c, c_deform_v, c_admit_arr,
        rc_mode, arc_c, n_arc_pts, float(rf), s_values_c, cc,
        cab_off_c, cab_cnt_c, hmap_c, hpay_c, js_adm_c, do_payload)
    graph_time = time.perf_counter() - t0
    edge_mem_mb = (edge_to.nbytes + edge_cost.nbytes
                   + edge_start.nbytes) / 1e6

    if verbose:
        print(f"  Baseline graph:   {n_edges:,} edges "
              f"({edge_mem_mb:.0f} MB, {graph_time:.3f}s)")

    # Compact src / goal indices.
    src = int(node_id[s0_key])
    goal_mask = np.zeros(n_valid, dtype=np.bool_)
    for gk in goal_keys:
        gid = int(node_id[gk])
        if gid >= 0:
            goal_mask[gid] = True

    heap_cap = int(n_valid) + 16

    # ── Phase 2c: BFS feasibility gate (Algorithm 2, line 10) ─
    t0 = time.perf_counter()
    reachable, n_bfs = bfs_reachable(edge_start, edge_to, n_valid,
                                     src, goal_mask)
    bfs_time = time.perf_counter() - t0
    if verbose:
        print(f"  Baseline BFS:     {'reachable' if reachable else 'NO PATH'} "
              f"({n_bfs:,} visited, {bfs_time:.3f}s)")

    timings = dict(map_time=map_time, graph_time=graph_time,
                   bfs_time=bfs_time, search_time=0.0,
                   n_valid=int(n_valid), n_edges=int(n_edges),
                   edge_mem_mb=edge_mem_mb, n_bfs=int(n_bfs))

    if not reachable:
        # Line 16-17: BFS failed -> skip cost/Dijkstra entirely.
        return (None, float('inf'), n_bfs, timings) if return_timings \
            else (None, float('inf'), n_bfs)

    # ── Phase 2d: Dijkstra on the CSR graph (lines 12-13) ─────
    t0 = time.perf_counter()
    found, goal_node, cost, n_settled, parent_c = dijkstra_csr(
        edge_start, edge_to, edge_cost, n_valid,
        src, goal_mask, heap_cap)
    search_time = time.perf_counter() - t0
    timings['search_time'] = search_time

    if not found:
        if verbose:
            print(f"  Baseline Dijkstra: no path "
                  f"({n_settled:,} settled, {search_time:.3f}s)")
        return (None, float('inf'), n_settled, timings) if return_timings \
            else (None, float('inf'), n_settled)

    # ── Reconstruct path (compact index -> packed key -> tuple) ─
    path = []
    cur = int(goal_node)
    while cur != -1:
        key = int(key_of_node[cur])
        ic = key % n_config; rest = key // n_config
        js = rest % n_s; rest //= n_s
        it = rest % n_theta; rest //= n_theta
        iy = rest % n_iy; ix = rest // n_iy
        path.append((int(ix), int(iy), int(it), int(js), int(ic)))
        cur = int(parent_c[cur])
    path.reverse()

    if verbose:
        print(f"  Baseline Dijkstra: {n_settled:,} settled, "
              f"cost {cost:.2f}, path len {len(path)} ({search_time:.3f}s)")

    if return_timings:
        return path, float(cost), n_settled, timings
    return path, float(cost), n_settled

"""Numba-jitted A* over the implicit CCO_planner anchor graph.

Nodes are integer IDs ``(i * len_b + j) * 2 + slot_idx`` with
slot_idx ∈ {0=LEFT, 1=RIGHT} — the two slot positions the kernels
can populate for an anchor pair.  Adjacency is computed on-the-fly
inside the kernel:

- 8-neighbour same-slot (always, if both nodes valid)
- intra-anchor LEFT↔RIGHT (if the other slot is populated)

Edge weights are computed on the fly: |Δmain_L| + |Δmain_R|.

The kernel returns a 1D int64 array with the path of node IDs
(start..goal); an empty array means no path.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


SLOT_LEFT_IDX = 0
SLOT_RIGHT_IDX = 1
N_SLOTS = 2


@njit(cache=True, fastmath=True, inline='always')
def _node_id(i, j, slot_idx, len_b):
    return (i * len_b + j) * 2 + slot_idx


@njit(cache=True, fastmath=True, inline='always')
def _decode(node_id, len_b):
    slot_idx = node_id % 2
    cell = node_id // 2
    j = cell % len_b
    i = cell // len_b
    return i, j, slot_idx


@njit(cache=True, fastmath=True, inline='always')
def _edge_weight(mL_xy, mR_xy, a, b):
    dLx = mL_xy[b, 0] - mL_xy[a, 0]
    dLy = mL_xy[b, 1] - mL_xy[a, 1]
    dRx = mR_xy[b, 0] - mR_xy[a, 0]
    dRy = mR_xy[b, 1] - mR_xy[a, 1]
    return (math.sqrt(dLx * dLx + dLy * dLy)
            + math.sqrt(dRx * dRx + dRy * dRy))


@njit(cache=True, fastmath=True, inline='always')
def _heuristic(mL_xy, mR_xy, a, goal):
    dLx = mL_xy[goal, 0] - mL_xy[a, 0]
    dLy = mL_xy[goal, 1] - mL_xy[a, 1]
    dRx = mR_xy[goal, 0] - mR_xy[a, 0]
    dRy = mR_xy[goal, 1] - mR_xy[a, 1]
    return (math.sqrt(dLx * dLx + dLy * dLy)
            + math.sqrt(dRx * dRx + dRy * dRy))


# ─── Min-heap (binary) ────────────────────────────────────
# Numba doesn't support heapq, so we inline a minimal binary heap
# storing (priority, tiebreak, node_id) triples in three parallel
# arrays. Priorities are float64; tiebreaks ensure FIFO between
# equal priorities.

@njit(cache=True, fastmath=True, inline='always')
def _heap_push(prio, tie, nid, heap_prio, heap_tie, heap_nid, size):
    heap_prio[size] = prio
    heap_tie[size] = tie
    heap_nid[size] = nid
    # sift up
    k = size
    while k > 0:
        parent = (k - 1) // 2
        if (heap_prio[parent] > heap_prio[k]
                or (heap_prio[parent] == heap_prio[k]
                    and heap_tie[parent] > heap_tie[k])):
            # swap
            heap_prio[parent], heap_prio[k] = heap_prio[k], heap_prio[parent]
            heap_tie[parent], heap_tie[k] = heap_tie[k], heap_tie[parent]
            heap_nid[parent], heap_nid[k] = heap_nid[k], heap_nid[parent]
            k = parent
        else:
            break
    return size + 1


@njit(cache=True, fastmath=True, inline='always')
def _heap_pop(heap_prio, heap_tie, heap_nid, size):
    top_prio = heap_prio[0]
    top_tie = heap_tie[0]
    top_nid = heap_nid[0]
    size -= 1
    if size > 0:
        heap_prio[0] = heap_prio[size]
        heap_tie[0] = heap_tie[size]
        heap_nid[0] = heap_nid[size]
        # sift down
        k = 0
        while True:
            l = 2 * k + 1
            r = 2 * k + 2
            best = k
            if l < size:
                if (heap_prio[l] < heap_prio[best]
                        or (heap_prio[l] == heap_prio[best]
                            and heap_tie[l] < heap_tie[best])):
                    best = l
            if r < size:
                if (heap_prio[r] < heap_prio[best]
                        or (heap_prio[r] == heap_prio[best]
                            and heap_tie[r] < heap_tie[best])):
                    best = r
            if best == k:
                break
            heap_prio[k], heap_prio[best] = heap_prio[best], heap_prio[k]
            heap_tie[k], heap_tie[best] = heap_tie[best], heap_tie[k]
            heap_nid[k], heap_nid[best] = heap_nid[best], heap_nid[k]
            k = best
    return top_prio, top_tie, top_nid, size


# ─── A* ─────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def astar_small_jit(start_id, goal_id, len_a, len_b,
                    is_valid, mL_xy, mR_xy):
    """A* over the implicit LEFT/RIGHT anchor graph.

    Adjacency:
    - 8-neighbour same-slot (always, if the neighbour node is valid)
    - intra-anchor LEFT↔RIGHT (if the other slot is populated)
    """
    N = len_a * len_b * 2
    INF = 1e30
    g_score = np.full(N, INF, dtype=np.float64)
    parent = np.full(N, -1, dtype=np.int64)
    closed = np.zeros(N, dtype=np.bool_)

    cap = N + 16
    heap_prio = np.empty(cap, dtype=np.float64)
    heap_tie = np.empty(cap, dtype=np.int64)
    heap_nid = np.empty(cap, dtype=np.int64)
    heap_size = 0
    tie_counter = 0

    g_score[start_id] = 0.0
    h0 = _heuristic(mL_xy, mR_xy, start_id, goal_id)
    heap_size = _heap_push(h0, tie_counter, start_id,
                            heap_prio, heap_tie, heap_nid, heap_size)
    tie_counter += 1

    found = False
    while heap_size > 0:
        _, _, u, heap_size = _heap_pop(heap_prio, heap_tie, heap_nid,
                                         heap_size)
        if closed[u]:
            continue
        if u == goal_id:
            found = True
            break
        closed[u] = True
        i_u, j_u, slot_u = _decode(u, len_b)
        gu = g_score[u]

        # 8-neighbour spatial moves, same slot.
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni = i_u + di
                nj = j_u + dj
                if ni < 0 or ni >= len_a or nj < 0 or nj >= len_b:
                    continue
                v = _node_id(ni, nj, slot_u, len_b)
                if not is_valid[v]:
                    continue
                w = _edge_weight(mL_xy, mR_xy, u, v)
                tentative = gu + w
                if tentative < g_score[v]:
                    g_score[v] = tentative
                    parent[v] = u
                    h = _heuristic(mL_xy, mR_xy, v, goal_id)
                    heap_size = _heap_push(tentative + h,
                                            tie_counter, v,
                                            heap_prio, heap_tie,
                                            heap_nid, heap_size)
                    tie_counter += 1

        # Intra-anchor: the other slot on the same anchor pair.
        v = _node_id(i_u, j_u, 1 - slot_u, len_b)
        if is_valid[v]:
            w = _edge_weight(mL_xy, mR_xy, u, v)
            tentative = gu + w
            if tentative < g_score[v]:
                g_score[v] = tentative
                parent[v] = u
                h = _heuristic(mL_xy, mR_xy, v, goal_id)
                heap_size = _heap_push(tentative + h, tie_counter,
                                        v, heap_prio, heap_tie,
                                        heap_nid, heap_size)
                tie_counter += 1

    if not found:
        return np.empty(0, dtype=np.int64)

    path_rev = np.empty(N, dtype=np.int64)
    n = 0
    cur = goal_id
    while cur != -1:
        path_rev[n] = cur
        n += 1
        cur = parent[cur]
    path = np.empty(n, dtype=np.int64)
    for k in range(n):
        path[k] = path_rev[n - 1 - k]
    return path


# ─── Helpers ────────────────────────────────────────────────

def resolve_node_id(i, j, is_valid, len_b, search_radius=3):
    """Find a populated node at (i, j) (either slot), with fallback to
    a small neighbourhood. Returns the integer node ID."""
    for slot_idx in range(2):
        nid = (i * len_b + j) * 2 + slot_idx
        if is_valid[nid]:
            return nid
    for r in range(1, search_radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                for slot_idx in range(2):
                    nid = ((i + di) * len_b + (j + dj)) * 2 + slot_idx
                    if 0 <= nid < is_valid.size and is_valid[nid]:
                        print(f"  resolve fallback: ({i},{j}) → "
                              f"({i+di},{j+dj},slot={slot_idx})")
                        return nid
    raise RuntimeError(
        f"No valid configuration at boundary indices ({i}, {j})")


def warmup():
    """Trigger compilation of the A* kernel."""
    # Trivial 1×2 grid: cell (0,0) has both slots, cell (0,1) LEFT only.
    is_valid = np.array([True, True, True, False], dtype=np.bool_)
    mL = np.zeros((4, 2), dtype=np.float64)
    mR = np.zeros((4, 2), dtype=np.float64)
    astar_small_jit(0, 2, 1, 2, is_valid, mL, mR)

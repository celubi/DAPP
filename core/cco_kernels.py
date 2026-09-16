"""Numba-jitted kernels for the CCO_planner (Critical Crossable
Obstacle Planner).

Per anchor pair the kernel produces up to two slot positions (LEFT
and RIGHT) via a 2D nested loop on (p_L, p_R) along the anchor line,
under the scale constraint ``MIN_DIST ≤ p_R − p_L ≤ MAX_DIST``.  The
loop uses a main-only early cull on the outer index and a coarse +
refine scan (``STEP_COARSE``, then 1 px) on the inner one;
``optimise_anchors_batch`` applies the kernel to all kept anchors in
parallel with Numba ``prange``.

Output convention: ``(L_ok, L_pL, L_pR, R_ok, R_pL, R_pR)``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .cco_obstacle import _circle_free


# ─── Tunables ──────────────────────────────────────────────
# Coarse step in pixels for the 2D scan. Refinement at 1 px is done
# inside a small neighbourhood around the first coarse hit. 
STEP_COARSE = 4
REFINE_HALF_WIN = STEP_COARSE  # refine ± this many px around a coarse hit


# ─── Geometry primitives (jitted, inlined) ─────────────────

@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _cluster_clear_AB(mAx, mAy, mBx, mBy, robot_r, deg, clearance,
                      side):
    """Cluster on ``side`` clears (main + 2 satellites).

    side = 0 → cluster L (main = A). side = 1 → cluster R (main = B).
    """
    dx = mBx - mAx
    dy = mBy - mAy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        return False
    Cx = (mAx + mBx) * 0.5
    Cy = (mAy + mBy) * 0.5
    r = dist * 0.5
    ratio = deg / (2.0 * r)
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < 0.0:
        ratio = 0.0
    alpha = 2.0 * math.asin(ratio)

    if side == 0:
        ang = math.atan2(mAy - Cy, mAx - Cx)
        if not _circle_free(mAx, mAy, robot_r, clearance):
            return False
    else:
        ang = math.atan2(mBy - Cy, mBx - Cx)
        if not _circle_free(mBx, mBy, robot_r, clearance):
            return False

    cs = math.cos(alpha)
    sn = math.sin(alpha)
    ca = math.cos(ang)
    sa = math.sin(ang)
    # Rotation by ±alpha applied to unit vector (ca, sa), scaled by r.
    rx_p = r * (ca * cs - sa * sn)
    ry_p = r * (sa * cs + ca * sn)
    rx_m = r * (ca * cs + sa * sn)
    ry_m = r * (sa * cs - ca * sn)
    s1x = Cx + rx_p
    s1y = Cy + ry_p
    s2x = Cx + rx_m
    s2y = Cy + ry_m
    if not _circle_free(s1x, s1y, robot_r, clearance):
        return False
    if not _circle_free(s2x, s2y, robot_r, clearance):
        return False
    return True


@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _cable_clear(Cx, Cy, mx, my, h_thresh, sample_step_px, height_map):
    """True when no sample along the cable C→(mx, my) sits under a tall
    obstacle.

    Mirrors the DA A* payload check (``_check_payload_free`` /
    ``precompute_cable_offsets``): the cable is the segment from the
    formation centre ``C`` to a robot at ``(mx, my)``; we place
    ``M = max(1, round(r / step) − 1)`` samples along it, *excluding*
    both endpoints (the robot endpoint is covered by the
    body-clearance check; the centre pixel is shared by all cables and
    is tested once by the caller, :func:`_payload_clear_AB`).  A pixel
    whose height-map value exceeds ``h_thresh`` blocks the cable.
    """
    H, W = height_map.shape
    dx = mx - Cx
    dy = my - Cy
    r = math.sqrt(dx * dx + dy * dy)
    if r < 1e-9:
        return True
    m = int(round(r / sample_step_px)) - 1
    if m < 1:
        m = 1
    inv = 1.0 / (m + 1.0)
    for k in range(1, m + 1):
        t = k * inv
        rx = int(round(Cx + t * dx))
        ry = int(round(Cy + t * dy))
        if rx < 0 or rx >= W or ry < 0 or ry >= H:
            return False
        if height_map[ry, rx] > h_thresh:
            return False
    return True


@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _payload_clear_AB(mLx, mLy, mRx, mRy, deg,
                      L_pole, L_rope, sample_step_px, height_map):
    """Payload / cable height check for the full 6-robot formation.

    Recomputes the same 6 robot positions as ``_cluster_clear_AB``
    (two mains + four satellites) and tests every centre→robot cable
    against the height map.  The payload sits above the centre at
    height ``h = L_pole − √(L_rope² − r²)`` with ``r = |mR−mL|/2``;
    scales with ``r ≥ L_rope`` are physically infeasible and rejected,
    exactly as DA A*'s ``compute_h_payload`` marks them inadmissible.
    """
    dx = mRx - mLx
    dy = mRy - mLy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-9:
        return False
    r = dist * 0.5
    # Physically realisable scale? (r < L_rope, ropes can stay taut.)
    if r >= L_rope:
        return False
    h_thresh = L_pole - math.sqrt(L_rope * L_rope - r * r)

    Cx = (mLx + mRx) * 0.5
    Cy = (mLy + mRy) * 0.5

    # Formation centre = payload attachment point.  It is the lowest
    # point of the cable system and the point ``h_thresh`` is derived
    # from, yet ``_cable_clear`` excludes both endpoints of every
    # cable, so test it once here (it is shared by all six cables).
    # Mirrors slot 0 of ``precompute_cable_offsets``.
    Hm, Wm = height_map.shape
    pcx = int(round(Cx))
    pcy = int(round(Cy))
    if pcx < 0 or pcx >= Wm or pcy < 0 or pcy >= Hm:
        return False
    if height_map[pcy, pcx] > h_thresh:
        return False

    ratio = deg / (2.0 * r)
    if ratio > 1.0:
        ratio = 1.0
    elif ratio < 0.0:
        ratio = 0.0
    alpha = 2.0 * math.asin(ratio)
    cs = math.cos(alpha)
    sn = math.sin(alpha)

    # Both clusters: main on the L/R side, two satellites at ±alpha.
    for side in range(2):
        if side == 0:
            mx = mLx
            my = mLy
        else:
            mx = mRx
            my = mRy
        ang = math.atan2(my - Cy, mx - Cx)
        ca = math.cos(ang)
        sa = math.sin(ang)
        s1x = Cx + r * (ca * cs - sa * sn)
        s1y = Cy + r * (sa * cs + ca * sn)
        s2x = Cx + r * (ca * cs + sa * sn)
        s2y = Cy + r * (sa * cs - ca * sn)
        if not _cable_clear(Cx, Cy, mx, my, h_thresh,
                            sample_step_px, height_map):
            return False
        if not _cable_clear(Cx, Cy, s1x, s1y, h_thresh,
                            sample_step_px, height_map):
            return False
        if not _cable_clear(Cx, Cy, s2x, s2y, h_thresh,
                            sample_step_px, height_map):
            return False
    return True


@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _exact_check_v2(p_L, p_R, l_ix, l_iy, ux, uy, robot_r, deg,
                    clearance,
                    L_pole, L_rope, sample_step_px, height_map,
                    do_payload_check):
    """6-robot exact check at (p_L, p_R).

    Body-clearance for both clusters, plus (when
    ``do_payload_check != 0``) the payload / cable height check that
    mirrors the DA A* planner.
    """
    mLx = l_ix + p_L * ux
    mLy = l_iy + p_L * uy
    mRx = l_ix + p_R * ux
    mRy = l_iy + p_R * uy
    if not _cluster_clear_AB(mLx, mLy, mRx, mRy, robot_r, deg,
                             clearance, 0):
        return False
    if not _cluster_clear_AB(mLx, mLy, mRx, mRy, robot_r, deg,
                             clearance, 1):
        return False
    if do_payload_check != 0:
        if not _payload_clear_AB(mLx, mLy, mRx, mRy, deg,
                                 L_pole, L_rope, sample_step_px,
                                 height_map):
            return False
    return True


@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _main_clear(mx, my, robot_r, clearance):
    """Check the main robot alone — cheapest possible cull."""
    return _circle_free(mx, my, robot_r, clearance)


# ─── Inner search: scan p_R for a fixed p_L (RIGHT-slot) ───

@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _scan_pR_for_fixed_pL(p_L, l_ix, l_iy, ux, uy,
                           p_R_lo, p_R_hi,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
    """Find the smallest p_R in [p_R_lo, p_R_hi] (integer grid) that
    passes ``_exact_check_v2(p_L, p_R, …)``, via coarse + refine.

    Returns p_R as float, or -1.0 if nothing clears. The p_L position
    is fixed (and the L-cluster main is pre-validated by the caller).
    """
    # NOT_FOUND sentinel must not collide with a valid (possibly negative)
    # p_R; see the matching note in _scan_pL_for_fixed_pR.
    NOT_FOUND = -1.0e18

    if p_R_lo > p_R_hi:
        return NOT_FOUND

    p_R_lo_i = int(math.ceil(p_R_lo))
    p_R_hi_i = int(math.floor(p_R_hi))
    if p_R_lo_i > p_R_hi_i:
        return NOT_FOUND

    # Coarse scan: STEP_COARSE pixel steps.
    found = False
    coarse_hit = 0
    p_R = p_R_lo_i
    while p_R <= p_R_hi_i:
        if _exact_check_v2(p_L, float(p_R), l_ix, l_iy, ux, uy,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
            coarse_hit = p_R
            found = True
            break
        p_R += STEP_COARSE

    if not found:
        return NOT_FOUND

    # Refine: scan backward 1 px from coarse_hit toward p_R_lo_i, up
    # to REFINE_HALF_WIN steps. Returns the SMALLEST p_R in
    # [coarse_hit - REFINE_HALF_WIN, coarse_hit] that clears.
    best = coarse_hit
    lo_refine = coarse_hit - REFINE_HALF_WIN
    if lo_refine < p_R_lo_i:
        lo_refine = p_R_lo_i
    p_R = coarse_hit - 1
    while p_R >= lo_refine:
        if _exact_check_v2(p_L, float(p_R), l_ix, l_iy, ux, uy,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
            best = p_R
            p_R -= 1
        else:
            break
    return float(best)


# ─── Inner search: scan p_L for a fixed p_R (LEFT-slot) ────

@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _scan_pL_for_fixed_pR(p_R, l_ix, l_iy, ux, uy,
                           p_L_lo, p_L_hi,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
    """Symmetric: find the LARGEST p_L in [p_L_lo, p_L_hi] (integer
    grid) that clears.

    The 'largest p_L' is what the LEFT slot wants (L-main as close to
    the obstacle as possible, R-main far away). Coarse + refine.
    """
    # NOT_FOUND sentinel: p_L is negative for many LEFT-slot configs (main L
    # sits behind l_i, on the far side of the anchor), so neither -1 nor any
    # "< 0" test can flag "not found" without colliding with a legitimate
    # negative p_L.  Use a value no valid p_L can ever reach.
    NOT_FOUND = -1.0e18

    if p_L_lo > p_L_hi:
        return NOT_FOUND

    p_L_lo_i = int(math.ceil(p_L_lo))
    p_L_hi_i = int(math.floor(p_L_hi))
    if p_L_lo_i > p_L_hi_i:
        return NOT_FOUND

    found = False
    coarse_hit = 0
    # Scan DOWNWARD from p_L_hi (largest p_L = closest to obstacle).
    p_L = p_L_hi_i
    while p_L >= p_L_lo_i:
        if _exact_check_v2(float(p_L), p_R, l_ix, l_iy, ux, uy,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
            coarse_hit = p_L
            found = True
            break
        p_L -= STEP_COARSE

    if not found:
        return NOT_FOUND

    # Refine: scan UP from coarse_hit + 1, up to REFINE_HALF_WIN
    # steps, looking for the LARGEST p_L that still clears.
    best = coarse_hit
    hi_refine = coarse_hit + REFINE_HALF_WIN
    if hi_refine > p_L_hi_i:
        hi_refine = p_L_hi_i
    p_L = coarse_hit + 1
    while p_L <= hi_refine:
        if _exact_check_v2(float(p_L), p_R, l_ix, l_iy, ux, uy,
                           robot_r, deg, clearance,
                           L_pole, L_rope, sample_step_px, height_map,
                           do_payload_check):
            best = p_L
            p_L += 1
        else:
            break
    return float(best)


# ─── Anchor-line geometry ──────────────────────────────────

@njit(cache=True, fastmath=True, inline='always', boundscheck=False)
def _anchor_line(l_ix, l_iy, r_jx, r_jy):
    """(ux, uy, d_ij). u_hat unit vector L→R."""
    dx = r_jx - l_ix
    dy = r_jy - l_iy
    d = math.sqrt(dx * dx + dy * dy)
    if d < 1e-9:
        return 1.0, 0.0, 0.0
    return dx / d, dy / d, d


# ─── Per-anchor kernel v2 ──────────────────────────────────

@njit(cache=True, fastmath=True, boundscheck=False)
def optimise_v2_kernel(l_ix, l_iy, r_jx, r_jy,
                       MIN_DIST, MAX_DIST, R_infl,
                       robot_r, deg, clearance,
                       L_pole, L_rope, cable_sample_step_px, height_map,
                       do_payload_check):
    """v2 per-anchor: produces up to 2 slots (LEFT, RIGHT).

    Returns (L_ok, L_pL, L_pR, R_ok, R_pL, R_pR) with *_ok ∈ {0, 1}.

    Algorithm (4-b: nested loop with smart ordering, coarse+refine,
    main-only early cull):

      RIGHT slot — wants L far from obstacle, R close to obstacle:
        for p_L = 0, 1, …, R_infl (smallest p_L first = farthest):
          if main_L collides → skip the entire row
          scan p_R upward over [r', r*] clipped to scale constraint;
          first feasible (p_L, p_R) wins.

      LEFT slot — symmetric: outer loop on p_R from d_ij downward to
        d_ij - R_infl; inner loop p_L downward over [l*, l'].
    """
    L_ok, L_pL, L_pR = 0.0, 0.0, 0.0
    R_ok, R_pL, R_pR = 0.0, 0.0, 0.0

    ux, uy, d_ij = _anchor_line(l_ix, l_iy, r_jx, r_jy)
    if d_ij < 1e-9:
        return (L_ok, L_pL, L_pR, R_ok, R_pL, R_pR)
    if d_ij > MAX_DIST + 2.0 * R_infl:
        return (L_ok, L_pL, L_pR, R_ok, R_pL, R_pR)

    # ── RIGHT slot ──
    p_L_lo = 0.0
    p_L_hi = R_infl
    # r' = d_ij - R_infl, r* = R_infl + MAX_DIST.
    p_R_dom_lo = d_ij - R_infl
    p_R_dom_hi = R_infl + MAX_DIST

    p_L_int = int(math.floor(p_L_lo))
    p_L_hi_int = int(math.floor(p_L_hi))
    while p_L_int <= p_L_hi_int:
        p_L_f = float(p_L_int)
        # Main-only early cull on L.
        mLx = l_ix + p_L_f * ux
        mLy = l_iy + p_L_f * uy
        if _main_clear(mLx, mLy, robot_r, clearance):
            # Clip p_R domain by scale constraint.
            pr_lo = p_R_dom_lo
            if p_L_f + MIN_DIST > pr_lo:
                pr_lo = p_L_f + MIN_DIST
            pr_hi = p_R_dom_hi
            if p_L_f + MAX_DIST < pr_hi:
                pr_hi = p_L_f + MAX_DIST
            if pr_lo <= pr_hi:
                p_R_found = _scan_pR_for_fixed_pL(
                    p_L_f, l_ix, l_iy, ux, uy, pr_lo, pr_hi,
                    robot_r, deg, clearance,
                    L_pole, L_rope, cable_sample_step_px, height_map,
                    do_payload_check)
                if p_R_found > -1.0e17:
                    R_ok = 1.0
                    R_pL = p_L_f
                    R_pR = p_R_found
                    break
        p_L_int += 1

    # ── LEFT slot ──
    p_R_lo = d_ij - R_infl
    p_R_hi = d_ij
    # l* = d_ij - R_infl - MAX_DIST, l' = R_infl.
    p_L_dom_lo = d_ij - R_infl - MAX_DIST
    p_L_dom_hi = R_infl

    p_R_int = int(math.floor(p_R_hi))
    p_R_lo_int = int(math.ceil(p_R_lo))
    while p_R_int >= p_R_lo_int:
        p_R_f = float(p_R_int)
        # Main-only early cull on R.
        mRx = l_ix + p_R_f * ux
        mRy = l_iy + p_R_f * uy
        if _main_clear(mRx, mRy, robot_r, clearance):
            pl_lo = p_L_dom_lo
            if p_R_f - MAX_DIST > pl_lo:
                pl_lo = p_R_f - MAX_DIST
            pl_hi = p_L_dom_hi
            if p_R_f - MIN_DIST < pl_hi:
                pl_hi = p_R_f - MIN_DIST
            if pl_lo <= pl_hi:
                p_L_found = _scan_pL_for_fixed_pR(
                    p_R_f, l_ix, l_iy, ux, uy, pl_lo, pl_hi,
                    robot_r, deg, clearance,
                    L_pole, L_rope, cable_sample_step_px, height_map,
                    do_payload_check)
                # p_L_found can be legitimately negative; compare against the
                # NOT_FOUND sentinel, not 0.
                if p_L_found > -1.0e17:
                    L_ok = 1.0
                    L_pL = p_L_found
                    L_pR = p_R_f
                    break
        p_R_int -= 1

    return (L_ok, L_pL, L_pR, R_ok, R_pL, R_pR)


# ─── Parallel batch wrapper ────────────────────────────────

@njit(cache=True, fastmath=True, parallel=True, boundscheck=False)
def optimise_anchors_batch(ii, jj, array_a, array_b,
                            MIN_DIST, MAX_DIST, R_infl,
                            robot_r, deg, clearance,
                            L_pole, L_rope, cable_sample_step_px,
                            height_map, do_payload_check):
    """Apply ``optimise_v2_kernel`` to every (i, j) pair in (ii, jj).

    Inputs:
      ii, jj    : int32 arrays, indices of anchors to process
      array_a   : float32 (N_A, 2)  L-chain coordinates
      array_b   : float32 (N_B, 2)  R-chain coordinates
      clearance : float32 (H, W) distance-transform map
      MIN_DIST, MAX_DIST, R_infl, robot_r, deg : scalars (float64)
      L_pole, L_rope, cable_sample_step_px : scalars (float64) — payload
        pole height, rope length, and cable sampling step (px).  Mirror
        the DA A* payload model; only consulted when
        ``do_payload_check != 0``.
      height_map : uint8 (H, W) — obstacle height per pixel (0 = free),
        same grid as ``clearance``.  Pass a dummy 1×1 array when the
        check is disabled.
      do_payload_check : int — 0 disables the payload / cable height
        check; non-zero enables it.

    Output: out (M, 6) float32 where M = len(ii). Columns:
      [L_ok, L_pL, L_pR, R_ok, R_pL, R_pR]

    Anchors are independent and processed with Numba ``prange``.
    """
    M = ii.shape[0]
    out = np.zeros((M, 6), dtype=np.float64)
    for k in prange(M):
        i = ii[k]
        j = jj[k]
        l_ix = array_a[i, 0]
        l_iy = array_a[i, 1]
        r_jx = array_b[j, 0]
        r_jy = array_b[j, 1]
        L_ok, L_pL, L_pR, R_ok, R_pL, R_pR = optimise_v2_kernel(
            l_ix, l_iy, r_jx, r_jy,
            MIN_DIST, MAX_DIST, R_infl,
            robot_r, deg, clearance,
            L_pole, L_rope, cable_sample_step_px, height_map,
            do_payload_check)
        out[k, 0] = L_ok
        out[k, 1] = L_pL
        out[k, 2] = L_pR
        out[k, 3] = R_ok
        out[k, 4] = R_pL
        out[k, 5] = R_pR
    return out


# ─── Batch output → dense A* arrays (jitted) ───────────────
# Slot indices match cco_astar: 0=LEFT, 1=RIGHT.
_SLOT_LEFT_IDX = 0
_SLOT_RIGHT_IDX = 1


@njit(cache=True, boundscheck=False)
def batch_to_arrays(ii, jj, out, array_a, array_b, len_a, len_b):
    """Reshape the (M, 6) batch output straight into the dense arrays
    consumed by ``astar_small_jit`` — one allocation-light JIT pass
    with no Python dict intermediate.

    Each kept anchor pair k maps to node block ``(i*len_b + j)*2``; the
    LEFT slot (slot 0) is filled from columns [1,2] when ``out[k,0]``
    flags it, the RIGHT slot (slot 1) from columns [4,5] when
    ``out[k,3]`` flags it.  Slot positions are reconstructed in absolute
    pixels along the anchor-line unit vector:  ``m = l + p · û``.

    Returns ``(is_valid, mL_xy, mR_xy)``.
    """
    N = len_a * len_b * 2
    is_valid = np.zeros(N, dtype=np.bool_)
    mL_xy = np.zeros((N, 2), dtype=np.float64)
    mR_xy = np.zeros((N, 2), dtype=np.float64)

    M = ii.shape[0]
    for k in range(M):
        L_ok = out[k, 0] > 0.5
        R_ok = out[k, 3] > 0.5
        if not (L_ok or R_ok):
            continue
        i = ii[k]
        j = jj[k]
        lx = array_a[i, 0]
        ly = array_a[i, 1]
        rx = array_b[j, 0]
        ry = array_b[j, 1]
        dx = rx - lx
        dy = ry - ly
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-9:
            continue
        ux = dx / d
        uy = dy / d
        base = (i * len_b + j) * 2

        if L_ok:
            pL = out[k, 1]
            pR = out[k, 2]
            nid = base + _SLOT_LEFT_IDX
            is_valid[nid] = True
            mL_xy[nid, 0] = lx + pL * ux
            mL_xy[nid, 1] = ly + pL * uy
            mR_xy[nid, 0] = lx + pR * ux
            mR_xy[nid, 1] = ly + pR * uy
        if R_ok:
            pL = out[k, 4]
            pR = out[k, 5]
            nid = base + _SLOT_RIGHT_IDX
            is_valid[nid] = True
            mL_xy[nid, 0] = lx + pL * ux
            mL_xy[nid, 1] = ly + pL * uy
            mR_xy[nid, 0] = lx + pR * ux
            mR_xy[nid, 1] = ly + pR * uy

    return is_valid, mL_xy, mR_xy


# ─── Warmup ────────────────────────────────────────────────

def warmup():
    """Force AOT compilation of all kernels by calling them once.

    First call compiles ~1–3 s; subsequent runs hit the cache."""
    dummy_clearance = np.full((16, 16), 1e6, dtype=np.float32)
    # Free everywhere → payload check, when enabled, never blocks during
    # warmup. Compile both the disabled (do_payload=0) and enabled
    # (do_payload=1) specialisations so neither path pays at runtime.
    dummy_height = np.zeros((16, 16), dtype=np.uint8)
    for do_payload in (0, 1):
        optimise_v2_kernel(0.0, 0.0, 300.0, 0.0,
                           240.0, 360.0, 50.0, 15.0, 35.0,
                           dummy_clearance,
                           200.0, 180.0, 20.0, dummy_height, do_payload)
    # Build a tiny batch to force the parallel wrapper compile.
    ii = np.array([0], dtype=np.int64)
    jj = np.array([0], dtype=np.int64)
    aa = np.array([[0.0, 0.0]], dtype=np.float64)
    bb = np.array([[300.0, 0.0]], dtype=np.float64)
    out = None
    for do_payload in (0, 1):
        out = optimise_anchors_batch(ii, jj, aa, bb,
                                     240.0, 360.0, 50.0, 15.0, 35.0,
                                     dummy_clearance,
                                     200.0, 180.0, 20.0, dummy_height,
                                     do_payload)
    # Compile the reshape kernel too.
    batch_to_arrays(ii, jj, out, aa, bb, 1, 1)

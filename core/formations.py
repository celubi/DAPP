"""Formation geometry — parsing, symmetry, reconfiguration, offsets.

The cluster-based formation schema
-----------------------------------
A *formation* is a list of *clusters*; each cluster is a list of robot
angles (degrees) on the formation circle.  The :func:`parse_formations`
validator enforces the invariants that make a formation rotationally
symmetric: equal cluster size, evenly-spaced cluster slot angles, and
identical per-cluster local geometry.  Under those invariants the
rotational symmetry order equals the number of clusters.

Example input (the top-level list holds several alternative formations
of the same robot team)::

    [
        [[0], [90], [180], [270]],         # 4 singletons, sym=4
        [[-10, 10], [170, 190]],           # 2 pairs of 2 robots, sym=2
    ]
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


# ═══════════════════════════════════════════════════════════
#  Parsing / symmetry
# ═══════════════════════════════════════════════════════════

def parse_formations(formations_clusters_deg, tol_deg=1e-6):
    """Validate and normalise the cluster-based formation schema.

    Returns
    -------
    formations_rad : list of ndarray (n_robots,) float64
        Flat base angles per formation (radians), in the order the
        clusters (and robots within clusters) were given.
    clusters : list of list[list[int]]
        Per-formation cluster partition of robot indices.
    sym_orders : list of int
        Rotational symmetry order of each formation (= n_clusters).
    """
    if formations_clusters_deg is None or len(formations_clusters_deg) == 0:
        raise ValueError("formations_clusters_deg must be a non-empty list")

    tol_rad = math.radians(tol_deg)
    TWO_PI = 2.0 * math.pi

    formations_rad = []
    clusters = []
    sym_orders = []

    n_robots_ref = None

    for ic, formation in enumerate(formations_clusters_deg):
        if not formation or any(len(grp) == 0 for grp in formation):
            raise ValueError(
                f"formation {ic}: must be a non-empty list of non-empty "
                f"clusters")

        n_clusters = len(formation)
        m = len(formation[0])
        if any(len(grp) != m for grp in formation):
            raise ValueError(
                f"formation {ic}: all clusters must have the same size; "
                f"got sizes {[len(g) for g in formation]}")

        n_robots = n_clusters * m
        if n_robots_ref is None:
            n_robots_ref = n_robots
        elif n_robots != n_robots_ref:
            raise ValueError(
                f"formation {ic}: has {n_robots} robots; formation 0 has "
                f"{n_robots_ref}. Every formation must describe the same "
                f"team size.")

        flat = np.array(
            [math.radians(a) % TWO_PI for grp in formation for a in grp],
            dtype=np.float64,
        )

        cluster_def = []
        idx = 0
        for _ in range(n_clusters):
            cluster_def.append(list(range(idx, idx + m)))
            idx += m

        slots = np.zeros(n_clusters, dtype=np.float64)
        locals_sorted = []
        for j, grp in enumerate(cluster_def):
            angles = flat[grp]
            cx = np.cos(angles).sum()
            cy = np.sin(angles).sum()
            if cx * cx + cy * cy < 1e-12:
                raise ValueError(
                    f"formation {ic}, cluster {j}: slot angle is undefined "
                    f"(robots diametrically opposite). Clusters must group "
                    f"robots on the same side of the centre.")
            slots[j] = math.atan2(cy, cx) % TWO_PI
            local = ((angles - slots[j] + math.pi) % TWO_PI) - math.pi
            locals_sorted.append(np.sort(local))

        slots_sorted = np.sort(slots)
        expected_gap = TWO_PI / n_clusters
        gaps = np.diff(np.concatenate([slots_sorted, [slots_sorted[0] + TWO_PI]]))
        if not np.allclose(gaps, expected_gap, atol=tol_rad):
            raise ValueError(
                f"formation {ic}: cluster slot angles are not evenly "
                f"spaced. Got slots (deg) "
                f"{np.rad2deg(slots_sorted).round(3).tolist()}; "
                f"expected gap {math.degrees(expected_gap):.3f}°, "
                f"got gaps {np.rad2deg(gaps).round(3).tolist()}.")

        ref = locals_sorted[0]
        for j, loc in enumerate(locals_sorted[1:], start=1):
            if not np.allclose(loc, ref, atol=tol_rad):
                raise ValueError(
                    f"formation {ic}, cluster {j}: internal geometry "
                    f"differs from cluster 0. Each cluster must have the "
                    f"same local angle pattern relative to its slot "
                    f"(got {np.rad2deg(loc).round(3).tolist()} vs "
                    f"{np.rad2deg(ref).round(3).tolist()}).")

        formations_rad.append(flat)
        clusters.append(cluster_def)
        sym_orders.append(n_clusters)

    return formations_rad, clusters, sym_orders


def symmetry_order(angles_rad, tol=1e-9):
    """Largest k such that rotating every angle by 2π/k permutes the set."""
    n = len(angles_rad)
    s = np.sort(np.asarray(angles_rad, dtype=float) % (2 * np.pi))
    for k in range(n, 0, -1):
        rot = np.sort((s + 2 * np.pi / k) % (2 * np.pi))
        if np.allclose(rot, s, atol=tol):
            return k
    return 1


# ═══════════════════════════════════════════════════════════
#  Reconfiguration
# ═══════════════════════════════════════════════════════════

def _bottleneck_assignment(C):
    """Min-makespan (bottleneck) assignment: minimize ``max_i C[i, σ(i)]``.

    Binary-searches the bottleneck value over the sorted distinct cost
    entries; for each candidate threshold ``t``, feasibility of a
    perfect matching using only edges ``≤ t`` is tested via a
    sum-assignment on a masked matrix.  The smallest feasible ``t`` is
    the min-makespan.  Complexity ``O(n³ log n)``.

    Returns ``(bottleneck_cost, col)`` where ``col`` is the permutation
    mapping source robot index → destination slot index.
    """
    n = C.shape[0]
    vals = np.unique(C)
    INF = C.max() + 1.0
    lo, hi = 0, len(vals) - 1
    best_cost = vals[-1]
    # Identity is always a valid fallback permutation.
    best_col = np.arange(n)
    while lo <= hi:
        mid = (lo + hi) // 2
        t = vals[mid]
        masked = np.where(C <= t, C, INF)
        row, col = linear_sum_assignment(masked)
        if masked[row, col].max() <= t + 1e-12:   # perfect matching ≤ t
            best_cost, best_col = t, col
            hi = mid - 1
        else:
            lo = mid + 1
    return float(best_cost), best_col


def compute_reconfig_costs(formations_rad):
    """Min-makespan robot reassignment for every config pair.

    Returns (costs, assignments) where ``costs`` is (n_cfg, n_cfg) and
    ``assignments[(src, dst)]`` is the permutation array mapping source
    robot index → destination robot index.

    Units: ``costs[i, j]`` is the **angular** travel in *radians* of
    the single robot that must move the most under the
    makespan-optimal reassignment (see :func:`_bottleneck_assignment`).
    It is radius-independent: the search kernel scales it to pixels of
    real arc travel by multiplying by ``rf · s`` at the current scale,
    so it must NOT be pre-multiplied by ``rf`` here.
    """
    n_cfg = len(formations_rad)
    n_robots = len(formations_rad[0])
    costs = np.zeros((n_cfg, n_cfg))
    assignments = {}

    for i in range(n_cfg):
        for j in range(n_cfg):
            if i == j:
                assignments[(i, j)] = np.arange(n_robots, dtype=np.int32)
                continue
            C = np.empty((n_robots, n_robots))
            for a in range(n_robots):
                for b in range(n_robots):
                    d = abs(formations_rad[i][a] - formations_rad[j][b]) % (2 * np.pi)
                    C[a, b] = min(d, 2 * np.pi - d)
            cost, col = _bottleneck_assignment(C)
            costs[i, j] = cost
            assignments[(i, j)] = col.astype(np.int32)

    return costs, assignments


# ═══════════════════════════════════════════════════════════
#  Robot offset precomputation
# ═══════════════════════════════════════════════════════════

def precompute_offsets(formations_rad, rf, n_theta, s_values, periods,
                       clusters):
    """Robot-position offsets for every (config, θ, scale) combination.

    Every robot lies on the formation circle of radius ``rf·s`` at all
    scales.  Within a cluster, the chord between any two robots is
    invariant in pixels — the ``s=1`` chords are preserved by adjusting
    each robot's angle on the (now larger/smaller) circle so that its
    distance to the cluster slot direction stays constant.

    For robot ``k`` belonging to a cluster with slot ``sa``:
        chord_k = 2 · rf · |sin((base_angle_k − sa) / 2)|     (at s=1)
        side_k  = sign(base_angle_k − sa wrapped to [-π, π])
        δ_k(s)  = side_k · 2 · asin(chord_k / (2 · rf · s))
        position(s, θ) = rf·s · (cos(sa + θ + δ_k), sin(sa + θ + δ_k))

    When ``rf·s < chord_k / 2`` the chord is geometrically infeasible;
    δ_k is clamped to ±π/2 (antipodal).

    Returns
    -------
    dict[(ic, iθ, is)] → ndarray (n_robots, 2) int
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TWO_PI = 2.0 * math.pi

    off = {}
    for ic, base_angles in enumerate(formations_rad):
        base_angles = np.asarray(base_angles, dtype=float)
        n_robots = base_angles.size
        cluster_def = clusters[ic]

        slot_angle = np.zeros(n_robots, dtype=float)
        chord = np.zeros(n_robots, dtype=float)
        side = np.zeros(n_robots, dtype=float)
        for grp in cluster_def:
            grp_arr = np.asarray(grp, dtype=int)
            angles = base_angles[grp_arr]
            sa = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
            for k in grp_arr:
                slot_angle[k] = sa
                d_ang = ((base_angles[k] - sa + math.pi) % TWO_PI) - math.pi
                chord[k] = 2.0 * rf * abs(math.sin(d_ang / 2.0))
                side[k] = 1.0 if d_ang >= 0.0 else -1.0
                if chord[k] > 2.0 * rf + 1e-9:
                    raise ValueError(
                        f"formation {ic}, robot {k}: chord {chord[k]:.3f} "
                        f"exceeds diameter 2·rf = {2.0 * rf:.3f}; robot "
                        f"cannot lie on the formation circle at s=1.")

        for js, s in enumerate(s_values):
            r = rf * s
            ratio = chord / (2.0 * r)
            ratio_clamped = np.clip(ratio, -1.0, 1.0)
            delta = side * 2.0 * np.arcsin(ratio_clamped)

            for it in range(periods[ic]):
                th = thetas[it]
                ang = slot_angle + th + delta
                vx = r * np.cos(ang)
                vy = r * np.sin(ang)
                off[(ic, it, js)] = np.round(
                    np.column_stack([vx, vy])).astype(int)

    return off


# ═══════════════════════════════════════════════════════════
#  Payload / cable geometry — for height-map collision check
# ═══════════════════════════════════════════════════════════

def compute_h_payload(s_values, rf, L_pole, L_rope):
    """Payload height (and admissibility mask) at every scale.

    Robots stand on poles of height ``L_pole`` and hold ropes of
    length ``L_rope`` that converge to the payload above the formation
    centre.  The robots lie on a circle of radius ``r = rf · s``; the
    rope from one robot to the payload is therefore the hypotenuse of
    a right triangle with legs ``r`` (horizontal) and ``L_pole − h``
    (vertical), giving ``h = L_pole − √(L_rope² − r²)``.

    Scales with ``r ≥ L_rope`` are physically infeasible (the ropes
    cannot reach across the circle without lifting the payload above
    the pole tops or going slack).

    Parameters
    ----------
    s_values : (n_s,) array-like — scale values
    rf : float — formation radius at s=1, pixels (height units assumed
        commensurate with map cm for the in-plane comparison)
    L_pole : float — pole height
    L_rope : float — rope length (taut)

    Returns
    -------
    h_payload : (n_s,) float64
        Payload height at each scale.  Inadmissible scales receive
        ``-inf`` so they fail any downstream height comparison.
    js_admissible : (n_s,) bool
        True where the scale is physically realisable.
    """
    s_values = np.asarray(s_values, dtype=np.float64)
    r = rf * s_values
    admissible = r < L_rope
    h = np.full(s_values.shape, -np.inf, dtype=np.float64)
    r_ok = r[admissible]
    h[admissible] = L_pole - np.sqrt(L_rope * L_rope - r_ok * r_ok)
    return h, admissible


def precompute_cable_offsets(formations_rad, rf, n_theta, s_values,
                             periods, clusters, sample_step_px=1.5):
    """Pixel offsets of sample points along the N robot-to-centre cables.

    Mirrors :func:`precompute_offsets` but samples the *cables* instead
    of the robot positions.  Each cable is the segment from the
    formation centre to a robot; samples are placed along it at
    spacing ``sample_step_px``, excluding both endpoints (the robot
    endpoint is already covered by the standard robot collision check).

    Slot ``0`` is the formation centre itself, offset ``(0, 0)``.  The
    centre is where the payload hangs: it is the *lowest* point of the
    whole cable system and the very point the height threshold
    ``h_payload[js]`` is derived from, so it has to be tested.  It is
    stored once instead of once per cable because every cable shares
    it.

    The number of samples per cable depends on scale::

        M(js) = max(1, round(rf · s_values[js] / sample_step_px) − 1)

    The array is padded to ``M_max`` along the sample axis; valid
    sample counts are returned in ``counts``.

    Returns
    -------
    offsets_arr : (n_config, n_theta, n_s, 1 + n_robots * M_max, 2) int32
        Integer pixel offsets of cable sample points, indexed in the
        same (ic, it, js) layout used by the robot offsets.  Slot 0 is
        the formation centre; slots ``1 .. counts[js] - 1`` are the
        cable samples.  Slots beyond ``counts[js]`` are unused (left
        as zeros).
    counts : (n_s,) int32
        Total valid sample count ``1 + n_robots * M(js)`` for each
        scale.
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TWO_PI = 2.0 * math.pi

    s_values = np.asarray(s_values, dtype=np.float64)
    n_s = s_values.size
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])

    m_per_cable = np.maximum(
        1, np.round(rf * s_values / sample_step_px).astype(int) - 1)
    m_max = int(m_per_cable.max())
    # +1 for the shared centre sample stored in slot 0.
    counts = (n_robots * m_per_cable + 1).astype(np.int32)

    offsets_arr = np.zeros(
        (n_config, n_theta, n_s, n_robots * m_max + 1, 2), dtype=np.int32)
    # Slot 0 is the centre offset (0, 0) — already zero from np.zeros.

    for ic, base_angles in enumerate(formations_rad):
        base_angles = np.asarray(base_angles, dtype=float)
        cluster_def = clusters[ic]

        slot_angle = np.zeros(n_robots, dtype=float)
        chord = np.zeros(n_robots, dtype=float)
        side = np.zeros(n_robots, dtype=float)
        for grp in cluster_def:
            grp_arr = np.asarray(grp, dtype=int)
            angles = base_angles[grp_arr]
            sa = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
            for k in grp_arr:
                slot_angle[k] = sa
                d_ang = ((base_angles[k] - sa + math.pi) % TWO_PI) - math.pi
                chord[k] = 2.0 * rf * abs(math.sin(d_ang / 2.0))
                side[k] = 1.0 if d_ang >= 0.0 else -1.0

        for js, s in enumerate(s_values):
            r = rf * s
            ratio_clamped = np.clip(chord / (2.0 * r), -1.0, 1.0)
            delta = side * 2.0 * np.arcsin(ratio_clamped)
            m = int(m_per_cable[js])
            ts = (np.arange(1, m + 1, dtype=np.float64) / (m + 1)) * r

            for it in range(periods[ic]):
                th = thetas[it]
                ang = slot_angle + th + delta
                cos_a = np.cos(ang)
                sin_a = np.sin(ang)
                vx = ts[:, None] * cos_a[None, :]
                vy = ts[:, None] * sin_a[None, :]
                pts = np.round(np.stack(
                    [vx.reshape(-1), vy.reshape(-1)], axis=1)).astype(np.int32)
                offsets_arr[ic, it, js, 1:1 + pts.shape[0], :] = pts

    # Fold the θ-axis through symmetry so every it ∈ [0, n_theta) holds
    # a valid sample pattern (callers such as the dynamic planner index
    # by an unfolded θ).
    for ic in range(n_config):
        per = int(periods[ic])
        if per >= n_theta:
            continue
        for it in range(per, n_theta):
            offsets_arr[ic, it] = offsets_arr[ic, it % per]

    return offsets_arr, counts

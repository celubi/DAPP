#!/usr/bin/env python3
"""Benchmark — CCO planner vs DA A* (pruning) on a single map.

Both planners solve the same physical obstacle.  DA A* runs on
``MAP_PATH``, a variant with a tight enclosing border that shrinks its
free space to the minimum, over the full ``XY_STEPS`` × ``N_THETAS`` ×
``N_SS`` cartesian product; the CCO planner runs on the un-bordered
``MAP_PATH_ANCHOR`` / ``WALL_PATH_ANCHOR`` across the boundary step
sizes in ``ANCHOR_STEP_SIZES``.

The reported time is the **search time only** (``split_timing=True``):
for DA A* it is the A* kernel alone (open list + expansions) — map
loading, precomputation, start/goal resolution and path reconstruction
are excluded.  Each row also reports
``d_cluster`` (travel of the two cluster mains, summed) and ``d_c``
(travel of the formation centre).  Tables list only the successful
runs, sorted fastest first:

* **TABLE 1** — DA A* over the parameter sweep.
* **TABLE 2** — the CCO planner over the step sizes.
* **TABLE 3** — the fastest run of each planner side by side.  The
  comparison that matters is time; path lengths are informational.

Uses ``config.bilateral_only`` (a single bilateral formation, sym
order 2).  The test only measures and logs — no pass/fail on cost.

Outputs (one figure per planner, ``comparison_grid.png``, path cache)
are written to ``test_output/``.

Usage (from the repo root)::

    python -m test.test_cco_vs_da_astar
"""

import argparse
import math
import pickle

import cv2
import matplotlib
matplotlib.use("Agg")          # write PNGs, never open a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pathlib import Path

from core.da_astar import find_path_da_from_map
from core.cco_obstacle import get_cluster_positions
from core.cco_planner import (
    prepare_cco_scene, build_cco_chains, find_path_cco,
)
from core.cco_kernels import warmup as warmup_cco_kernels
from core.cco_astar import warmup as warmup_cco_astar
from config import bilateral_only as cfg
from config import cco_planner as acfg
from test._output import out_path


# ── Maps ─────────────────────────────────────────────────────────────────
# One physical obstacle in three flavours: the enclosed map for DA A*,
# the bare obstacle map for the CCO planner, and the wall map the CCO
# planner checks clearance against.
_ROOT = Path(__file__).resolve().parent.parent

MAP_PATH = str(_ROOT / "maps" / "wall_narrow_flip_close.png")
MAP_PATH_ANCHOR = str(_ROOT / "maps" / "mp_narrow_flip.png")
WALL_PATH_ANCHOR = str(_ROOT / "maps" / "wall_narrow_flip_open.png")

# ── Plot output ──────────────────────────────────────────────────────────
# One figure per planner (fastest successful run), plus the combined 1×2
# grid — one panel per planner — with both panels cropped to a same-size
# window (see ``plot_grid``).  Everything lands in the shared
# test_output/ folder.
PLOT_DA_PNG = out_path("da_astar_fastest_path.png")
PLOT_CCO_PNG = out_path("cco_fastest_path.png")
PLOT_GRID_PNG = out_path("comparison_grid.png")

# The fastest paths of a completed run are cached here so the grid figure
# can be regenerated (``--plot-only``) without redoing the whole sweep.
PATHS_CACHE = out_path("comparison_paths.pkl")

# Payload / cable height check (test-level knob; the input map doubles
# as the height map when enabled).
PAYLOAD_CHECK = False

# Start / goal in PIXELS, as (x, y, theta_idx, scale_VALUE, config_idx).
# ⚠ The 4th field is a scale VALUE, NOT a js index: it is resolved to
# the nearest valid index per row (see _to_grid), so every row searches
# the same physical scale and no index goes out of range.
START_PX = (618, 2160, 0, 15, 0)
GOAL_PX = (804, 672, 0, 15, 0)

S_MIN, S_MAX = 0.8, 1.2
FREE_THETA = False
FREE_S = False
FREE_CONFIG = False

RECONFIG_CHECK = "sampling"

# ── The CCO planner sweep ────────────────────────────────────────────
# Boundary-sampling step sizes (px): a smaller step samples the
# inflated boundary more densely → more anchor pairs.
ANCHOR_STEP_SIZES = [4, 6, 8, 10]

# ── Configuration matrix (DA A* side, full cartesian product) ────────
XY_STEPS = [10, 9, 8, 7, 6]
N_THETAS = [144, 72, 36]
N_SS = [40, 30, 20]

# One row per (label, xy_step, n_theta, n_s).
CONFIGS = [
    (f"{xy}/{nt}/{ns}", xy, nt, ns)
    for xy in XY_STEPS
    for nt in N_THETAS
    for ns in N_SS
]


def _scale_value_to_index(s_value, n_s):
    """Nearest valid js index for a physical scale, clamped to [0, n_s-1].

    The s axis is ``np.linspace(S_MIN, S_MAX, n_s)``, matching the
    planner.
    """
    if n_s <= 1:
        return 0
    s_step = (S_MAX - S_MIN) / (n_s - 1)
    js = round((s_value - S_MIN) / s_step)
    return int(max(0, min(n_s - 1, js)))


def _to_grid(px, xy_step, n_s):
    """Pixel start/goal → grid-cell units for the given xy_step / n_s.

    px[3] is a physical scale VALUE; it is resolved to the nearest valid
    js index for this row's n_s (see _scale_value_to_index).
    """
    js = _scale_value_to_index(px[3], n_s)
    return (px[0] / xy_step, px[1] / xy_step, px[2], js, px[4])


def _path_to_AB(path, xy_step, n_theta, n_s):
    """DA A* path → ``[(m_L, m_R), …]`` — the two cluster mains per node.

    A state ``(ix, iy, iθ, is, ic)`` places the formation centre at
    ``(ix·xy_step, iy·xy_step)`` with the two cluster mains
    diametrically opposite along θ at radius ``rf·s``.
    """
    s_values = np.linspace(S_MIN, S_MAX, n_s)
    th_unit = 2.0 * math.pi / n_theta
    out = []
    for ix, iy, it, jsi, _ic in path:
        x, y = ix * xy_step, iy * xy_step
        th = it * th_unit
        r = cfg.RF * s_values[jsi]
        dx, dy = r * math.cos(th), r * math.sin(th)
        out.append((np.array([x - dx, y - dy]), np.array([x + dx, y + dy])))
    return out


def _cluster_distances(path_AB):
    """Distance walked by (both clusters together, centre) along the path.

    * ``d_cluster`` — Σ(‖Δm_L‖ + ‖Δm_R‖), travel of the two cluster
      mains summed.
    * ``d_c`` — Σ‖Δc‖, travel of the formation centre.

    Each step is charged the cheaper of the two label pairings: the
    mains are interchangeable, so a θ-wrap that relabels the ends must
    not be charged as a diameter-long jump.
    """
    d_cluster = d_c = 0.0
    for (aL, aR), (bL, bR) in zip(path_AB[:-1], path_AB[1:]):
        direct = (math.hypot(bL[0] - aL[0], bL[1] - aL[1])
                  + math.hypot(bR[0] - aR[0], bR[1] - aR[1]))
        swapped = (math.hypot(bR[0] - aL[0], bR[1] - aL[1])
                   + math.hypot(bL[0] - aR[0], bL[1] - aR[1]))
        d_cluster += min(direct, swapped)
        acx, acy = 0.5 * (aL[0] + aR[0]), 0.5 * (aL[1] + aR[1])
        bcx, bcy = 0.5 * (bL[0] + bR[0]), 0.5 * (bL[1] + bR[1])
        d_c += math.hypot(bcx - acx, bcy - acy)
    return d_cluster, d_c


def _run(xy_step, n_theta, n_s, map_path, start_px, goal_px):
    """One DA A* solve with symmetry pruning at the given resolution.

    Returns ``(path, cost, n_exp, t_search, d_cluster, d_c)``.  ``t_search``
    is the wall-clock time of the A* kernel alone (open list +
    expansions), obtained via ``split_timing``: map load, offset /
    clearance / payload precomputation, start/goal resolution and path
    reconstruction are all excluded.
    ``d_cluster`` is the summed travel of both cluster mains and ``d_c``
    that of the formation centre (both ``None`` when no path is found).
    """
    start_g = _to_grid(start_px, xy_step, n_s)
    goal_g = _to_grid(goal_px, xy_step, n_s)

    # c_deform threshold is sized on the circumscribed formation radius; it
    # must be recomputed per-row because it depends on xy_step.
    c_deform = (cfg.RF * S_MAX) + cfg.RB + xy_step

    kw = dict(
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=xy_step, n_theta=n_theta,
        s_min=S_MIN, s_max=S_MAX, n_s=n_s,
        w_move=cfg.W_MOVE, w_rot=cfg.W_ROT,
        w_scale=cfg.W_SCALE, w_config=cfg.W_CONFIG,
        c_deform=c_deform,
        reconfig_check=RECONFIG_CHECK,
        n_arc_samples=cfg.N_ARC_SAMPLES,
        use_symmetry=True,
        free_theta=FREE_THETA,
        free_s=FREE_S,
        free_config=FREE_CONFIG,
        verbose=False,
        split_timing=True,
    )
    # Height check with the height map = the input map itself (see
    # PAYLOAD_CHECK).  Left off entirely when the knob is disabled.
    if PAYLOAD_CHECK:
        kw.update(height_map_path=map_path,
                  L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
                  cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
                  height_max=cfg.HEIGHT_MAX)

    path, cost, n_exp, t_prep, t_search = find_path_da_from_map(
        map_path, start_g, goal_g, **kw)

    if path is None:
        return path, cost, n_exp, t_search, None, None

    d_cluster, d_c = _cluster_distances(
        _path_to_AB(path, xy_step, n_theta, n_s))
    return path, cost, n_exp, t_search, d_cluster, d_c


# ═══════════════════════════════════════════════════════════
#  CCO planner runner
# ═══════════════════════════════════════════════════════════

def _run_anchor(step_size, scene, obstacle_map):
    """One CCO planner solve at the given boundary step size.

    Thin wrapper over :func:`core.cco_planner.find_path_cco`; the
    chains are rebuilt here (untimed) and the wall clearance / height
    maps come prebuilt in ``scene``.

    Returns ``(path_AB, t_search, d_cluster, d_c, n_valid)``.
    ``path_AB`` is ``None`` on failure.
    """
    array_a, array_b, _ = build_cco_chains(obstacle_map, acfg.R_INFL,
                                           step_size)
    path_AB, res = find_path_cco(
        array_a, array_b, scene,
        bar=acfg.BAR, tol_frac=acfg.TOL_FRAC, r_infl=acfg.R_INFL,
        robot_r=acfg.ROBOT_R, clearance_margin=acfg.CLEARANCE_MARGIN,
        intra_robot_dist=acfg.INTRA_ROBOT_DIST,
        L_pole=acfg.L_POLE, L_rope=acfg.L_ROPE,
        cable_sample_step_px=acfg.CABLE_SAMPLE_STEP_PX)
    if path_AB is None:
        return None, res.t_search, None, None, res.n_valid
    d_cluster, d_c = _cluster_distances(path_AB)
    return path_AB, res.t_search, d_cluster, d_c, res.n_valid


# ═══════════════════════════════════════════════════════════
#  Plotting — the path of each planner's fastest run
# ═══════════════════════════════════════════════════════════

def _draw_formation(ax, mL, mR, color, alpha=0.9, lw=1.0):
    """Draw the 6 robot bodies + the formation circle for one pose."""
    cl_a, cl_b = get_cluster_positions(mL, mR, acfg.INTRA_ROBOT_DIST)
    cx, cy = 0.5 * (mL[0] + mR[0]), 0.5 * (mL[1] + mR[1])
    r = 0.5 * math.hypot(mR[0] - mL[0], mR[1] - mL[1])
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=color,
                            linestyle="--", linewidth=lw, alpha=0.45))
    for p in list(cl_a) + list(cl_b):
        ax.add_patch(plt.Circle((p[0], p[1]), acfg.ROBOT_R, color=color,
                                alpha=alpha, zorder=6))


def _consistent_cluster_tracks(path_AB):
    """Un-swap the ``(m_L, m_R)`` labels so each track follows one cluster.

    A θ-wrap flips the reconstructed labels, and plotting them verbatim
    would draw a spurious jump across the diameter; the cheaper pairing
    is carried forward along the path instead.

    Returns ``(track_a, track_b)`` as ``(N, 2)`` arrays.
    """
    a, b = [np.asarray(path_AB[0][0], dtype=float)], \
           [np.asarray(path_AB[0][1], dtype=float)]
    for L, R in path_AB[1:]:
        L = np.asarray(L, dtype=float)
        R = np.asarray(R, dtype=float)
        pa, pb = a[-1], b[-1]
        direct = math.hypot(L[0] - pa[0], L[1] - pa[1]) \
            + math.hypot(R[0] - pb[0], R[1] - pb[1])
        swapped = math.hypot(R[0] - pa[0], R[1] - pa[1]) \
            + math.hypot(L[0] - pb[0], L[1] - pb[1])
        if swapped < direct:
            L, R = R, L
        a.append(L)
        b.append(R)
    return np.array(a), np.array(b)


def _draw_run(ax, path_AB, color, ms_marker=12, labels=False):
    """Tracks + endpoint formations + start/goal markers for one run.

    Three tracks are drawn: the formation centre and the two cluster
    mains, the latter as thinner dashed lines so the centre stays the
    dominant curve.  The formation itself (6 robot bodies + the circle the
    two cluster mains sit on) is drawn only at the start and the goal, so
    the figure stays readable.
    """
    ctr = np.array([[0.5 * (L[0] + R[0]), 0.5 * (L[1] + R[1])]
                    for L, R in path_AB])
    track_a, track_b = _consistent_cluster_tracks(path_AB)

    lab = (lambda t: t) if labels else (lambda t: None)
    ax.plot(track_a[:, 0], track_a[:, 1], "--", color="darkorange", lw=1.2,
            alpha=0.9, label=lab("cluster paths"))
    ax.plot(track_b[:, 0], track_b[:, 1], "--", color="darkorange", lw=1.2,
            alpha=0.9)
    ax.plot(ctr[:, 0], ctr[:, 1], "-", color=color, lw=2.0,
            label=lab("formation centre path"))

    _draw_formation(ax, path_AB[0][0], path_AB[0][1], color)
    _draw_formation(ax, path_AB[-1][0], path_AB[-1][1], color)
    ax.plot(ctr[0, 0], ctr[0, 1], "o", color="lime", ms=ms_marker, zorder=10,
            label=lab("start"))
    ax.plot(ctr[-1, 0], ctr[-1, 1], "o", color="red", ms=ms_marker,
            zorder=10, label=lab("goal"))


def plot_path(path_AB, map_path, title, out_png, color):
    """Draw a formation path over its (full, uncropped) map."""
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read map for plotting: {map_path}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    _draw_run(ax, path_AB, color, labels=True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def _run_bbox(path_AB):
    """(xmin, ymin, xmax, ymax) of everything a run draws.

    Covers the centre track and both cluster tracks, padded by the largest
    formation extent (circle radius at s_max + robot body) so the endpoint
    formations fit too.
    """
    track_a, track_b = _consistent_cluster_tracks(path_AB)
    pts = np.vstack([track_a, track_b])
    pad = cfg.RF * S_MAX + cfg.RB + 30.0
    return (pts[:, 0].min() - pad, pts[:, 1].min() - pad,
            pts[:, 0].max() + pad, pts[:, 1].max() + pad)


# Display names used in the figures.
PLANNER_DISPLAY = ("deformation-aware A*", "CCO planner")


def plot_grid(result, out_png):
    """1×2 comparison figure: one panel per planner.

    Both panels are cropped to the SAME window so they are directly
    comparable: the crop is the union of the two paths' bounding boxes,
    clamped to fit inside both maps, so the two panels show the identical
    region of the two map variants.
    """
    boxes = [_run_bbox(result["da_path"]), _run_bbox(result["anchor_path"])]
    u = (min(b[0] for b in boxes), min(b[1] for b in boxes),
         max(b[2] for b in boxes), max(b[3] for b in boxes))
    crop_w = u[2] - u[0]
    crop_h = u[3] - u[1]

    # The crop must fit inside both maps involved.
    shapes = {}
    for key in ("da_map", "anchor_wall"):
        p = result[key]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read map: {p}")
        shapes[p] = img.shape
    crop_w = min(crop_w, min(s[1] for s in shapes.values()))
    crop_h = min(crop_h, min(s[0] for s in shapes.values()))

    def _window(lo, hi, size, limit):
        """[lo, hi] centre → clamped [x0, x0+size] within [0, limit]."""
        x0 = 0.5 * (lo + hi) - 0.5 * size
        return max(0.0, min(x0, limit - size))

    aspect = crop_h / crop_w
    panel_w = 3.4
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * panel_w + 1.0, panel_w * aspect + 1.6),
        constrained_layout=True)

    panels = (
        (result["da_map"], result["da_path"], "mediumseagreen"),
        (result["anchor_wall"], result["anchor_path"], "deepskyblue"),
    )
    for col, (map_path, path_AB, color) in enumerate(panels):
        ax = axes[col]
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        x0 = _window(u[0], u[2], crop_w, W)
        y0 = _window(u[1], u[3], crop_h, H)
        ax.imshow(img, cmap="gray", origin="upper")
        _draw_run(ax, path_AB, color, ms_marker=8)
        ax.set_xlim(x0, x0 + crop_w)
        ax.set_ylim(y0 + crop_h, y0)          # y down, image coords
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PLANNER_DISPLAY[col], fontsize=14)

    # Figure-level legend with neutral proxies (the centre-path colour
    # differs per planner, so no single data handle could represent it);
    # "outside" keeps constrained_layout from drawing panels beneath it.
    handles = [
        Line2D([], [], ls="--", color="darkorange", lw=2.0),
        Line2D([], [], ls="-", color="0.25", lw=2.6),
        Line2D([], [], ls="none", marker="o", color="lime", ms=11),
        Line2D([], [], ls="none", marker="o", color="red", ms=11),
    ]
    fig.legend(handles,
               ["cluster paths", "formation centre path", "start", "goal"],
               loc="outside lower center", ncol=4, fontsize=13,
               frameon=False)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def run_benchmark():
    """The full benchmark (both planners + tables) on the configured map.

    Returns ``dict(da_map, anchor_wall, da_path, anchor_path)`` for the
    plots, or ``None`` when either planner had no successful run.
    """
    start_px, goal_px = START_PX, GOAL_PX

    print("\nDA A* (symmetry pruning) — configuration matrix")
    print(f"  Map:   {MAP_PATH}")
    print(f"  Start: {start_px}")
    print(f"  Goal:  {goal_px}")
    print(f"  Formations: {len(cfg.FORMATIONS_DEG)} configurations")
    print(f"  reconfig_check: '{RECONFIG_CHECK}'  |  symmetry pruning: ON")
    print(f"  payload check: {'ON (height map = input map)' if PAYLOAD_CHECK else 'OFF'}")
    print(f"  Sweep: xy_step={XY_STEPS} × n_theta={N_THETAS} × n_s={N_SS}"
          f"  →  {len(CONFIGS)} combinations")

    # One record per row (success OR failure) in CONFIGS order.  Failures
    # are kept here so they can be counted, but the summary table lists
    # only the successful ones — with a large sweep the failures would
    # otherwise bury the results.
    rows = []
    for i, (label, xy_step, n_theta, n_s) in enumerate(CONFIGS, 1):
        print(f"\n▸ [{i}/{len(CONFIGS)}] xy={xy_step}, n_θ={n_theta}, "
              f"n_s={n_s} …")
        # First call compiles, not timed (JIT specialisation is per
        # parameter triple); the second is the measured run.
        _run(xy_step, n_theta, n_s, MAP_PATH, start_px, goal_px)
        path, cost, n_exp, t_search, d_cluster, d_c = _run(
            xy_step, n_theta, n_s, MAP_PATH, start_px, goal_px)
        ok = path is not None
        if ok:
            print(f"  cost={cost:.2f}  steps={len(path)}  "
                  f"expanded={n_exp:,}  search={t_search:.3f}s  "
                  f"d_cluster={d_cluster:.0f}px  d_c={d_c:.0f}px")
        else:
            print(f"  ✗ no path found  (search={t_search:.3f}s)")
        rows.append(dict(label=label, xy=xy_step, n_theta=n_theta, n_s=n_s,
                         ok=ok, cost=cost,
                         steps=(len(path) if ok else None),
                         n_exp=n_exp, t_search=t_search,
                         d_cluster=d_cluster, d_c=d_c,
                         # Kept so the fastest run can be plotted later.
                         path_AB=(_path_to_AB(path, xy_step, n_theta, n_s)
                                  if ok else None)))

    # ── TABLE 1 — DA A* (search time net of preprocessing) ──────────
    # Successful runs only: the sweep is large and the failures carry no
    # path metrics, so listing them would only add noise.
    da_ok = [r for r in rows if r["ok"]]
    n_fail = len(rows) - len(da_ok)

    print("\n══ TABLE 1 — DA A* (symmetry) — search time, net of "
          "preprocessing (successful runs only) ══")
    hdr = (f"  {'xy':>4} {'n_θ':>5} {'n_s':>4} "
           f"{'cost':>10} {'steps':>6} {'expanded':>12} {'search(s)':>10} "
           f"{'d_cluster(px)':>14} {'d_c(px)':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(da_ok, key=lambda r: r["t_search"]):
        print(f"  {r['xy']:>4} {r['n_theta']:>5} {r['n_s']:>4} "
              f"{r['cost']:>10.2f} {r['steps']:>6} {r['n_exp']:>12,} "
              f"{r['t_search']:>10.3f} {r['d_cluster']:>14.0f} "
              f"{r['d_c']:>9.0f}")
    if not da_ok:
        print("  (none — every combination failed to find a path)")

    print("\n  Reading guide:")
    print(f"    • {len(da_ok)}/{len(rows)} combinations found a path; "
          f"{n_fail} failed and {'is' if n_fail == 1 else 'are'} omitted "
          f"from the table.")
    print("    • rows are sorted by search time (fastest first).")
    print("    • search(s) is the A* kernel only (open list + expansions) — "
          "map loading, precomputation, start/goal resolution and path "
          "reconstruction are excluded.")
    print("    • d_cluster: pixels walked by the two cluster mains "
          "summed; d_c: by the formation centre.")

    # ══════════════════════════════════════════════════════════════════
    #  TABLE 2 — the CCO planner on the same obstacle (no enclosing border)
    # ══════════════════════════════════════════════════════════════════
    print("\n\n══════════════════════════════════════════════════════════")
    print("  CCO planner — same obstacle, no enclosing border")
    print(f"    obstacle map: {MAP_PATH_ANCHOR}")
    print(f"    wall map:     {WALL_PATH_ANCHOR}")
    print(f"    payload check: {'ON (height map = wall map)' if PAYLOAD_CHECK else 'OFF'}")
    print(f"    step sizes:   {ANCHOR_STEP_SIZES}")
    print("══════════════════════════════════════════════════════════")

    # Preprocessing that does not depend on step size (clearance + height
    # map) is built once; the obstacle inflation / chain split (which do
    # depend on step size) happen inside _run_anchor, not timed.
    # Height map = the wall map itself when the payload check is on,
    # mirroring the DA side and config.cco_planner.
    scene = prepare_cco_scene(
        WALL_PATH_ANCHOR,
        height_map_path=(WALL_PATH_ANCHOR if PAYLOAD_CHECK else None),
        height_max=acfg.HEIGHT_MAX)

    anchor_rows = []
    for step in ANCHOR_STEP_SIZES:
        print(f"\n▸ the CCO planner  (step_size={step}) …")
        path_AB, t_search, d_cluster, d_c, n_valid = _run_anchor(
            step, scene, MAP_PATH_ANCHOR)
        ok = path_AB is not None
        if ok:
            print(f"  steps={len(path_AB)}  valid_nodes={n_valid:,}  "
                  f"search={t_search:.3f}s  d_cluster={d_cluster:.0f}px  "
                  f"d_c={d_c:.0f}px")
        else:
            print(f"  ✗ no path found  (search={t_search:.3f}s)")
        anchor_rows.append(dict(step=step, ok=ok,
                                steps=(len(path_AB) if ok else None),
                                n_valid=n_valid, t_search=t_search,
                                d_cluster=d_cluster, d_c=d_c,
                                # Kept so the fastest run can be plotted.
                                path_AB=path_AB))

    anchor_ok = [r for r in anchor_rows if r["ok"]]

    print("\n══ TABLE 2 — the CCO planner — pathfinding time, net of "
          "preprocessing (successful runs only) ══")
    hdr2 = (f"  {'step_size':>9} {'steps':>6} {'valid_nodes':>12} "
            f"{'search(s)':>10} {'d_cluster(px)':>14} {'d_c(px)':>9}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for r in sorted(anchor_ok, key=lambda r: r["t_search"]):
        print(f"  {r['step']:>9} {r['steps']:>6} {r['n_valid']:>12,} "
              f"{r['t_search']:>10.3f} {r['d_cluster']:>14.0f} "
              f"{r['d_c']:>9.0f}")
    if not anchor_ok:
        print("  (none — every step size failed to find a path)")

    n_fail_a = len(anchor_rows) - len(anchor_ok)
    print("\n  Reading guide:")
    print(f"    • {len(anchor_ok)}/{len(anchor_rows)} step sizes found a "
          f"path; {n_fail_a} failed and are omitted from the table.")
    print("    • search(s) is the the CCO planner 'find path' phase only "
          "(prefilter + kernel + reshape + mini A*);")
    print("      building the inflated obstacle and the L/R chains is "
          "preprocessing and is excluded.")

    # ══════════════════════════════════════════════════════════════════
    #  TABLE 3 — best-of-each comparison
    #  (da_ok / anchor_ok were built alongside TABLE 1 / TABLE 2)
    # ══════════════════════════════════════════════════════════════════
    print("\n\n══ TABLE 3 — fastest run of each planner ══")
    if not da_ok or not anchor_ok:
        missing = []
        if not da_ok:
            missing.append("DA A*")
        if not anchor_ok:
            missing.append("the CCO planner")
        print(f"  Cannot compare: no successful run for {', '.join(missing)}.")
        return None

    best_da = min(da_ok, key=lambda r: r["t_search"])
    best_anchor = min(anchor_ok, key=lambda r: r["t_search"])

    da_params = (f"xy={best_da['xy']}, n_θ={best_da['n_theta']}, "
                   f"n_s={best_da['n_s']}")
    anchor_params = f"step_size={best_anchor['step']}"

    hdr3 = (f"  {'planner':<14} {'fastest params':<34} {'search(s)':>10} "
            f"{'d_cluster(px)':>14} {'d_c(px)':>9}")
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))
    print(f"  {'DA A*':<14} {da_params:<34} "
          f"{best_da['t_search']:>10.3f} {best_da['d_cluster']:>14.0f} "
          f"{best_da['d_c']:>9.0f}")
    print(f"  {'CCO planner':<14} {anchor_params:<34} "
          f"{best_anchor['t_search']:>10.3f} "
          f"{best_anchor['d_cluster']:>14.0f} {best_anchor['d_c']:>9.0f}")

    # The real comparison is on TIME.
    faster, slower = ((best_anchor, best_da)
                      if best_anchor["t_search"] < best_da["t_search"]
                      else (best_da, best_anchor))
    speedup = slower["t_search"] / faster["t_search"] if faster["t_search"] else float("inf")
    faster_name = ("CCO planner" if faster is best_anchor else "DA A*")

    # Path-length similarity (informational — no winner).
    def _pct_diff(a, b):
        m = 0.5 * (a + b)
        return abs(a - b) / m * 100.0 if m else 0.0

    print("\n  ── Time (the actual comparison) ──")
    print(f"    {faster_name} is faster: {faster['t_search']:.3f}s vs "
          f"{slower['t_search']:.3f}s  →  {speedup:.2f}× speedup.")
    print("\n  ── Path length (informational — no winner) ──")
    print("    The two planners work the same obstacle in different ways, "
          "so paths need only be *similar*:")
    print(f"    cluster travel (L+R): DA {best_da['d_cluster']:.0f} px  "
          f"vs  CCO {best_anchor['d_cluster']:.0f} px  "
          f"(Δ {_pct_diff(best_da['d_cluster'], best_anchor['d_cluster']):.1f}%)")
    print(f"    centre travel:        DA {best_da['d_c']:.0f} px  vs  "
          f"CCO {best_anchor['d_c']:.0f} px  "
          f"(Δ {_pct_diff(best_da['d_c'], best_anchor['d_c']):.1f}%)")

    # ── Plots — the path of each planner's fastest run ────────────────
    # Each planner is drawn over the map it actually planned on: DA A*
    # on the enclosed map, the CCO planner on the un-bordered wall map.
    print("\n  ── Plots (fastest run of each planner) ──")
    plot_path(best_da["path_AB"], MAP_PATH,
              f"{PLANNER_DISPLAY[0]} — fastest run: {da_params}  "
              f"({best_da['t_search']:.3f}s)",
              PLOT_DA_PNG, "mediumseagreen")
    plot_path(best_anchor["path_AB"], WALL_PATH_ANCHOR,
              f"{PLANNER_DISPLAY[1]} — fastest run: {anchor_params}  "
              f"({best_anchor['t_search']:.3f}s)",
              PLOT_CCO_PNG, "deepskyblue")

    return dict(da_map=MAP_PATH, anchor_wall=WALL_PATH_ANCHOR,
                da_path=best_da["path_AB"],
                anchor_path=best_anchor["path_AB"])


def main(plot_only=False):
    if plot_only:
        # Regenerate the combined grid from the cached fastest paths of a
        # previous full run — no sweep, instant.
        with open(PATHS_CACHE, "rb") as f:
            result = pickle.load(f)
        plot_grid(result, PLOT_GRID_PNG)
        return

    # Warm up the CCO planner JIT kernels once (compile, not timed).
    warmup_cco_kernels()
    warmup_cco_astar()

    result = run_benchmark()

    # ── Combined 1×2 figure: one panel per planner ────────────────────
    if result is not None:
        with open(PATHS_CACHE, "wb") as f:
            pickle.dump(result, f)
        print(f"\n  cached fastest paths → {PATHS_CACHE}  "
              f"(reuse with --plot-only)")
        print("\n  ── Combined comparison grid ──")
        plot_grid(result, PLOT_GRID_PNG)
    else:
        print("\n  (skipping the combined grid — no successful pair of runs)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the benchmark and rebuild comparison_grid.png "
                         f"from the cached paths in {PATHS_CACHE}")
    main(plot_only=ap.parse_args().plot_only)

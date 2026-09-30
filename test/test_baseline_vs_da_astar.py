#!/usr/bin/env python3
"""Benchmark — baseline planner vs DA_astar (full and pruned) on a single map.

The map, discretisation (``XY_STEP`` / ``N_THETA`` / ``S_MIN`` /
``S_MAX`` / ``N_S``), goal-relaxation flags and ``C_DEFORM`` are set in
the parameter block below and OVERRIDE ``config/da_astar``; everything
else (formations, cost weights, robot geometry, cable/height params)
still comes from that config.

Three planners on the same problem:

    baseline     = full mapping + uniform-cost Dijkstra, FULL space
                   (no symmetry folding, no c_deform) — the reference
                   technique of Liu et al., arXiv:2210.03340
    DA-full      = DA_astar, FULL space (use_symmetry=False,
                   c_deform=None) — explores the same graph as the
                   baseline → same optimum
    DA-pruning   = DA_astar, symmetry pruning ON + c_deform loose-space
                   pruning + sampling check

All three planners share the same cost model (identical expansion
kernel, same ``reconfig_check='sampling'`` rule).  baseline vs DA-full
must match (the fairness check); DA-pruning may differ — symmetry is
pruning, not exact equivalence, and can even return a cheaper path via
the free-relabelling reconfig branch.

Usage (from the repo root)::

    python -m test.test_baseline_vs_da_astar
"""

import time

from pathlib import Path

import matplotlib.pyplot as plt

from core.da_astar import find_path_da_from_map
from core.map_io import load_map
from core.baseline_planner import plan_baseline_from_map
from config import da_astar as cfg
from test._output import out_path

_ROOT = Path(__file__).resolve().parent.parent

# ─── Parameters for THIS benchmark ──────────────────────────
# These override ``config.da_astar``; anything not set here is taken
# from ``cfg`` (formations, weights, robot geometry, cable params …).
XY_STEP = 20
S_MIN, S_MAX, N_S = 0.7, 1.3, 10
N_THETA = 36
FREE_THETA = False
FREE_CONFIG = False
FREE_S = False

MAP_PATH = str(_ROOT / "random_maps" / "random_map_4.png")
HEIGHT_MAP_PATH = str(_ROOT / "random_maps" / "random_map_4_height.png")

# Adaptive-branching radius, rebuilt from the local RF/S_MAX/RB/XY_STEP
# (cfg.C_DEFORM is derived from cfg's values, so it can't be reused).
C_DEFORM = (cfg.RF * S_MAX) + cfg.RB + XY_STEP

START = (200 // XY_STEP, 3800 // XY_STEP, 0, 5, 0)
GOAL  = (3800 // XY_STEP, 3800 // XY_STEP, 0, 5, 0)

# Final figure: the three paths overlaid on the map, written to the
# shared test_output/ folder.  Set to None to pop up an interactive
# window instead of writing a file.
PLOT_PATH = out_path("baseline_vs_da_astar_paths.png")
PLOT_DPI = 200


# Shared kwargs for the FULL-space planners (baseline + DA-full).
# These omit c_deform / use_symmetry so each call can set them.
def _full_common():
    return dict(
        obs_thresh=cfg.OBS_THRESH, rb=cfg.RB, rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=XY_STEP, n_theta=N_THETA,
        s_min=S_MIN, s_max=S_MAX, n_s=N_S,
        w_move=cfg.W_MOVE, w_rot=cfg.W_ROT,
        w_scale=cfg.W_SCALE, w_config=cfg.W_CONFIG,
        reconfig_check=cfg.RECONFIG_CHECK, n_arc_samples=cfg.N_ARC_SAMPLES,
        free_theta=FREE_THETA, free_s=FREE_S,
        free_config=FREE_CONFIG,
        height_map_path=HEIGHT_MAP_PATH,
        L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        height_max=cfg.HEIGHT_MAX,
        verbose=False,
    )


def _to_pixels(path):
    """[(ix,iy,iθ,is,ic), …] → (xs, ys) pixel polyline of the centre.

    Uses THIS benchmark's XY_STEP (not cfg's), so the polyline lines up
    with the map the paths were actually planned on.
    """
    xs = [s[0] * XY_STEP for s in path]
    ys = [s[1] * XY_STEP for s in path]
    return xs, ys


def _plot_paths(paths, save=None, dpi=200):
    """Overlay the planners' centre paths on the occupancy map.

    ``paths`` is a list of ``(label, path, cost, colour, style)``; entries
    whose path is None (no solution) are skipped.
    """
    img, _, _ = load_map(MAP_PATH, cfg.OBS_THRESH)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_axis_off()

    for label, path, cost, colour, style in paths:
        if not path:
            continue
        xs, ys = _to_pixels(path)
        ax.plot(xs, ys, style, color=colour, lw=2.0, alpha=0.9,
                label=f"{label}  (cost {cost:.1f}, {len(path)} steps)")

    # Start / goal markers, in pixels, from the states the planners got.
    ax.plot(START[0] * XY_STEP, START[1] * XY_STEP, 'o', color='lime',
            ms=12, mec='k', zorder=5, label='start')
    ax.plot(GOAL[0] * XY_STEP, GOAL[1] * XY_STEP, '*', color='gold',
            ms=20, mec='k', zorder=5, label='goal')

    # Legend below the axes: the goal sits in the bottom-right corner, so
    # an in-axes legend would cover both it and the paths converging on it.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=9, framealpha=0.9)
    ax.set_title("baseline vs DA-full vs DA-pruning — planned paths",
                 fontsize=13)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"\n  Saved path figure to {save}")
        plt.close(fig)
    else:
        print("\n  Showing path figure (close window to exit) …")
        plt.show()


def main():
    print("=" * 78)
    print("BASELINE   vs   DA A* (full)   vs   DA A* (symmetry + c_deform)")
    print("=" * 78)
    print(f"  map={MAP_PATH}  xy_step={XY_STEP}  "
          f"n_theta={N_THETA}  n_s={N_S}  "
          f"n_config={len(cfg.FORMATIONS_DEG)}")
    print(f"  start={START}  goal={GOAL}")
    print(f"  goal mode: free_theta={FREE_THETA} free_s={FREE_S} "
          f"free_config={FREE_CONFIG}")

    common = _full_common()
    # Pin the same transition-feasibility rule for every planner.
    common["reconfig_check"] = "sampling"

    # baseline + DA-full run on the SAME full space (fair, must match).
    da_full = dict(common, c_deform=None, use_symmetry=False)
    # DA-pruning: symmetry pruning ON + c_deform, same check.
    da_prune = dict(common, c_deform=C_DEFORM, use_symmetry=True)
    print(f"  DA-pruning: c_deform={C_DEFORM:.1f}px, "
          f"reconfig_check='sampling'")

    # ── Warm up every JIT path once (compile, not timed) ──────
    print("\n  Warming up JIT …")
    plan_baseline_from_map(MAP_PATH, START, GOAL,
                           return_timings=True, **common)
    find_path_da_from_map(MAP_PATH, START, GOAL, **da_full)
    find_path_da_from_map(MAP_PATH, START, GOAL, **da_prune)

    print("\n--- baseline / DA-full / DA-pruning ---")

    pp, cp, settled, tm = plan_baseline_from_map(
        MAP_PATH, START, GOAL, return_timings=True, **common)
    baseline_total = (tm['map_time'] + tm['graph_time']
                      + tm['bfs_time'] + tm['search_time'])

    t = time.perf_counter()
    pl, cl, expanded = find_path_da_from_map(
        MAP_PATH, START, GOAL, **da_full)
    t_full = time.perf_counter() - t

    t = time.perf_counter()
    ps, cs, exp_s = find_path_da_from_map(
        MAP_PATH, START, GOAL, **da_prune)
    t_prune = time.perf_counter() - t

    # baseline vs DA-full fairness check.
    pl_match = (pp is not None and pl is not None
                and abs(cp - cl) < 1e-6 and len(pp) == len(pl))

    print(f"  {'planner':<24} {'cost':>10} {'steps':>6} "
          f"{'expanded/settled':>16} {'time(s)':>9}")
    print("  " + "-" * 68)
    print(f"  {'BASELINE (full, Dijkstra)':<24} {cp:>10.2f} "
          f"{len(pp) if pp else 0:>6} {settled:>16,} {baseline_total:>9.3f}")
    print(f"  {'DA-full A*':<24} {cl:>10.2f} "
          f"{len(pl) if pl else 0:>6} {expanded:>16,} {t_full:>9.3f}")
    cs_str = f"{cs:.2f}" if ps is not None else "NO PATH"
    print(f"  {'DA-pruning':<24} {cs_str:>10} "
          f"{len(ps) if ps else 0:>6} {exp_s:>16,} {t_prune:>9.3f}")

    # ── Reading guide ─────────────────────────────────────
    bar = "═" * 78
    print("\n" + bar)
    if pl_match:
        print("  ✓ baseline == DA-full (fairness check: same graph, same "
              "unfolded optimum).")
    else:
        print("  ✗ baseline != DA-full — fairness check FAILED, "
              "investigate.")
    print(bar)
    if ps is not None and pp is not None:
        d_prune = cs - cp
        print(f"  Δ = DA-pruning − baseline = {d_prune:+.4f}")
        if d_prune < -1e-6:
            print("  → NEGATIVE: symmetry pruning found a cheaper path than "
                  "the unfolded baseline")
            print("    optimum — the legitimate relabelling shortcut.")
        elif d_prune > 1e-6:
            print("  → POSITIVE: the pruning did not help on this query "
                  "(folding / c_deform")
            print("    pruning lost a bit of optimality).")
        else:
            print("  → ≈ 0: same cost; symmetry acted as pure search "
                  "pruning here.")
    print(bar)
    print("  Δ is NOT expected to be zero: symmetry pruning exploits robot")
    print("  interchangeability + symmetry folding. Symmetry is pruning, "
          "not exact equality.")
    print(bar)

    # ── Final figure: the three paths on the map ──────────────
    # DA-full is dashed on top of the solid baseline so the two remain
    # visible where they coincide.
    _plot_paths(
        [("BASELINE (full, Dijkstra)", pp, cp, '#e41a1c', '-'),
         ("DA-full A*",                pl, cl, '#4daf4a', '--'),
         ("DA-pruning",                ps, cs, '#377eb8', '-')],
        save=PLOT_PATH, dpi=PLOT_DPI,
    )


if __name__ == "__main__":
    main()

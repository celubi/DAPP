#!/usr/bin/env python3
"""Measure the effect of the XY resolution in DA A*.

Runs the same DA A* query on the same map / start / goal once per XY
resolution in ``XY_STEPS``.

Reports expansions, cost, runtime and path length for all runs.

Usage (from the repo root)::

    python -m test.test_xy_resolution
"""

import time

from pathlib import Path

from core.da_astar import find_path_da_from_map
from config import da_astar as cfg

MAP_PATH = str(Path(__file__).resolve().parent.parent
               / "experimental_maps" / "scaling_map.png")

GOAL = (180, 120, 0, 5, 0)
START  = (1310, 1360, 0, 5, 0)
XY_STEPS = [10, 15, 20, 25]  # XY resolutions (px per cell) to sweep, finest first
S_MIN, S_MAX, N_S = 0.7, 1.3, 10
N_THETA = 36
C_DEFORM = None       # adaptive branching off — only the XY resolution varies
USE_SYMMETRY = False  # symmetry θ-pruning off — only the XY resolution varies
FREE_THETA = False
FREE_S = False
FREE_CONFIG = False


def run(xy_step):
    t0 = time.perf_counter()
    path, cost, n_exp = find_path_da_from_map(
        MAP_PATH, (START[0]/xy_step, START[1]/xy_step, START[2], START[3], START[4]),
        (GOAL[0]/xy_step, GOAL[1]/xy_step, GOAL[2], GOAL[3], GOAL[4]),
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=xy_step, n_theta=N_THETA,
        s_min=S_MIN, s_max=S_MAX, n_s=N_S,
        w_move=cfg.W_MOVE, w_rot=cfg.W_ROT,
        w_scale=cfg.W_SCALE, w_config=cfg.W_CONFIG,
        c_deform=C_DEFORM,
        reconfig_check=cfg.RECONFIG_CHECK,
        n_arc_samples=cfg.N_ARC_SAMPLES,
        use_symmetry=USE_SYMMETRY,
        free_theta=FREE_THETA,
        free_s=FREE_S,
        free_config=FREE_CONFIG,
        height_map_path=None,
        L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        height_max=cfg.HEIGHT_MAX,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    return path, cost, n_exp, dt


def main():
    print("XY resolution effect on DA A*")
    print(f"  Map:   {MAP_PATH}")
    print(f"  Start: {START}")
    print(f"  Goal:  {GOAL}")
    print(f"  Formations: {len(cfg.FORMATIONS_DEG)} configurations")
    print(f"  XY steps: {XY_STEPS}")
    print(f"  S values: {S_MIN} to {S_MAX} in {N_S} steps")
    print(f"  N thetas: {N_THETA}")

    # Warm up the numba JIT once (compile time is not measured).
    print("  Warming up JIT…")
    run(xy_step=XY_STEPS[-1])

    costs = {}
    steps = {}
    expandeds = {}
    times = {}

    for xy_step in XY_STEPS:
        print(f"\n▸ xy_step={xy_step} …")
        path, cost, n_exp, dt = run(xy_step=xy_step)
        if path is None:
            print(f"  [xy_step={xy_step}] ✗ no path found")
            return
        print(f"  [xy_step={xy_step}] cost={cost:.2f}  steps={len(path)}  "
              f"expanded={n_exp:,}  time={dt:.3f}s")
        costs[xy_step] = cost
        steps[xy_step] = len(path)
        expandeds[xy_step] = n_exp
        times[xy_step] = dt

    print("\n── Summary ──")

    cost_base = costs[XY_STEPS[-1]]
    for xy_step, cost in costs.items():
        print(f"  xy_step={xy_step} cost: {cost:.2f} ratio={cost / cost_base:.4f}×")

    print("----------------------------------------")

    steps_base = steps[XY_STEPS[-1]]
    for xy_step, n_steps in steps.items():
        print(f"  xy_step={xy_step} steps: {n_steps} ratio={n_steps / steps_base:.4f}×")

    print("----------------------------------------")

    expanded_base = expandeds[XY_STEPS[-1]]
    for xy_step, n_exp in expandeds.items():
        print(f"  xy_step={xy_step} expanded: {n_exp:,} ratio={n_exp / expanded_base:.4f}×")  

    print("----------------------------------------")

    time_base = times[XY_STEPS[-1]]
    for xy_step, dt in times.items():
        print(f"  xy_step={xy_step} time: {dt:.3f}s ratio={dt / time_base:.4f}×")

    print("----------------------------------------")

    grid_base = XY_STEPS[-1]
    for xy_step in XY_STEPS:
        factor = grid_base / xy_step
        print(f"  xy_step={xy_step} expected ratio={factor**2:.4f}×")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Measure the effect of the θ resolution in DA A*.

Runs the same DA A* query on the same map / start / goal once per θ
resolution in ``N_THETAS``.

Reports expansions, cost, runtime and path length for all runs.

Usage (from the repo root)::

    python -m test.test_rot_resolution
"""

import time

from pathlib import Path

from core.da_astar import find_path_da_from_map
from config import da_astar as cfg

MAP_PATH = str(Path(__file__).resolve().parent.parent
               / "experimental_maps" / "scaling_map.png")

GOAL = (180, 120, 0, 5, 0)
START  = (1310, 1360, 0, 5, 0)
XY_STEP = 25
S_MIN, S_MAX, N_S = 0.7, 1.3, 10
N_THETAS = [144, 108, 72, 36]  # θ resolutions to sweep, finest first
C_DEFORM = None       # adaptive branching off — only the θ resolution varies
USE_SYMMETRY = False  # symmetry θ-pruning off — only the θ resolution varies
FREE_THETA = False
FREE_CONFIG = False
FREE_S = False


def run(n_theta):
    t0 = time.perf_counter()
    path, cost, n_exp = find_path_da_from_map(
        MAP_PATH, (START[0]/XY_STEP, START[1]/XY_STEP, START[2], START[3], START[4]),
        (GOAL[0]/XY_STEP, GOAL[1]/XY_STEP, GOAL[2], GOAL[3], GOAL[4]),
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=XY_STEP, n_theta=n_theta,
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
    print("Rotation resolution effect on DA A*")
    print(f"  Map:   {MAP_PATH}")
    print(f"  Start: {START}")
    print(f"  Goal:  {GOAL}")
    print(f"  Formations: {len(cfg.FORMATIONS_DEG)} configurations")
    print(f"  XY step: {XY_STEP}")
    print(f"  S values: {S_MIN} to {S_MAX} in {N_S} steps")
    print(f"  N thetas: {N_THETAS}")

    # Warm up the numba JIT once (compile time is not measured).
    print("  Warming up JIT …")
    run(n_theta=N_THETAS[-1])

    costs = {}
    steps = {}
    expandeds = {}
    times = {}

    for n_theta in N_THETAS:
        print(f"\n▸ n_theta={n_theta} …")
        path, cost, n_exp, dt = run(n_theta=n_theta)
        if path is None:
            print(f"  [n_theta={n_theta}] ✗ no path found")
            return
        print(f"  [n_theta={n_theta}] cost={cost:.2f}  steps={len(path)}  "
              f"expanded={n_exp:,}  time={dt:.3f}s")
        costs[n_theta] = cost
        steps[n_theta] = len(path)
        expandeds[n_theta] = n_exp
        times[n_theta] = dt

    print("\n── Summary ──")

    cost_base = costs[N_THETAS[-1]]
    for n_theta, cost in costs.items():
        print(f"  n_theta={n_theta} cost: {cost:.2f} ratio={cost / cost_base:.4f}×")

    print("----------------------------------------")

    steps_base = steps[N_THETAS[-1]]
    for n_theta, n_steps in steps.items():
        print(f"  n_theta={n_theta} steps: {n_steps} ratio={n_steps / steps_base:.4f}×")

    print("----------------------------------------")

    expanded_base = expandeds[N_THETAS[-1]]
    for n_theta, n_exp in expandeds.items():
        print(f"  n_theta={n_theta} expanded: {n_exp:,} ratio={n_exp / expanded_base:.4f}×")  

    print("----------------------------------------")

    time_base = times[N_THETAS[-1]]
    for n_theta, dt in times.items():
        print(f"  n_theta={n_theta} time: {dt:.3f}s ratio={dt / time_base:.4f}×")

    print("----------------------------------------")

    angle_base = N_THETAS[-1]
    for n in N_THETAS:
        factor = n / angle_base
        print(f"  n_thetas={n} expected ratio={factor:.4f}×")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run DA A* over combinations of maps / starts / goals / formation sets.

For each (map, start, goal) triple in ``MAP_PATHS`` / ``STARTS`` /
``GOALS``, the same query is solved once per formation set in
``FORMATIONS``.

Reports expansions, cost, runtime and path length for all runs.

Usage (from the repo root)::

    python -m test.test_different_maps
"""

import time

from pathlib import Path

from core.da_astar import find_path_da_from_map
from config import da_astar as cfg

MAP_PATHS = [str(Path(__file__).resolve().parent.parent
                 / "experimental_maps" / "scaling_map.png")]

STARTS = [(1310, 1360, 0, 5, 0)]

GOALS  = [(180, 120, 0, 5, 0)]

FORMATIONS = [
    [cfg.FORMATIONS_DEG[0], cfg.FORMATIONS_DEG[1], cfg.FORMATIONS_DEG[2]],
    [cfg.FORMATIONS_DEG[0], cfg.FORMATIONS_DEG[1]],
    #[cfg.FORMATIONS_DEG[0], cfg.FORMATIONS_DEG[2]],
    #[cfg.FORMATIONS_DEG[1], cfg.FORMATIONS_DEG[2]],
    [cfg.FORMATIONS_DEG[0]],
    #[cfg.FORMATIONS_DEG[1]],
    #[cfg.FORMATIONS_DEG[2]],
]

XY_STEP = 25
S_MIN, S_MAX = 0.7, 1.3
N_S = 10
N_THETA = 36
C_DEFORM = None       # e.g. ((cfg.RF + cfg.RB) * S_MAX) + XY_STEP to enable adaptive branching
USE_SYMMETRY = False  # symmetry θ-pruning off by default
FREE_THETA = False
FREE_S = False
FREE_CONFIG = False


def run(map_path, start, goal, formations_deg):
    t0 = time.perf_counter()
    path, cost, n_exp = find_path_da_from_map(
        map_path, (start[0]/XY_STEP, start[1]/XY_STEP, start[2], start[3], start[4]),
        (goal[0]/XY_STEP, goal[1]/XY_STEP, goal[2], goal[3], goal[4]),
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=formations_deg,
        xy_step=XY_STEP, n_theta=N_THETA,
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
    print("DA A* over maps / starts / goals / formation sets")
    print(f"  Maps:   {MAP_PATHS}")
    print(f"  Starts: {STARTS}")
    print(f"  Goals:  {GOALS}")
    print(f"  Formation sets: {len(FORMATIONS)} options")
    print(f"  XY step: {XY_STEP}")
    print(f"  S values: {S_MIN} to {S_MAX} in {N_S} steps")
    print(f"  N thetas: {N_THETA}")

    # Warm up the numba JIT once (compile time is not measured).
    print("  Warming up JIT …")
    run(map_path=MAP_PATHS[-1], start=STARTS[-1], goal=GOALS[-1], formations_deg=FORMATIONS[-1])

    

    for map_path, start, goal in zip(MAP_PATHS, STARTS, GOALS):
        print(f"\n▸ start={start} goal={goal} map={map_path} ...")

        costs = {}
        steps = {}
        expandeds = {}
        times = {}

        for i, formations_deg in enumerate(FORMATIONS):
            print(f"  Testing formation option {i} ...")
            path, cost, n_exp, dt = run(map_path=map_path, start=start, goal=goal, formations_deg=formations_deg)
            if path is None:
                print(f"    [start={start} goal={goal} formations={i}] ✗ no path found")
                return
            print(f"    [start={start} goal={goal} formations={i}] cost={cost:.2f}  steps={len(path)}  "
                  f"expanded={n_exp:,}  time={dt:.3f}s")
            costs[i] = cost
            steps[i] = len(path)
            expandeds[i] = n_exp
            times[i] = dt
             
        print("\n── Summary ──")

        cost_base = costs[len(FORMATIONS)-1]
        
        expanded_base = expandeds[len(FORMATIONS)-1]
        steps_base = steps[len(FORMATIONS)-1]
        time_base = times[len(FORMATIONS)-1]
        for i in range(len(FORMATIONS)):
            print(f"  FM={i} ({len(FORMATIONS[i])} cgfs) | cost: {costs[i]:.2f} ratio={costs[i] / cost_base:.4f}× | expanded: {expandeds[i]:,} ratio={expandeds[i] / expanded_base:.4f}× | time: {times[i]:.3f}s ratio={times[i] / time_base:.4f}×")
            
        print("----------------------------------------")

if __name__ == '__main__':
    main()

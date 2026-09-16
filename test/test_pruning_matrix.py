#!/usr/bin/env python3
"""Benchmark the DA_astar pruning matrix: symmetry × c_deform (2×2).

Runs the same DA A* query on the same map / start / goal four times,
once per combination of the two pruning knobs:

    ┌──────────────┬──────────────────────┬──────────────────────────┐
    │              │  c_deform OFF        │  c_deform ON             │
    ├──────────────┼──────────────────────┼──────────────────────────┤
    │ sym OFF      │ 1. NO PRUNING (ref)  │ 3. c_deform only         │
    │ sym ON       │ 2. symmetry only     │ 4. symmetry + c_deform   │
    └──────────────┴──────────────────────┴──────────────────────────┘

* **symmetry** (``use_symmetry``) — folds θ to each template's
  fundamental period and branches reconfigurations over the distinct
  orbit representatives.
* **c_deform** — suppresses scale / reconfiguration edges where the
  formation-centre clearance exceeds ``C_DEFORM_ON`` (open space).

Reports cost, path length, expansions and runtime for all runs, with
ratios vs the no-pruning reference.  Pruning is not exact: cost may
differ from the reference in either direction, so nothing is asserted.
c_deform only prunes in open space; the printed "% cells open" shows
how much of the map it can act on.

Usage (from the repo root)::

    python -m test.test_pruning_matrix
"""

import time

from pathlib import Path

from core.da_astar import find_path_da_from_map, _precompute_clearance_grid
from core.map_io import load_map
from config import da_astar as cfg

MAP_PATH = str(Path(__file__).resolve().parent.parent
               / "experimental_maps" / "pruning_map.png")

START = (130, 130, 0, 5, 0)
GOAL = (2730, 1305, 0, 5, 0)

XY_STEP = 15

# The planner expects start/goal in grid-cell units, not pixels
# (see test/test_rot_resolution.py). START/GOAL above are in pixels.
START_G = (START[0] / XY_STEP, START[1] / XY_STEP, START[2], START[3], START[4])
GOAL_G = (GOAL[0] / XY_STEP, GOAL[1] / XY_STEP, GOAL[2], GOAL[3], GOAL[4])
S_MIN, S_MAX, N_S = 0.7, 1.3, 10
N_THETA = 72
FREE_THETA = False
FREE_CONFIG = False
FREE_S = False

# c_deform ON value, rebuilt from the local S_MAX/XY_STEP
# (cfg.C_DEFORM is derived from cfg's values, so it can't be reused).
C_DEFORM_ON = (cfg.RF * S_MAX) + cfg.RB + XY_STEP

# Reconfig transition check, shared by every cell so only pruning differs.
# 'sampling' exercises the per-robot arc-sampling path; 'clearance' is
# the coarser rotation-invariant disc — swap if desired.
RECONFIG_CHECK = "sampling"


def run(use_symmetry, c_deform):
    t0 = time.perf_counter()
    path, cost, n_exp = find_path_da_from_map(
        MAP_PATH, START_G, GOAL_G,
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=XY_STEP, n_theta=N_THETA,
        s_min=S_MIN, s_max=S_MAX, n_s=N_S,
        w_move=cfg.W_MOVE, w_rot=cfg.W_ROT,
        w_scale=cfg.W_SCALE, w_config=cfg.W_CONFIG,
        c_deform=c_deform,
        reconfig_check=RECONFIG_CHECK,
        n_arc_samples=cfg.N_ARC_SAMPLES,
        use_symmetry=use_symmetry,
        free_theta=FREE_THETA,
        free_s=FREE_S,
        free_config=FREE_CONFIG,
        height_map_path=None,
        L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        height_max=cfg.HEIGHT_MAX,
        verbose=False,
    )
    return path, cost, n_exp, time.perf_counter() - t0


# (label, use_symmetry, c_deform).  Ordered so the no-pruning reference
# is first (Δ is always vs cell 1).
RUNS = [
    ("1. no pruning",          False, None),
    ("2. symmetry",            True,  None),
    ("3. c_deform",            False, C_DEFORM_ON),
    ("4. symmetry + c_deform", True,  C_DEFORM_ON),
]


def main():
    print("Pruning matrix benchmark: symmetry × c_deform")
    print(f"  Map:   {MAP_PATH}")
    print(f"  Start: {START}")
    print(f"  Goal:  {GOAL}")
    print(f"  Formations: {len(cfg.FORMATIONS_DEG)} configurations")
    print(f"  reconfig_check: '{RECONFIG_CHECK}'  |  "
          f"c_deform ON = {C_DEFORM_ON:.1f}px")

    # Diagnostic: what fraction of the map is open enough for c_deform to
    # actually prune?  If this is small, cells 3 & 4 ≈ cells 1 & 2.
    _, _, dist_map = load_map(MAP_PATH, cfg.OBS_THRESH)
    clr = _precompute_clearance_grid(dist_map, XY_STEP)
    frac_open = float((clr > C_DEFORM_ON).mean())
    print(f"  clearance: {frac_open * 100:.1f}% of cells clear the c_deform "
          f"threshold (that is all c_deform can prune)")
    if frac_open < 0.10:
        print("    ⚠ dense map: c_deform will prune almost nothing here — "
              "expect cells 3≈1 and 4≈2.")
    print()

    # Warm up every JIT path once (compile, not timed): symmetry on/off
    # and c_deform on/off, since the clearance-grid branch differs.
    print("  Warming up JIT …")
    for _, use_symmetry, c_deform in RUNS:
        run(use_symmetry, c_deform)

    results = {}
    for label, use_symmetry, c_deform in RUNS:
        print(f"\n▸ {label} …")
        path, cost, n_exp, dt = run(use_symmetry, c_deform)
        if path is None:
            print(f"  [{label}] ✗ no path found")
            return
        print(f"  cost={cost:.2f}  steps={len(path)}  "
              f"expanded={n_exp:,}  time={dt:.3f}s")
        results[label] = (path, cost, n_exp, dt)

    ref_label = RUNS[0][0]
    _, c_ref, e_ref, t_ref = results[ref_label]

    print("\n── Summary (Δ / ratios vs cell 1, no pruning) ──")
    hdr = (f"  {'configuration':<24} {'cost':>10} {'Δcost%':>8} "
           f"{'steps':>6} {'expanded':>12} {'exp×':>6} "
           f"{'time(s)':>9} {'time×':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, _, _ in RUNS:
        path, cost, n_exp, dt = results[label]
        dcost = (cost - c_ref) / c_ref * 100.0 if c_ref else 0.0
        exp_x = e_ref / n_exp if n_exp else float('inf')
        time_x = t_ref / dt if dt else float('inf')
        print(f"  {label:<24} {cost:>10.2f} {dcost:>+7.2f}% "
              f"{len(path):>6} {n_exp:>12,} {exp_x:>5.2f}× "
              f"{dt:>9.3f} {time_x:>5.2f}×")

    print("\n  Reading guide (pruning, not exact):")
    print("    • exp× / time× > 1 ⇒ that pruning searched LESS than the "
          "no-pruning reference.")
    print("    • Δcost% may be + or − : symmetry can find a cheaper path "
          "(free relabelling), c_deform only prunes so its cost is ≥ "
          "reference.")
    print("    • cell 4 shows whether the two prunings COMPOSE (combined "
          "speed-up) or overlap (one subsumes the other's savings).")


if __name__ == '__main__':
    main()

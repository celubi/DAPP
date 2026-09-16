#!/usr/bin/env python3
"""Demo — CCO_planner (Critical Crossable Obstacle) formation planning.

Start and goal are determined AUTOMATICALLY at the two ends of the
obstacle: ``find_start`` walks the L/R anchor chains from one end, and
again on the reversed chains from the other.

The whole pipeline lives in :mod:`core.cco_planner`; this demo only
parses arguments, calls the driver and animates the result:

1. :func:`core.cco_planner.prepare_cco_scene` — wall map, clearance
   map, optional payload height map.
2. :func:`core.cco_planner.build_cco_chains` — inflate the obstacle,
   split its boundary into the LEFT / RIGHT anchor chains.
3. :func:`core.cco_planner.find_path_cco` — prefilter → parallel
   kernel batch → reshape → start/goal at the obstacle ends → mini
   A*.  Its ``t_search`` is the same "Find path" quantity
   ``test_cco_vs_da_astar`` reports.
4. Animate the 6-robot bilateral formation along the path.

Usage (from the repo root)::

    python -m demo.demo_cco_planner
    python -m demo.demo_cco_planner --no-animate
"""

from __future__ import annotations

import argparse
import time

from numba import set_num_threads, get_num_threads

from core.cco_planner import (
    prepare_cco_scene, build_cco_chains, find_path_cco,
)
from core.cco_kernels import warmup as warmup_kernels, STEP_COARSE
from core.cco_astar import warmup as warmup_astar
from core.cco_animate import animate_path
from config import cco_planner as cfg


def _print_scene_breakdown(t_scene, scene_timings, chain_timings,
                           do_payload_check):
    """Print the scene-prep total plus its per-sub-step breakdown."""
    print(f"\n  Scene preparation: {t_scene*1000:8.1f} ms")
    print(f"    wall imread:       {scene_timings['wall_imread']*1000:8.1f} ms")
    print(f"    clearance map:     {scene_timings['clearance_map']*1000:8.1f} ms")
    print(f"    obstacle inflate:  {chain_timings['obstacle_inflate']*1000:8.1f} ms")
    print(f"    split L/R chains:  {chain_timings['split_chains']*1000:8.1f} ms")
    if do_payload_check:
        print(f"    height-map load:   {scene_timings['height_load']*1000:8.1f} ms")


def run_cco(map_path=cfg.MAP_PATH, wall_path=cfg.WALL_PATH,
            bar=cfg.BAR, tol_frac=cfg.TOL_FRAC, step_size=cfg.STEP_SIZE,
            robot_r=cfg.ROBOT_R, clearance_margin=cfg.CLEARANCE_MARGIN,
            intra_robot_dist=cfg.INTRA_ROBOT_DIST, r_infl=cfg.R_INFL,
            height_map_path=cfg.HEIGHT_MAP_PATH, l_pole=cfg.L_POLE,
            l_rope=cfg.L_ROPE, cable_step=cfg.CABLE_SAMPLE_STEP_PX,
            height_max=cfg.HEIGHT_MAX, verbose=True):
    """Scene → chains → path.  Shared with the video renderer.

    Returns ``(scene, path_AB, res, chain_times, t_scene)``.
    """
    height_map_path = (height_map_path
                       if (height_map_path and cfg.PAYLOAD_CHECK) else None)
    t_scene0 = time.perf_counter()
    scene = prepare_cco_scene(wall_path, height_map_path=height_map_path,
                              height_max=height_max)
    array_a, array_b, chain_times = build_cco_chains(map_path, r_infl,
                                                     step_size)
    t_scene = time.perf_counter() - t_scene0

    if verbose:
        print(f"L-chain: {len(array_a)}  R-chain: {len(array_b)}  "
              f"map: {scene.wall_img.shape}")
        print("payload check: "
              + (f"ON  | L_pole={l_pole}, L_rope={l_rope}, "
                 f"cable_step={cable_step}px, height_max={height_max}"
                 if scene.do_payload else "OFF"))

    path_AB, res = find_path_cco(
        array_a, array_b, scene,
        bar=bar, tol_frac=tol_frac, r_infl=r_infl,
        robot_r=robot_r, clearance_margin=clearance_margin,
        intra_robot_dist=intra_robot_dist,
        L_pole=l_pole, L_rope=l_rope,
        cable_sample_step_px=cable_step)
    return scene, path_AB, res, chain_times, t_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map",  default=cfg.MAP_PATH)
    p.add_argument("--wall", default=cfg.WALL_PATH)
    p.add_argument("--bar",  type=float, default=cfg.BAR)
    p.add_argument("--tol",  type=float, default=cfg.TOL_FRAC)
    p.add_argument("--step-size", type=int,   default=cfg.STEP_SIZE)
    p.add_argument("--robot-r",   type=float, default=cfg.ROBOT_R)
    p.add_argument("--clearance-margin", type=float,
                   default=cfg.CLEARANCE_MARGIN,
                   help="safety margin (px) added to robot radius for the "
                        "kernel collision check only (not for drawing); "
                        "0 disables it")
    p.add_argument("--deg",       type=float, default=cfg.INTRA_ROBOT_DIST)
    p.add_argument("--r-infl",    type=float, default=cfg.R_INFL)
    p.add_argument("--height-map", default=cfg.HEIGHT_MAP_PATH,
                   help="height map for the payload/cable check "
                        "(omit / pass '' to disable)")
    p.add_argument("--l-pole", type=float, default=cfg.L_POLE)
    p.add_argument("--l-rope", type=float, default=cfg.L_ROPE)
    p.add_argument("--cable-step", type=float,
                   default=cfg.CABLE_SAMPLE_STEP_PX,
                   help="cable sample spacing in px")
    p.add_argument("--height-max", type=int, default=cfg.HEIGHT_MAX,
                   help="values above this in the height map are free")
    p.add_argument("--threads",   type=int,   default=cfg.THREADS,
                   help="Numba thread count (0 = all)")
    p.add_argument("--no-animate", action="store_true",
                   help="skip the final animation (headless runs)")
    args = p.parse_args()

    if args.threads > 0:
        set_num_threads(args.threads)

    min_dist = args.bar - args.tol * args.bar
    max_dist = args.bar + args.tol * args.bar
    print(f"BAR={args.bar}, tol_frac={args.tol}, "
          f"MIN_DIST={min_dist:.0f}, MAX_DIST={max_dist:.0f}, "
          f"R_infl={args.r_infl}, coarse={STEP_COARSE}px, "
          f"step_size={args.step_size}px, "
          f"robot_r={args.robot_r}(+{args.clearance_margin} check), "
          f"threads={get_num_threads()}")

    # ── JIT warmup (one-time compilation cost) ────────────
    t = time.perf_counter()
    warmup_kernels()
    warmup_astar()
    print(f"JIT warmup: {time.perf_counter() - t:.2f} s")

    # ── 1-2) Scene, chains, path ──────────────────────────
    scene, path_AB, res, chain_times, t_scene = run_cco(
        map_path=args.map, wall_path=args.wall, bar=args.bar,
        tol_frac=args.tol, step_size=args.step_size, robot_r=args.robot_r,
        clearance_margin=args.clearance_margin, intra_robot_dist=args.deg,
        r_infl=args.r_infl, height_map_path=args.height_map,
        l_pole=args.l_pole, l_rope=args.l_rope, cable_step=args.cable_step,
        height_max=args.height_max)

    print(f"pre-filter: {res.n_kept} kept (of {res.n_pairs})")
    print(f"kernel: {res.t_kernel*1000:.0f} ms "
          f"({res.t_kernel/max(1, res.n_kept)*1e6:.2f} µs/anchor)")
    print(f"  LEFT:  {res.n_left}    RIGHT: {res.n_right}    "
          f"any: {res.n_any} / {res.n_kept} "
          f"({100*res.n_any/max(1, res.n_kept):.1f}%)")
    print(f"valid nodes: {res.n_valid}")
    print(f"\n--- A* search ---")
    print(f"Start: {res.start_ij}  (node id {res.start_id})")
    print(f"Goal:  {res.goal_ij}  (node id {res.goal_id})")
    print(f"A*: {res.t_astar*1000:.0f} ms, "
          f"path length {len(path_AB) if path_AB else 0}")

    # ── Timing summary ────────────────────────────────────
    _print_scene_breakdown(t_scene, scene.timings, chain_times,
                           scene.do_payload)
    print(f"  Find path (all):   {res.t_search*1000:8.1f} ms  "
          f"(kernel {res.t_kernel*1000:.0f} ms, "
          f"mini A* {res.t_astar*1000:.0f} ms)")

    if path_AB is None:
        raise RuntimeError("No path found by JIT A*")

    # ── 3) Animate ────────────────────────────────────────
    if args.no_animate:
        return
    animate_path(path_AB, scene.wall_img, args.bar, args.robot_r,
                 args.deg, scene.clearance,
                 title="CCO_planner — Formation Path")


if __name__ == "__main__":
    main()

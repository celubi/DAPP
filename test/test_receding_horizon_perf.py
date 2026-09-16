#!/usr/bin/env python3
"""Receding horizon vs full search on the fully-known map.

Measures what the *online* (progressively-revealed) planner costs
relative to knowing the whole map up front, across a grid of

* ``step_xy``        — the values in ``STEP_XY_M`` (1 px = 1 cm);
* ``sensing radius`` — the factors in ``SENSOR_FACTORS`` of
  ``(RF·S_MAX + RB)``.

Both the online run and its baseline run DA_astar with symmetry
pruning ON.  For each ``step_xy`` the baseline solves the same query
once on the fully-known map; ``expanded_online / expanded_baseline``
and ``time_online / time_baseline`` are the overheads of planning
under partial information.  The baseline is per ``step_xy`` only (with
the whole map known the sensing radius is irrelevant).

With ``USE_HEIGHT = True`` the payload / cable check is on for both
runs: the baseline knows every obstacle height up front, while the
online run measures heights with the same sensor disc that reveals
obstacles (unseen cells stay optimistically overflyable).

Every planner parameter is fixed in this file (PLANNER PARAMETERS
block), so the test is self-contained and reproducible.

Usage (from the repo root)::

    python -m test.test_receding_horizon_perf
    python -m test.test_receding_horizon_perf --quick   # 1 combo, smoke test
    python -m test.test_receding_horizon_perf --plot    # + figures
    python -m test.test_receding_horizon_perf --save out.png

``--plot`` draws the runs the timing loop already measured, at the
finest ``step_xy``, one panel per sensing radius — it re-solves
nothing, so the figures and the table always agree.
"""

import argparse
import math
import time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.map_io import load_map, load_height_map
from core.formations import parse_formations, compute_reconfig_costs
from core.receding_horizon import plan_receding_horizon
from core.da_astar import find_path_da_from_map
from test._output import resolve_save


# ═══════════════════════════════════════════════════════════
#  PLANNER PARAMETERS — every knob the planner sees lives here
# ═══════════════════════════════════════════════════════════

OBS_THRESH = 128

_MAPS = str(Path(__file__).resolve().parent.parent / "random_maps")

# ─── Query: map + height map + start/goal + a warm-up hop ──
# Start and goal must be free, land on every step_xy lattice in the
# sweep, and have clearance ≥ RB + RF·S_MAX.  WARMUP_GOAL_PX exists
# only to compile the JIT kernels: a few cells from START_PX, but not
# the same cell at the coarsest step.
MAP_NAME = "random_map_3"
MAP_PATH = f"{_MAPS}/{MAP_NAME}.png"
HEIGHT_MAP_PATH = f"{_MAPS}/{MAP_NAME}_height.png"
START_PX = (200, 200)
GOAL_PX = (3800, 3800)
WARMUP_GOAL_PX = (240, 240)

# ─── Robot / formation geometry (pixels) ───────────────────
RB = 10.0          # robot body radius
RF = 100.0         # formation circle radius at scale s=1

FORMATIONS_DEG = [
    [[0.0], [60.0], [120.0], [180.0], [240.0], [300.0]],   # sym 6
    [[-10.0, 10.0], [110.0, 130.0], [230.0, 250.0]],       # sym 3
    [[-20.0, 0.0, 20.0], [160.0, 180.0, 200.0]],           # sym 2
]

# ─── State discretisation ──────────────────────────────────
# 1 px = 1 cm, so a step of 0.10 m is 10 px on the map.
STEP_XY_M = [0.12, 0.20]
PX_PER_M = 100.0
N_THETA = 36       # must be divisible by every sym order (6, 3, 2)
S_MIN, S_MAX, N_S = 0.7, 1.3, 10
S_MID = N_S // 2   # start/goal scale index

# ─── Sensing ───────────────────────────────────────────────
# sensor_radius = factor · (RF·S_MAX + RB).  The planner's hard floor
# is 2·RF·S_MAX (see plan_receding_horizon's safe_radius check); every
# factor in the sweep must clear it.
SENSOR_FACTORS = [3.0, 4.0]
SENSOR_BASE = (RF * S_MAX) + RB
MAX_ITERATIONS = 200

# ─── Payload / cable height check ──────────────────────────
# USE_HEIGHT = False switches the height check off (pure 2-D planning).
# The payload flies at h = L_POLE − √(L_ROPE² − (RF·s)²); an obstacle
# blocks it where height > h_payload[js].
USE_HEIGHT = True
L_POLE = 200.0
L_ROPE = 180.0
CABLE_SAMPLE_STEP_PX = 20
HEIGHT_MAX = 100

# ─── Cost weights ──────────────────────────────────────────
W_MOVE = 1.0
W_ROT = 1.0
W_SCALE = 1.0
W_CONFIG = 1.0

# ─── Search behaviour ──────────────────────────────────────
C_DEFORM = None            # no clearance-gated branching: isolate the
                           # online-vs-offline effect
RECONFIG_CHECK = 'sampling'
N_ARC_SAMPLES = 8
USE_SYMMETRY = True        # pruning ON for BOTH baseline and online run
FREE_THETA = False
FREE_S = False
FREE_CONFIG = False


# ═══════════════════════════════════════════════════════════
#  Runs
# ═══════════════════════════════════════════════════════════

def _to_grid(px, step_xy):
    """Pixel (x, y) → full grid state (ix, iy, iθ, is, ic)."""
    return (px[0] // step_xy, px[1] // step_xy, 0, S_MID, 0)


def _step_px(step_m):
    return int(round(step_m * PX_PER_M))


def path_cost(path, step_xy):
    """Cost of an EXECUTED trajectory, under the planner's own metric.

    ``RecedingResult.costs`` stores optimistic plan costs, most of
    which are never executed; re-scoring the trajectory actually driven
    is what makes it comparable with the baseline's single-shot cost.

    Mirrors the edge weights in ``da_astar._expand``:
      move    — xy_step (√2·xy_step diagonal) · w_move
      rotate  — rf · s · θ_step · w_rot
      scale   — rf · s_step · w_scale
      reconfig— reconfig_costs[ic,jc] · rf · s · w_config

    A branching reconfig edge's θ jump is free relabelling, not a turn,
    so when the config changes the θ change must NOT be billed as
    rotation.
    """
    if not path or len(path) < 2:
        return 0.0

    formations_rad, _, sym_orders = parse_formations(FORMATIONS_DEG)
    rc_costs, _ = compute_reconfig_costs(formations_rad)
    s_values = np.linspace(S_MIN, S_MAX, N_S)
    th_step = 2.0 * math.pi / N_THETA
    s_step = ((S_MAX - S_MIN) / max(N_S - 1, 1)) if N_S > 1 else 0.0
    # θ lives on each config's fundamental period, so a turn wraps
    # around that period, not the full circle.
    periods = [N_THETA // k for k in
               (sym_orders if USE_SYMMETRY else [1] * len(formations_rad))]

    total = 0.0
    for (ix0, iy0, it0, js0, ic0), (ix1, iy1, it1, js1, ic1) in zip(path,
                                                                    path[1:]):
        if (ix0, iy0) != (ix1, iy1):
            dx, dy = abs(ix1 - ix0), abs(iy1 - iy0)
            total += (math.sqrt(2.0) if dx and dy else 1.0) * step_xy * W_MOVE
        if it0 != it1 and ic0 == ic1:
            # A real turn: shorter way round the config's period.  When
            # the config also changes, the θ jump belongs to the
            # reconfig edge and is free — hence the ic0 == ic1 guard.
            p = periods[ic0]
            d = abs(it1 - it0) % p
            d = min(d, p - d)
            total += d * th_step * RF * s_values[js1] * W_ROT
        if js0 != js1:
            total += abs(js1 - js0) * s_step * RF * W_SCALE
        if ic0 != ic1:
            total += rc_costs[ic0, ic1] * RF * s_values[js1] * W_CONFIG
    return total


def run_baseline(step_xy, goal_px=None):
    """Fully-known map, single shot, symmetry pruning ON — the reference."""
    goal_px = GOAL_PX if goal_px is None else goal_px
    t0 = time.perf_counter()
    path, cost, n_exp = find_path_da_from_map(
        MAP_PATH, _to_grid(START_PX, step_xy),
        _to_grid(goal_px, step_xy),
        obs_thresh=OBS_THRESH,
        rb=RB, rf=RF,
        formations_deg=FORMATIONS_DEG,
        xy_step=step_xy, n_theta=N_THETA,
        s_min=S_MIN, s_max=S_MAX, n_s=N_S,
        w_move=W_MOVE, w_rot=W_ROT,
        w_scale=W_SCALE, w_config=W_CONFIG,
        c_deform=C_DEFORM,
        reconfig_check=RECONFIG_CHECK,
        n_arc_samples=N_ARC_SAMPLES,
        use_symmetry=USE_SYMMETRY,
        free_theta=FREE_THETA,
        free_s=FREE_S,
        free_config=FREE_CONFIG,
        # The baseline knows the whole map — including every height.
        height_map_path=HEIGHT_MAP_PATH if USE_HEIGHT else None,
        L_pole=L_POLE, L_rope=L_ROPE,
        cable_sample_step_px=CABLE_SAMPLE_STEP_PX,
        height_max=HEIGHT_MAX,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    return path, cost, n_exp, dt


def run_receding(occ, step_xy, sensor_factor, goal_px=None,
                 true_height=None):
    """Online run: progressive sensing + replanning, symmetry pruning ON.

    ``true_height`` is the GROUND-TRUTH height map.  The planner never
    sees it directly: plan_receding_horizon reveals it through the
    sensor disc and passes only the measured part to the search.
    """
    goal_px = GOAL_PX if goal_px is None else goal_px
    t0 = time.perf_counter()
    result = plan_receding_horizon(
        occ, _to_grid(START_PX, step_xy), _to_grid(goal_px, step_xy),
        sensor_radius=sensor_factor * SENSOR_BASE,
        rb=RB, rf=RF,
        formations_deg=FORMATIONS_DEG,
        xy_step=step_xy, n_theta=N_THETA,
        s_min=S_MIN, s_max=S_MAX, n_s=N_S,
        w_move=W_MOVE, w_rot=W_ROT,
        w_scale=W_SCALE, w_config=W_CONFIG,
        c_deform=C_DEFORM,
        reconfig_check=RECONFIG_CHECK,
        n_arc_samples=N_ARC_SAMPLES,
        use_symmetry=USE_SYMMETRY,
        free_theta=FREE_THETA,
        free_s=FREE_S,
        free_config=FREE_CONFIG,
        true_height=true_height,
        L_pole=L_POLE, L_rope=L_ROPE,
        cable_sample_step_px=CABLE_SAMPLE_STEP_PX,
        max_iterations=MAX_ITERATIONS,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    return result, dt


# ═══════════════════════════════════════════════════════════
#  Plots — baseline path vs receding-horizon path
# ═══════════════════════════════════════════════════════════

def _crop_bounds(known_mask, margin=80):
    """Bounding box of the sensed region, padded, for a tight view."""
    ys, xs = np.where(known_mask)
    H, W = known_mask.shape
    return (max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin, W),
            max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin, H))


def _plot_case(ax, img, res, b_path, rh_path, step_xy, factor, bounds,
               b_cost, rh_cost):
    """One panel: dimmed map + sensed region at original brightness.

    The map is drawn twice: washed out everywhere (never sensed), then
    again at full contrast masked to ``known_mask_final``.
    """
    import numpy.ma as ma

    x0, x1, y0, y1 = bounds
    radius = factor * SENSOR_BASE

    # Layer 1 — unexplored, with a blue-grey tint (plain alpha on a
    # mostly-white map would be invisible).
    ax.imshow(img, cmap='gray', origin='upper', vmin=0, vmax=255,
              interpolation='nearest')
    ax.imshow(np.ones_like(img), cmap='Blues', origin='upper',
              vmin=0, vmax=1.6, alpha=0.62, interpolation='nearest',
              zorder=1)

    # Layer 2 — explored: original colours, masked outside known_mask.
    seen = ma.masked_where(~res.known_mask_final, img)
    ax.imshow(seen, cmap='gray', origin='upper', vmin=0, vmax=255,
              interpolation='nearest', zorder=2)

    # Outline of the sensed region, so the frontier is unambiguous.
    ax.contour(res.known_mask_final.astype(float), levels=[0.5],
               colors='#f0c000', linewidths=1.2, alpha=0.9, zorder=3)

    bx = [p[0] * step_xy for p in b_path]
    by = [p[1] * step_xy for p in b_path]
    rx = [p[0] * step_xy for p in rh_path]
    ry = [p[1] * step_xy for p in rh_path]

    ax.plot(bx, by, '-', color='#377eb8', lw=2.6, alpha=0.95, zorder=4,
            label=f'baseline, known map — cost {b_cost:,.0f}, '
                  f'{len(b_path)} steps')
    ax.plot(rx, ry, '--', color='#e41a1c', lw=2.2, alpha=0.95, zorder=5,
            label=f'receding horizon — cost {rh_cost:,.0f}, '
                  f'{len(rh_path)} steps')

    ax.plot(bx[0], by[0], 'o', color='lime', ms=11, zorder=6,
            markeredgecolor='k')
    ax.plot(bx[-1], by[-1], '*', color='red', ms=17, zorder=6,
            markeredgecolor='k')

    # Sensor footprint drawn at the goal, to scale.
    ax.add_patch(plt.Circle((rx[-1], ry[-1]), radius, fill=False,
                            ec='#f0c000', ls=':', lw=1.6, zorder=6))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ratio = (rh_cost / b_cost) if b_cost else float('inf')
    ax.set_title(f"sensing radius {factor}×  =  {radius:.0f} px\n"
                 f"{len(res.expansions)} replans, "
                 f"{sum(res.expansions):,} expanded  ·  "
                 f"cost {ratio:.2f}× baseline",
                 fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fontsize=9, framealpha=0.9, ncol=1, borderaxespad=0.)


def make_plots(img, step_m, b_path, by_factor, save=None):
    """Draw the runs that were already measured — nothing is re-solved.

    One panel per sensing radius, at the finest resolution in the
    sweep.
    """
    step_xy = _step_px(step_m)
    cases = [(f, by_factor[f]) for f in sorted(by_factor)]

    print(f"\n  Plotting the measured runs at step_xy={step_m} m "
          f"({step_xy} px): radii {[f for f, _ in cases]}")

    # Re-score both paths under the planner's own metric so the two
    # numbers are directly comparable (see path_cost).
    b_cost = path_cost(b_path, step_xy)

    # Shared crop across panels so the two radii are visually comparable.
    union = np.zeros_like(cases[0][1].known_mask_final)
    for _, res in cases:
        union |= res.known_mask_final
    bounds = _crop_bounds(union)

    fig, axes = plt.subplots(1, len(cases), figsize=(7.5 * len(cases), 8.4))
    axes = np.atleast_1d(axes)
    for ax, (factor, res) in zip(axes, cases):
        rh_cost = path_cost(res.trajectory, step_xy)
        print(f"    sensor {factor}×: baseline cost {b_cost:,.1f}  vs  "
              f"receding cost {rh_cost:,.1f}  "
              f"({rh_cost / b_cost:.3f}× baseline)"
              if b_cost else "")
        _plot_case(ax, img, res, b_path, res.trajectory, step_xy,
                   factor, bounds, b_cost, rh_cost)

    fig.suptitle(
        f"Baseline (fully-known map) vs receding horizon — "
        f"step_xy = {step_m} m, symmetry pruning ON\n"
        f"original colours = sensed by the robot   ·   "
        f"blue tint = never observed",
        fontsize=13)
    # Leave room for the suptitle and for the legends sitting below
    # each panel.
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
        print(f"\n  Figure saved to {save}")
    else:
        print("\n  Showing figure (close window to exit) …")
        plt.show()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main(quick=False, plot=False, save=None):
    steps_m = STEP_XY_M[:1] if quick else STEP_XY_M
    factors = SENSOR_FACTORS[:1] if quick else SENSOR_FACTORS

    print("Receding horizon vs full search on the fully-known map")
    print(f"  Map:            {MAP_PATH}")
    print(f"  Start / Goal:   {START_PX} → {GOAL_PX} px")
    print(f"  Formations:     {len(FORMATIONS_DEG)} configurations")
    print(f"  Symmetry:       pruning ON (baseline AND online run)")
    print(f"  reconfig_check: '{RECONFIG_CHECK}'")
    print(f"  step_xy:        {steps_m} m  "
          f"({[_step_px(s) for s in steps_m]} px @ 1px=1cm)")
    print(f"  sensor factors: {factors}  "
          f"(radius {[f'{f * SENSOR_BASE:.0f}px' for f in factors]})")
    if USE_HEIGHT:
        print(f"  height check:   ON  | L_pole={L_POLE}, L_rope={L_ROPE} "
              f"→ payload flies at h = L_pole − √(L_rope² − (RF·s)²)")
        print(f"                  height map: {HEIGHT_MAP_PATH}")
        print(f"                  baseline knows every height; the "
              f"online run senses them with the same disc as obstacles")
    else:
        print(f"  height check:   OFF (pure 2-D planning)")
    print()

    img, occ, _ = load_map(MAP_PATH, OBS_THRESH)

    # Ground truth only — plan_receding_horizon reveals it through the
    # sensor disc, so the online planner still sees just what it has
    # measured.  The baseline reloads it itself from the same path.
    true_height = (load_height_map(HEIGHT_MAP_PATH, max_height=HEIGHT_MAX)
                   if USE_HEIGHT else None)

    # Warm up every JIT path once (compile time is not measured):
    # coarsest step, short hop, true_height passed so the payload check
    # compiles here too.
    print(f"  Warming up JIT (short hop {START_PX} → "
          f"{WARMUP_GOAL_PX}) …")
    warm = _step_px(max(steps_m))
    run_baseline(warm, WARMUP_GOAL_PX)
    run_receding(occ, warm, factors[0], WARMUP_GOAL_PX,
                 true_height=true_height)

    baselines = {}
    rows = []
    # Keep the actual measured runs so the figures can reuse them
    # instead of re-solving anything: {step_m: (b_path, {factor: res})}.
    runs = {}

    for step_m in steps_m:
        step_xy = _step_px(step_m)

        print(f"\n▸ baseline (known map) — step_xy={step_m} m "
              f"({step_xy} px) …")
        b_path, b_cost, b_exp, b_dt = run_baseline(step_xy)
        if b_path is None:
            print(f"  ✗ baseline found no path — check START_PX / GOAL_PX "
                  f"for this map.")
            return
        # Re-score the baseline path with path_cost too, so both sides
        # of the ratio come from the same function; the gap is printed
        # when they disagree.
        b_cost_rescored = path_cost(b_path, step_xy)
        baselines[step_m] = (b_cost_rescored, b_exp, b_dt, len(b_path))
        drift = abs(b_cost_rescored - b_cost)
        print(f"  cost={b_cost:.2f}  steps={len(b_path)}  "
              f"expanded={b_exp:,}  time={b_dt:.3f}s")
        if drift > 0.01 * max(b_cost, 1.0):
            print(f"  ⚠ path_cost re-score {b_cost_rescored:.2f} differs "
                  f"from the planner's {b_cost:.2f} by {drift:.2f} — "
                  f"cost ratios below are self-consistent but the metric "
                  f"does not match the planner exactly.")

        runs[step_m] = (b_path, {})
        for factor in factors:
            radius = factor * SENSOR_BASE
            print(f"\n▸ receding-horizon — step_xy={step_m} m, "
                  f"sensor={factor} ({radius:.0f} px) …")
            res, dt = run_receding(occ, step_xy, factor,
                                   true_height=true_height)
            n_exp = sum(res.expansions)
            n_iter = len(res.expansions)
            rh_cost = path_cost(res.trajectory, step_xy)
            print(f"  status={res.status}  traj={len(res.trajectory)}  "
                  f"iters={n_iter}  expanded={n_exp:,}  time={dt:.3f}s  "
                  f"cost={rh_cost:,.1f}")
            rows.append((step_m, factor, res.status, len(res.trajectory),
                         n_iter, n_exp, dt, rh_cost))
            runs[step_m][1][factor] = res

    # ── Summary ─────────────────────────────────────────────
    print(f"\n── Summary ──")
    hdr = (f"  {'step_xy':>8} {'sensor':>7} {'status':>14} {'traj':>5} "
           f"{'iters':>6} {'expanded':>12} {'exp/base':>9} "
           f"{'time(s)':>8} {'t/base':>7} {'cost':>10} {'c/base':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for step_m, factor, status, traj, n_iter, n_exp, dt, cost in rows:
        b_cost, b_exp, b_dt, b_steps = baselines[step_m]
        e_ratio = n_exp / b_exp if b_exp else float('inf')
        t_ratio = dt / b_dt if b_dt else float('inf')
        c_ratio = cost / b_cost if b_cost else float('inf')
        print(f"  {step_m:>8.2f} {factor:>7.1f} {status:>14} {traj:>5} "
              f"{n_iter:>6} {n_exp:>12,} {e_ratio:>8.2f}× "
              f"{dt:>8.3f} {t_ratio:>6.2f}× {cost:>10,.0f} "
              f"{c_ratio:>6.2f}×")

    print(f"\n  {'baseline (known map, single shot)':<40}")
    bh = (f"  {'step_xy':>8} {'cost':>10} {'steps':>6} "
          f"{'expanded':>12} {'time(s)':>9}")
    print(bh)
    print("  " + "-" * (len(bh) - 2))
    for step_m in steps_m:
        b_cost, b_exp, b_dt, b_steps = baselines[step_m]
        print(f"  {step_m:>8.2f} {b_cost:>10.2f} {b_steps:>6} "
              f"{b_exp:>12,} {b_dt:>9.3f}")

    print("\n  Reading the table: exp/base and t/base compare the online "
          "planner\n  against the same query solved with the whole map "
          "known.")
    print("    • > 1  — partial information COSTS: replanning at every "
          "commit\n             re-expands states the known-map planner "
          "touches once.")
    print("    • < 1  — partial information PAYS: freespace optimism "
          "erases the\n             obstacles the baseline must route "
          "around, so A* barely\n             branches.")
    print("  c/base closes the picture: it is the price paid in PATH "
          "QUALITY.\n  Cheap search (exp/base < 1) and a worse path "
          "(c/base > 1) are the\n  two halves of the same trade — the "
          "online planner is fast because\n  it is solving an easier, "
          "wrong problem.")

    failed = [r for r in rows if r[2] != "reached"]
    if failed:
        print(f"\n  ⚠ {len(failed)} combination(s) did not reach the goal: "
              f"{[(r[0], r[1], r[2]) for r in failed]}")

    if plot:
        # Finest resolution in the sweep — the most detailed run there is.
        step_m = min(steps_m)
        b_path, by_factor = runs[step_m]
        make_plots(img, step_m, b_path, by_factor, save=save)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='run only the first step_xy / sensor combination '
                         '(smoke test)')
    ap.add_argument('--plot', action='store_true',
                    help='after the timing table, draw the measured runs '
                         '(finest step_xy, one panel per sensing radius)')
    ap.add_argument('--save', type=str, default=None,
                    help='save the figure instead of showing it; '
                         'relative paths land in test_output/')
    args = ap.parse_args()
    main(quick=args.quick,
         plot=args.plot or args.save is not None,
         save=resolve_save(args.save))

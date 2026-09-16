#!/usr/bin/env python3
"""Demo — Interactive Receding-Horizon Formation Planning.

Click the map to set start (green) and goal (red), then watch the
formation execute a receding-horizon plan against a progressively-
revealed map.  The brightened region is what has been sensed so far;
discovered obstacles are highlighted in red.  The dashed yellow circle
is the sensor footprint.

The inner planner is DA_astar (``core.receding_horizon`` replans with
``find_path_da`` at every commit).  Parameters that affect planner
output live in :mod:`config.da_astar`.  Sensor radius, max iterations,
and animation overlays stay local.  Pass ``--start IX,IY --goal IX,IY``
(grid cells) to skip the interactive selection, ``--no-animate`` to
skip the animation.

Usage (from the repo root)::

    python -m demo.demo_receding_horizon
    python -m demo.demo_receding_horizon --start 10,10 --goal 90,90 --no-animate
"""

import argparse
import math, time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from core.map_io import load_map
from core.formations import parse_formations
from core.receding_horizon import plan_receding_horizon, sense_disc
from config import da_astar as cfg


# Demo-specific knobs.  SENSOR_FACTOR must be ≥ 2 (safety floor).
SENSOR_FACTOR  = 4.0
SENSOR_RADIUS  = SENSOR_FACTOR * ((cfg.RF * cfg.S_MAX) + cfg.RB)
MAX_ITERATIONS = 200


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _snap_to_grid(px, py, xy_step, dist_map, rb, rf, s_min):
    H, W = dist_map.shape
    n_ix = (W - 1) // xy_step + 1
    n_iy = (H - 1) // xy_step + 1
    ix0 = int(round(px / xy_step))
    iy0 = int(round(py / xy_step))
    for ring in range(11):
        best, best_d2 = None, float('inf')
        for dix in range(-ring, ring + 1):
            for diy in range(-ring, ring + 1):
                if max(abs(dix), abs(diy)) != ring:
                    continue
                ix, iy = ix0 + dix, iy0 + diy
                if ix < 0 or ix >= n_ix or iy < 0 or iy >= n_iy:
                    continue
                cx, cy = ix * xy_step, iy * xy_step
                if cx >= W or cy >= H:
                    continue
                if dist_map[cy, cx] < rb + rf * s_min:
                    continue
                d2 = (cx - px) ** 2 + (cy - py) ** 2
                if d2 < best_d2:
                    best, best_d2 = (ix, iy), d2
        if best is not None:
            return best
    return None


def _cluster_slots_chords(config_rad, cluster_def, rf):
    """Constant-chord cluster geometry: every robot lies on the rf·s
    circle and its chord to the slot direction is invariant."""
    n = len(config_rad)
    TWO_PI = 2.0 * math.pi
    slots = np.zeros(n, dtype=float)
    chord = np.zeros(n, dtype=float)
    side = np.zeros(n, dtype=float)
    for grp in cluster_def:
        grp = np.asarray(grp, dtype=int)
        angles = np.asarray(config_rad, dtype=float)[grp]
        slot = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
        for k in grp:
            d_ang = ((config_rad[k] - slot + math.pi) % TWO_PI) - math.pi
            slots[k] = slot
            chord[k] = 2.0 * rf * abs(math.sin(d_ang / 2.0))
            side[k] = 1.0 if d_ang >= 0.0 else -1.0
    return slots, chord, side


def formation_positions(x, y, theta, scale, config_rad, rf, cluster_def):
    slots, chord, side = _cluster_slots_chords(config_rad, cluster_def, rf)
    r = rf * scale
    ratio = np.clip(chord / (2.0 * r), -1.0, 1.0)
    delta = side * 2.0 * np.arcsin(ratio)
    a = slots + theta + delta
    return np.column_stack([x + r * np.cos(a),
                            y + r * np.sin(a)])


def path_to_continuous(path, xy_step, n_theta, s_values):
    th_unit = 2.0 * math.pi / n_theta
    return [(ix * xy_step, iy * xy_step, it * th_unit,
             s_values[js], int(ic))
            for ix, iy, it, js, ic in path]


def _precompute_disc_offsets(radius):
    r = int(math.ceil(radius))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    mask = ys * ys + xs * xs <= radius * radius
    return ys[mask].astype(np.int64), xs[mask].astype(np.int64)


# ═══════════════════════════════════════════════════════════
#  Interactive point selection
# ═══════════════════════════════════════════════════════════

def pick_start_goal(img, dist_map):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_title("Click START position", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()

    clicks = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        px, py = event.xdata, event.ydata
        cell = _snap_to_grid(px, py, cfg.XY_STEP, dist_map,
                             float(getattr(cfg, 'RB_PLAN', cfg.RB)),
                             cfg.RF, cfg.S_MIN)
        if cell is None:
            print(f"  No free cell near ({px:.0f}, {py:.0f}) — try again.")
            return
        ix, iy = cell
        sx, sy = ix * cfg.XY_STEP, iy * cfg.XY_STEP
        clicks.append(cell)

        if len(clicks) == 1:
            ax.plot(sx, sy, 'o', color='lime', markersize=12, zorder=5)
            ax.set_title("Click GOAL position", fontsize=14)
            print(f"  Start: pixel ({sx}, {sy})  grid ({ix}, {iy})")
            fig.canvas.draw_idle()
        elif len(clicks) == 2:
            ax.plot(sx, sy, 'o', color='red', markersize=12, zorder=5)
            ax.set_title("Selection done — closing …", fontsize=14)
            print(f"  Goal:  pixel ({sx}, {sy})  grid ({ix}, {iy})")
            fig.canvas.draw_idle()
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if len(clicks) < 2:
        return None, None
    s_mid = cfg.N_S // 2
    start = (clicks[0][0], clicks[0][1], 0, s_mid, 0)
    goal  = (clicks[1][0], clicks[1][1], 0, s_mid, 0)
    return start, goal


# ═══════════════════════════════════════════════════════════
#  Animation
# ═══════════════════════════════════════════════════════════

def animate(img, true_occ, cont, formations_rad, clusters,
            sensor_radius, status):
    """Animate the executed trajectory with a growing known overlay.

    The overlay re-simulates sensing at every animated frame; the
    planner itself sensed only at commit boundaries, so the displayed
    known region is slightly denser than the planner's.
    """
    n_robots = len(formations_rad[0])

    H, W = img.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(img, cmap='gray', origin='upper', alpha=0.35)

    known_mask = np.zeros((H, W), dtype=bool)
    known_occ = np.zeros((H, W), dtype=bool)
    disc_offsets = _precompute_disc_offsets(sensor_radius)

    overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
    im_known = ax.imshow(overlay_rgba, origin='upper', zorder=2)

    xs = [c[0] for c in cont]
    ys = [c[1] for c in cont]
    traj_line, = ax.plot([], [], '-', color='deepskyblue',
                         linewidth=1.8, alpha=0.9, zorder=3)
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=8, zorder=5)
    ax.plot(xs[-1], ys[-1], 'o', color='red', markersize=8, zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_robots, 3)))[:n_robots]
    circles = []
    for k in range(n_robots):
        c = plt.Circle((0, 0), cfg.RB, color=colors[k], alpha=0.9, zorder=10)
        ax.add_patch(c)
        circles.append(c)

    fc = plt.Circle((0, 0), cfg.RF, fill=False, edgecolor='white',
                    linestyle='--', linewidth=1, alpha=0.5, zorder=9)
    ax.add_patch(fc)

    sc = plt.Circle((0, 0), sensor_radius, fill=False,
                    edgecolor='yellow', linestyle=':', linewidth=1.3,
                    alpha=0.6, zorder=9)
    ax.add_patch(sc)

    cdot, = ax.plot([], [], 'x', color='white', markersize=6, zorder=11)

    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                  fontsize=10, color='white', va='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    def update(frame):
        x, y, theta, scale, ic = cont[frame]

        sense_disc(known_mask, known_occ, true_occ,
                   (int(round(x)), int(round(y))), sensor_radius,
                   disc_offsets)
        overlay_rgba.fill(0.0)
        free_here = known_mask & ~known_occ
        overlay_rgba[free_here] = (1.0, 1.0, 1.0, 0.35)
        overlay_rgba[known_occ] = (0.85, 0.10, 0.10, 0.75)
        im_known.set_data(overlay_rgba)

        traj_line.set_data(xs[:frame + 1], ys[:frame + 1])

        pos = formation_positions(x, y, theta, scale,
                                  formations_rad[ic], cfg.RF, clusters[ic])
        for k in range(n_robots):
            circles[k].center = (pos[k, 0], pos[k, 1])
        fc.center = (x, y)
        fc.set_radius(cfg.RF * scale)
        sc.center = (x, y)
        cdot.set_data([x], [y])

        txt.set_text(
            f"step {frame}/{len(cont)-1}  "
            f"cfg={ic}  θ={math.degrees(theta):.0f}°  s={scale:.2f}\n"
            f"status: {status}    sensor R={sensor_radius:.0f} px"
        )
        return [im_known, traj_line] + circles + [fc, sc, cdot, txt]

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title("Receding-Horizon Formation Planning", fontsize=14)

    anim = FuncAnimation(fig, update, frames=len(cont),
                         interval=cfg.INTERVAL, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def _parse_cell(text):
    """Parse 'IX,IY' into a grid-cell (ix, iy) tuple."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected IX,IY (grid cells)")
    return int(parts[0]), int(parts[1])


def run_receding(occ, start, goal, true_height=None, verbose=True):
    """Run the receding-horizon loop with this demo's settings.

    Shared with the video renderer so both drive exactly the same
    planner call.  ``true_height`` enables the payload / cable height
    check (heights are then revealed by the same sensor disc).
    """
    return plan_receding_horizon(
        occ, start, goal,
        sensor_radius=SENSOR_RADIUS,
        rb=float(getattr(cfg, 'RB_PLAN', cfg.RB)), rf=cfg.RF,
        formations_deg=cfg.FORMATIONS_DEG,
        xy_step=cfg.XY_STEP, n_theta=cfg.N_THETA,
        s_min=cfg.S_MIN, s_max=cfg.S_MAX, n_s=cfg.N_S,
        w_move=cfg.W_MOVE, w_rot=cfg.W_ROT,
        w_scale=cfg.W_SCALE, w_config=cfg.W_CONFIG,
        c_deform=cfg.C_DEFORM,
        reconfig_check=cfg.RECONFIG_CHECK,
        n_arc_samples=cfg.N_ARC_SAMPLES,
        use_symmetry=cfg.USE_SYMMETRY,
        free_theta=cfg.FREE_THETA,
        free_s=cfg.FREE_S,
        free_config=cfg.FREE_CONFIG,
        true_height=true_height,
        L_pole=cfg.L_POLE if true_height is not None else None,
        L_rope=cfg.L_ROPE if true_height is not None else None,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        max_iterations=MAX_ITERATIONS,
        verbose=verbose,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive receding-horizon planning demo "
                    "(DA_astar inner planner)")
    ap.add_argument('--start', type=_parse_cell, default=None,
                    metavar='IX,IY',
                    help='start grid cell — skips the interactive '
                         'selection (use together with --goal)')
    ap.add_argument('--goal', type=_parse_cell, default=None,
                    metavar='IX,IY', help='goal grid cell')
    ap.add_argument('--no-animate', action='store_true',
                    help='skip the final animation (headless runs)')
    args = ap.parse_args(argv)

    print("Receding-Horizon Interactive Demo")
    print(f"  Map:            {cfg.MAP_PATH}  |  c_deform: {cfg.C_DEFORM}")
    print(f"  Sensor radius:  {SENSOR_RADIUS:.0f} px  "
          f"(sensor_factor={SENSOR_FACTOR} · RF · S_MAX)")

    img, occ, dist_map = load_map(cfg.MAP_PATH, cfg.OBS_THRESH)
    s_values = np.linspace(cfg.S_MIN, cfg.S_MAX, cfg.N_S)
    formations_rad, clusters, _ = parse_formations(cfg.FORMATIONS_DEG)

    if args.start is not None and args.goal is not None:
        s_mid = cfg.N_S // 2
        start = (args.start[0], args.start[1], 0, s_mid, 0)
        goal = (args.goal[0], args.goal[1], 0, s_mid, 0)
    else:
        print("\n  Click on the map to select START and GOAL …")
        start, goal = pick_start_goal(img, dist_map)
        if start is None:
            print("  Selection cancelled.")
            return
    print(f"\n  Start: {start}")
    print(f"  Goal:  {goal}")

    print("\n  Running receding-horizon plan …")
    t0 = time.perf_counter()
    result = run_receding(occ, start, goal)
    dt = time.perf_counter() - t0

    print(f"\n  Status:          {result.status}")
    print(f"  Trajectory len:  {len(result.trajectory)} states")
    print(f"  Iterations:      {len(result.expansions)}")
    if result.expansions:
        print(f"  Total expanded:  {sum(result.expansions):,}")
        print(f"  Mean expanded:   "
              f"{sum(result.expansions) / len(result.expansions):,.0f}")
    print(f"  Wall time:       {dt:.3f}s")

    if args.no_animate or len(result.trajectory) < 2:
        if len(result.trajectory) < 2:
            print("\n  Trajectory has no motion — nothing to animate.")
        return

    cont = path_to_continuous(result.trajectory, cfg.XY_STEP,
                              cfg.N_THETA, s_values)
    animate(img, occ, cont, formations_rad, clusters,
            SENSOR_RADIUS, result.status)


if __name__ == '__main__':
    main()

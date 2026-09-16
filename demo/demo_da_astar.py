#!/usr/bin/env python3
"""Demo — Interactive DA_astar path planning with animation.

Click the map to set start (green) and goal (red), then watch the
formation move along the planned path.  Alternatively pass
``--start IX,IY --goal IX,IY`` (grid cells) to skip the interactive
selection, and ``--no-animate`` to skip the final animation.

Parameters that affect planner output live in
:mod:`config.bilateral_only`.

Usage (from the repo root)::

    python -m demo.demo_da_astar
    python -m demo.demo_da_astar --start 118,297 --goal 153,74 --no-animate
"""

import argparse
import math, time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from core.map_io import load_map, load_height_map
from core.formations import parse_formations, precompute_offsets
from core.da_astar import (find_path_da_from_map, _check_free,
                           _offsets_to_array, warmup as warmup_da)
from config import da_astar as cfg


# Colormap for obstacle height when the payload check is active:
# low obstacles green, tall obstacles red.
_HEIGHT_CMAP = plt.cm.RdYlGn_r
# Zero-height obstacles (pits — blocked on the ground map but passable
# beneath the payload) get a neutral grey, distinct from white free space.
_PIT_COLOR = (0.6, 0.6, 0.6)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _height_display(occ, height_map, height_max):
    """Build an RGB image colouring obstacles by height (green→red).

    Free space → white; obstacles with height ``1..height_max`` →
    green→red colormap; zero-height obstacles (pits) → neutral grey.
    ``occ`` is needed because a pit reads ``0`` in the height map,
    indistinguishable from free space without it.

    Returns ``(rgb, norm)`` — the ``(H, W, 3)`` image and the
    matplotlib Normalize, for a matching colorbar.
    """
    tall = height_map > 0                 # obstacles with a real height
    pit = occ & (height_map == 0)         # zero-height obstacles
    norm = Normalize(vmin=0, vmax=max(int(height_max), 1))
    rgb = np.ones((*height_map.shape, 3), dtype=float)   # white background
    if tall.any():
        colored = _HEIGHT_CMAP(norm(height_map[tall].astype(float)))
        rgb[tall] = colored[:, :3]
    rgb[pit] = _PIT_COLOR
    return rgb, norm

def _snap_to_grid(px, py, xy_step, dist_map, rb, offsets_arr, js_start):
    """Snap pixel (px, py) to the nearest grid cell the planner accepts.

    Uses the same validity test the planner applies to the start state
    (a full per-robot ``_check_free`` for config 0, θ=0, scale
    ``js_start``), so the snapped cell is never rejected as blocked.
    """
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
                cx, cy = int(ix * xy_step), int(iy * xy_step)
                if cx >= W or cy >= H:
                    continue
                if not _check_free(cx, cy, offsets_arr, 0, 0, js_start,
                                   dist_map, rb):
                    continue
                d2 = (cx - px) ** 2 + (cy - py) ** 2
                if d2 < best_d2:
                    best, best_d2 = (ix, iy), d2
        if best is not None:
            return best
    return None


def _cluster_slots_chords(config_rad, cluster_def, rf):
    """Per-robot (slot_angle, chord_to_slot, side) for a clustered config."""
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
    """(N, 2) robot positions on the rf·s circle, chord preserved across scales."""
    slots, chord, side = _cluster_slots_chords(config_rad, cluster_def, rf)
    r = rf * scale
    ratio = np.clip(chord / (2.0 * r), -1.0, 1.0)
    delta = side * 2.0 * np.arcsin(ratio)
    a = slots + theta + delta
    return np.column_stack([x + r * np.cos(a),
                            y + r * np.sin(a)])


def path_to_continuous(path, xy_step, n_theta, s_values):
    """Grid-index path → continuous (x, y, θ, scale, ic)."""
    th_unit = 2.0 * math.pi / n_theta
    return [(ix * xy_step, iy * xy_step, it * th_unit,
             s_values[js], int(ic))
            for ix, iy, it, js, ic in path]


# ═══════════════════════════════════════════════════════════
#  Interactive point selection
# ═══════════════════════════════════════════════════════════

def _show_base(ax, base_img):
    """imshow the base map, handling both grayscale and RGB inputs."""
    if base_img.ndim == 3:
        ax.imshow(base_img, origin='upper')
    else:
        ax.imshow(base_img, cmap='gray', origin='upper')


def _add_height_colorbar(fig, ax, norm):
    """Attach a green→red height colorbar to ``ax`` (no-op if norm None)."""
    if norm is None:
        return
    sm = ScalarMappable(norm=norm, cmap=_HEIGHT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Obstacle height", fontsize=10)


def pick_start_goal(base_img, dist_map, offsets_arr, js_start,
                    height_norm=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    _show_base(ax, base_img)
    _add_height_colorbar(fig, ax, height_norm)
    ax.set_title("Click START position", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()

    clicks = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        px, py = event.xdata, event.ydata
        cell = _snap_to_grid(px, py, cfg.XY_STEP, dist_map,
                             plan_radius(), offsets_arr, js_start)
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

def animate_path(base_img, cont, formations_rad, clusters, height_norm=None):
    n_robots = len(formations_rad[0])
    fig, ax = plt.subplots(figsize=(8, 8))
    _show_base(ax, base_img)
    _add_height_colorbar(fig, ax, height_norm)
    ax.set_axis_off()

    xs = [c[0] for c in cont]
    ys = [c[1] for c in cont]
    ax.plot(xs, ys, '-', color='deepskyblue', linewidth=1.5, alpha=0.6)
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=8, zorder=5)
    ax.plot(xs[-1], ys[-1], 'o', color='red', markersize=8, zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_robots, 3)))[:n_robots]
    circles = []
    for k in range(n_robots):
        c = plt.Circle((0, 0), cfg.RB, color=colors[k], alpha=0.8, zorder=10)
        ax.add_patch(c)
        circles.append(c)

    fc = plt.Circle((0, 0), cfg.RF, fill=False, edgecolor='white',
                    linestyle='--', linewidth=1, alpha=0.5, zorder=9)
    ax.add_patch(fc)
    cdot, = ax.plot([], [], 'x', color='white', markersize=6, zorder=11)

    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                  fontsize=10, color='white', va='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    def update(frame):
        x, y, theta, scale, ic = cont[frame]
        pos = formation_positions(x, y, theta, scale,
                                  formations_rad[ic], cfg.RF, clusters[ic])
        for k in range(n_robots):
            circles[k].center = (pos[k, 0], pos[k, 1])
        fc.center = (x, y)
        fc.set_radius(cfg.RF * scale)
        cdot.set_data([x], [y])
        txt.set_text(f"step {frame}/{len(cont)-1}  "
                     f"cfg={ic}  θ={math.degrees(theta):.0f}°  "
                     f"s={scale:.2f}")
        return circles + [fc, cdot, txt]

    ax.set_title("DA_astar — Formation Path", fontsize=14)
    anim = FuncAnimation(fig, update, frames=len(cont),
                         interval=cfg.INTERVAL, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def _parse_cell(text):
    """Parse 'IX,IY' into a grid-cell (ix, iy) tuple."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected IX,IY (grid cells)")
    return int(parts[0]), int(parts[1])


def plan_radius():
    """Robot radius the planner uses — inflated when the config asks.

    ``RB`` is the true body radius (what gets drawn); ``RB_PLAN``, when
    present, is the slightly larger radius the search runs with, to
    absorb the discretisation error.
    """
    return float(getattr(cfg, 'RB_PLAN', cfg.RB))


def run_da(start, goal):
    """Run the planner with this demo's settings (shared with the video).

    Returns ``(path, cost, n_exp, t_prep, t_search)``.
    """
    return find_path_da_from_map(
        cfg.MAP_PATH, start, goal,
        split_timing=True,
        obs_thresh=cfg.OBS_THRESH,
        rb=plan_radius(), rf=cfg.RF,
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
        height_map_path=cfg.HEIGHT_MAP_PATH,
        L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        height_max=cfg.HEIGHT_MAX,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive DA_astar path planning demo")
    ap.add_argument('--start', type=_parse_cell, default=None,
                    metavar='IX,IY',
                    help='start grid cell — skips the interactive '
                         'selection (use together with --goal)')
    ap.add_argument('--goal', type=_parse_cell, default=None,
                    metavar='IX,IY', help='goal grid cell')
    ap.add_argument('--no-animate', action='store_true',
                    help='skip the final animation (headless runs)')
    args = ap.parse_args(argv)

    print("DA_astar Interactive Demo")
    print(f"  Map:          {cfg.MAP_PATH}")
    print(f"  c_deform:     {cfg.C_DEFORM}")
    print(f"  use_symmetry: {cfg.USE_SYMMETRY}")

    # ── JIT warmup (one-time compilation cost, timed separately) ──
    t = time.perf_counter()
    warmup_da(formations_deg=cfg.FORMATIONS_DEG, rb=plan_radius(), rf=cfg.RF,
              n_theta=cfg.N_THETA, n_s=cfg.N_S,
              s_min=cfg.S_MIN, s_max=cfg.S_MAX,
              xy_step=cfg.XY_STEP, use_symmetry=cfg.USE_SYMMETRY)
    print(f"  JIT warmup:   {time.perf_counter() - t:.2f} s")

    img, occ, dist_map = load_map(cfg.MAP_PATH, cfg.OBS_THRESH)
    s_values = np.linspace(cfg.S_MIN, cfg.S_MAX, cfg.N_S)
    formations_rad, clusters, _ = parse_formations(cfg.FORMATIONS_DEG)

    # With the payload check active, colour obstacles by height;
    # otherwise show the grayscale occupancy map.
    payload_on = cfg.HEIGHT_MAP_PATH is not None
    height_norm = None
    if payload_on:
        height_map = load_height_map(cfg.HEIGHT_MAP_PATH,
                                     max_height=cfg.HEIGHT_MAX)
        base_img, height_norm = _height_display(occ, height_map,
                                                cfg.HEIGHT_MAX)
        print(f"  payload check: ON — obstacles coloured by height "
              f"(0→green, {cfg.HEIGHT_MAX}→red; pits in grey)")
    else:
        base_img = img
        print("  payload check: OFF — grayscale obstacles")

    # Robot offsets so the click-snap can use the planner's own
    # per-robot collision check (config 0, θ=0, start scale).  Only
    # index it=0 is read, so the full-θ period is fine here.
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])
    periods = [cfg.N_THETA] * n_config
    off_dict = precompute_offsets(formations_rad, cfg.RF, cfg.N_THETA,
                                  s_values, periods, clusters)
    offsets_arr = _offsets_to_array(off_dict, n_config, cfg.N_THETA,
                                    cfg.N_S, n_robots)
    offsets_c = np.ascontiguousarray(offsets_arr, dtype=np.int32)
    dist_c = np.ascontiguousarray(dist_map, dtype=np.float64)
    js_start = cfg.N_S // 2

    if args.start is not None and args.goal is not None:
        s_mid = cfg.N_S // 2
        start = (args.start[0], args.start[1], 0, s_mid, 0)
        goal = (args.goal[0], args.goal[1], 0, s_mid, 0)
    else:
        print("\n  Click on the map to select START and GOAL …")
        start, goal = pick_start_goal(base_img, dist_c, offsets_c, js_start,
                                      height_norm=height_norm)
        if start is None:
            print("  Selection cancelled.")
            return
    print(f"\n  Start: {start}")
    print(f"  Goal:  {goal}")
    # Grid cells (ix, iy) — reusable as --start/--goal to replay this
    # exact query non-interactively.
    print(f"  → replay: --start {start[0]},{start[1]} "
          f"--goal {goal[0]},{goal[1]}")

    print(f"\n  Running DA_astar …")
    path, cost, n_exp, t_prep, t_search = run_da(start, goal)

    if path is None:
        print("  No path found.")
        return

    print(f"  Cost: {cost:.2f}  |  Steps: {len(path)}  "
          f"|  Expanded: {n_exp:,}")
    print(f"\n  Scene preparation: {t_prep*1000:8.1f} ms")
    print(f"  Find path (A*):    {t_search*1000:8.1f} ms")

    if args.no_animate:
        return
    cont = path_to_continuous(path, cfg.XY_STEP, cfg.N_THETA, s_values)
    animate_path(base_img, cont, formations_rad, clusters,
                 height_norm=height_norm)


if __name__ == '__main__':
    main()

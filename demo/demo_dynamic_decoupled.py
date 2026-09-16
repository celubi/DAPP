#!/usr/bin/env python3
"""Demo — Dynamic A* with *decoupled* one-axis-at-a-time actions.

The action model is hardcoded to:

* ``decoupled_actions = True`` — every dynamic-A* action changes
  exactly one of (path index, θ-offset, scale-offset).
* per-axis dt costs (``dt_move``, ``dt_rot``, ``dt_scale``) applied
  uniformly to the spatial baseline and to the dynamic search, so the
  obstacle generator and ``stitch_plan`` share one timing model.
* ``deduplicate = False`` in :func:`extract_unique_waypoints` —
  each spatial step keeps its own dt, so in-place rotations/scales
  keep their duration in the obstacle timing.

Pipeline
--------
1. Click the map for START and GOAL (or pass ``--start IX,IY
   --goal IX,IY`` in grid cells; ``--no-animate`` skips step 5).
2. Run DA_astar (spatial) to get a 5-tuple path.
3. Generate intercepting obstacle(s) calibrated to the spatial schedule.
4. Run Dynamic A* (windowed, decoupled actions) to re-plan timing
   and small θ/scale offsets around the obstacle window.
5. Animate the formation, the moving obstacles and the naïve-ghost.

Usage (from the repo root)::

    python -m demo.demo_dynamic_decoupled
    python -m demo.demo_dynamic_decoupled --start 20,20 --goal 180,180 --no-animate
"""

import argparse
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from core.map_io import load_map, load_height_map
from core.formations import (parse_formations, precompute_offsets,
                             compute_h_payload, precompute_cable_offsets)
from core.obstacles import (generate_obstacle_cluster,
                            generate_obstacle_cluster_to_robot)
from core.da_astar import find_path_da_from_map
from core.dynamic import (
    extract_unique_waypoints,
    compute_path_static_free,
    compute_cumulative_dist,
    compute_wp_dt_offsets,
    prepare_obstacle_table,
    dynamic_astar_windowed,
)
from config import dynamic_decoupled as cfg


# ═══════════════════════════════════════════════════════════
#  Geometry helpers (cluster-aware formation rendering)
# ═══════════════════════════════════════════════════════════

def _cluster_slots_chords(config_rad, cluster_def, rf):
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


# ═══════════════════════════════════════════════════════════
#  Interactive point selection
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
                             cfg.RB, cfg.RF, cfg.S_MIN)
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
    goal = (clicks[1][0], clicks[1][1], 0, s_mid, 0)
    return start, goal


# ═══════════════════════════════════════════════════════════
#  Animation
# ═══════════════════════════════════════════════════════════

def animate_dynamic(img, waypoints_px, plan, wp_configs, wp_dt_offsets,
                    formations_rad, clusters, s_values,
                    obs_table, obs_radii, time_step,
                    cable_offsets=None, cable_counts=None):
    n_robots = len(formations_rad[0])
    H, W = img.shape[:2]
    show_cables = cable_offsets is not None

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_autoscale_on(False)

    xs = [waypoints_px[s[0], 0] for s in plan]
    ys = [waypoints_px[s[0], 1] for s in plan]
    ax.plot(waypoints_px[:, 0], waypoints_px[:, 1],
            '-', color='deepskyblue', linewidth=1.0, alpha=0.4,
            label='spatial path')
    ax.plot(xs[0], ys[0], 'o', color='lime', markersize=10, zorder=5)
    ax.plot(xs[-1], ys[-1], 'o', color='red', markersize=10, zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_robots, 3)))[:n_robots]
    live_circles = [plt.Circle((0, 0), cfg.RB, color=colors[k],
                               alpha=0.85, zorder=10)
                    for k in range(n_robots)]
    ghost_circles = [plt.Circle((0, 0), cfg.RB, color='orange',
                                alpha=0.25, linewidth=0, zorder=4)
                     for _ in range(n_robots)]
    for c in live_circles + ghost_circles:
        ax.add_patch(c)

    fc = plt.Circle((0, 0), cfg.RF, fill=False, edgecolor='white',
                    linestyle='--', linewidth=1, alpha=0.6, zorder=9)
    ax.add_patch(fc)

    if show_cables:
        cable_lc = LineCollection([[(0, 0), (0, 0)]] * n_robots,
                                   colors='yellow', linewidths=1.2,
                                   alpha=0.9, zorder=11)
        ax.add_collection(cable_lc)
        sample_scatter = ax.scatter([], [], s=4, c='yellow',
                                     edgecolors='black', linewidths=0.2,
                                     zorder=12)
    else:
        cable_lc = None
        sample_scatter = None

    n_obs = obs_table.shape[0]
    obs_patches = []
    for i in range(n_obs):
        cf = plt.Circle((-999, -999), obs_radii[i],
                        color='red', alpha=0.55, linewidth=0, zorder=7)
        ce = plt.Circle((-999, -999), obs_radii[i], fill=False,
                        edgecolor='darkred', linewidth=1.5, zorder=8)
        ax.add_patch(cf)
        ax.add_patch(ce)
        obs_patches.append((cf, ce))

    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                  fontsize=10, color='white', va='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    n_wp = len(waypoints_px)
    wp_dt_offsets = np.asarray(wp_dt_offsets)
    th_unit = 2.0 * math.pi / cfg.N_THETA
    obs_t_max = obs_table.shape[1]

    def update(frame):
        p, ith, isc, ic, t = plan[frame]
        cx, cy = waypoints_px[p]
        theta = ith * th_unit
        scale = s_values[isc]

        pos = formation_positions(
            cx, cy, theta, scale,
            formations_rad[ic], cfg.RF, clusters[ic])
        for k in range(n_robots):
            live_circles[k].center = (pos[k, 0], pos[k, 1])
        fc.center = (cx, cy)
        fc.set_radius(cfg.RF * scale)

        if show_cables:
            cable_lc.set_segments(
                [[(cx, cy), (pos[k, 0], pos[k, 1])] for k in range(n_robots)])
            K = int(cable_counts[isc])
            sx = cx + cable_offsets[ic, ith, isc, :K, 0]
            sy = cy + cable_offsets[ic, ith, isc, :K, 1]
            sample_scatter.set_offsets(np.column_stack([sx, sy]))

        # The ghost advances on the *naive* schedule, so it must be
        # indexed by the current time t — not by the frame number:
        # frames and time steps diverge whenever dt_rot / dt_scale != 1.
        gp = int(np.searchsorted(wp_dt_offsets, t, side='right')) - 1
        gp = max(0, min(gp, n_wp - 1))
        git, gjs, gic = wp_configs[gp]
        gx, gy = waypoints_px[gp]
        gtheta = git * th_unit
        gscale = s_values[gjs]
        gpos = formation_positions(
            gx, gy, gtheta, gscale,
            formations_rad[gic], cfg.RF, clusters[gic])
        for k in range(n_robots):
            ghost_circles[k].center = (gpos[k, 0], gpos[k, 1])

        for i in range(n_obs):
            if 0 <= t < obs_t_max:
                ox = obs_table[i, t, 0]
                oy = obs_table[i, t, 1]
                if math.isnan(ox):
                    cf, ce = obs_patches[i]
                    cf.center = (-999, -999)
                    ce.center = (-999, -999)
                    continue
                cf, ce = obs_patches[i]
                cf.center = (ox, oy)
                ce.center = (ox, oy)
            else:
                cf, ce = obs_patches[i]
                cf.center = (-999, -999)
                ce.center = (-999, -999)

        txt.set_text(
            f"frame {frame:3d}/{len(plan)-1}  "
            f"t={t:3d} ({t * time_step:.1f}s)  "
            f"p={p:3d}  cfg={ic}  θ={math.degrees(theta):.0f}°  "
            f"s={scale:.2f}")
        extra = [cable_lc, sample_scatter] if show_cables else []
        return (live_circles + ghost_circles
                + [fc, txt] + extra
                + [c for pair in obs_patches for c in pair])

    ax.set_title("Dynamic A* (decoupled actions) — formation vs. moving obstacles",
                 fontsize=13)
    anim = FuncAnimation(fig, update, frames=len(plan),
                         interval=cfg.INTERVAL, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim




# ═══════════════════════════════════════════════════════════
#  Pipeline (shared with the video renderer)
# ═══════════════════════════════════════════════════════════

def _parse_cell(text):
    """Parse 'IX,IY' into a grid-cell (ix, iy) tuple."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected IX,IY (grid cells)")
    return int(parts[0]), int(parts[1])


def resolve_start_goal(img, dist_map, start_cell=None, goal_cell=None):
    """Turn grid cells (or interactive clicks) into 5-tuple states."""
    s_mid = cfg.N_S // 2
    if start_cell is not None and goal_cell is not None:
        return ((start_cell[0], start_cell[1], 0, s_mid, 0),
                (goal_cell[0], goal_cell[1], 0, s_mid, 0))
    print("\n  Click on the map to select START and GOAL …")
    return pick_start_goal(img, dist_map)


def build_scenario(img, dist_map, start, goal, verbose=True):
    """Run the whole pipeline (spatial → obstacles → dynamic A*).

    Returns a dict with everything the animation / the video renderer
    needs, or ``None`` when either search fails.
    """
    def log(*a):
        if verbose:
            print(*a)

    s_values = np.linspace(cfg.S_MIN, cfg.S_MAX, cfg.N_S)
    formations_rad, clusters, sym_orders = parse_formations(cfg.FORMATIONS_DEG)
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])
    sym_orders_eff = sym_orders if cfg.USE_SYMMETRY else [1] * n_config
    periods = [cfg.N_THETA // k for k in sym_orders_eff]

    log(f"\n  Start: {start}")
    log(f"  Goal:  {goal}")

    # ── 1) Spatial planner ────────────────────────────────
    log(f"\n▸ DA_astar (spatial) …")
    t0 = time.perf_counter()
    spatial_path, cost, n_exp = find_path_da_from_map(
        cfg.MAP_PATH, start, goal,
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
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
        verbose=False,
    )
    dt_spatial = time.perf_counter() - t0
    if spatial_path is None:
        print("  ✗ No spatial path found.")
        return None
    log(f"  ✓ {len(spatial_path)} steps, cost {cost:.2f}, "
        f"{n_exp:,} expansions ({dt_spatial:.2f}s)")

    # ── 2) Waypoints, offsets, static-free LUT ────────────
    waypoints_px, wp_configs, _ = extract_unique_waypoints(
        spatial_path, cfg.XY_STEP, deduplicate=False)
    n_wp = len(waypoints_px)
    cum_dist = compute_cumulative_dist(waypoints_px)

    wp_dt_offsets = compute_wp_dt_offsets(
        waypoints_px, wp_configs,
        dt_move=cfg.DT_MOVE, dt_rot=cfg.DT_ROT, dt_scale=cfg.DT_SCALE)
    n_dt_total = int(wp_dt_offsets[-1])
    log(f"  {n_wp} waypoints  (path length {cum_dist[-1]:.0f} px, "
        f"{n_dt_total} dt baseline)")

    offsets = precompute_offsets(formations_rad, cfg.RF, cfg.N_THETA,
                                 s_values, periods, clusters)
    offsets_arr = np.zeros(
        (n_config, cfg.N_THETA, cfg.N_S, n_robots, 2), dtype=np.int32)
    for (ic, it, js), off in offsets.items():
        offsets_arr[ic, it, js] = off

    # ── Payload / cable height check (optional) ──────────
    if cfg.HEIGHT_MAP_PATH is not None:
        height_map = load_height_map(cfg.HEIGHT_MAP_PATH,
                                     max_height=cfg.HEIGHT_MAX)
        h_payload, js_admissible = compute_h_payload(
            s_values, cfg.RF, cfg.L_POLE, cfg.L_ROPE)
        cable_offsets, cable_counts = precompute_cable_offsets(
            formations_rad, cfg.RF, cfg.N_THETA, s_values,
            periods, clusters,
            sample_step_px=cfg.CABLE_SAMPLE_STEP_PX)
        log(f"  Payload check: ON  |  js admissible "
            f"{int(js_admissible.sum())}/{cfg.N_S}  |  "
            f"cable samples per scale {cable_counts.tolist()}")
    else:
        height_map = None
        h_payload = None
        cable_offsets = None
        cable_counts = None
        js_admissible = None

    log("▸ Computing static-free LUT along path …")
    t_pre = time.perf_counter()
    static_free = compute_path_static_free(
        waypoints_px, wp_configs, offsets_arr,
        cfg.RB, dist_map, cfg.N_THETA, cfg.N_S, periods,
        height_map=height_map, h_payload=h_payload,
        cable_offsets=cable_offsets, cable_counts=cable_counts,
        js_admissible=js_admissible)
    dt_pre = time.perf_counter() - t_pre
    sf_pct = static_free.sum() / static_free.size * 100
    log(f"  {static_free.shape}  {sf_pct:.1f}% free  ({dt_pre*1000:.1f} ms)")

    # ── 3) Generate intercepting obstacle(s) ──────────────
    full_naive_secs = (n_dt_total + 1) * cfg.TIME_STEP
    full_horizon_s = (full_naive_secs * cfg.TIME_BUDGET_FACTOR
                      + cfg.TIME_BUDGET_EXTRA)
    obs_t_max = int(math.ceil(full_horizon_s / cfg.TIME_STEP))
    wp_times = np.asarray(wp_dt_offsets, dtype=float) * cfg.TIME_STEP
    common_obs_kwargs = dict(
        base_intercept_frac=cfg.INTERCEPT_FRAC,
        base_approach_mode=cfg.APPROACH_MODE,
        base_approach_side=cfg.APPROACH_SIDE,
        base_speed=cfg.OBSTACLE_SPEED,
        base_radius=cfg.OBSTACLE_RADIUS,
        base_travel_before=cfg.OBSTACLE_TRAVEL_BEFORE,
        base_travel_after=cfg.OBSTACLE_TRAVEL_AFTER,
        n_extra=cfg.N_EXTRA_OBSTACLES,
        frac_spread=cfg.EXTRA_OBS_FRAC_SPREAD,
        speed_range=(cfg.EXTRA_OBS_SPEED_MIN, cfg.EXTRA_OBS_SPEED_MAX),
        radius_range=(cfg.EXTRA_OBS_RADIUS_MIN, cfg.EXTRA_OBS_RADIUS_MAX),
        travel_after_range=(cfg.EXTRA_OBS_TRAVEL_AFTER_MIN,
                            cfg.EXTRA_OBS_TRAVEL_AFTER_MAX),
        seed=cfg.EXTRA_OBS_SEED,
    )
    if cfg.TARGET_ROBOT is None:
        log(f"▸ Obstacle target: formation centre")
        obstacles, obs_infos = generate_obstacle_cluster(
            waypoints_px, wp_times, **common_obs_kwargs)
    else:
        log(f"▸ Obstacle target: robot #{cfg.TARGET_ROBOT}")
        obstacles, obs_infos = generate_obstacle_cluster_to_robot(
            waypoints_px, wp_times,
            wp_configs=wp_configs, offsets=offsets,
            target_robot=cfg.TARGET_ROBOT,
            **common_obs_kwargs)
    for i, oi in enumerate(obs_infos):
        tag = "main" if i == 0 else f"extra-{i}"
        tb = oi.get('travel_before', 0)
        ta = oi.get('travel_after', 0)
        tb_str = f"{tb:.0f}" if tb else "∞"
        ta_str = f"{ta:.0f}" if ta else "∞"
        log(f"▸ Obstacle {i} ({tag}): intercept step {oi['intercept_step']} "
            f"(t={oi['intercept_time']:.1f}s)  "
            f"angle={oi['approach_angle_deg']:.0f}°  "
            f"speed={oi['speed']:.1f}  radius={oi['radius']:.0f}  "
            f"travel=[{tb_str} → int → {ta_str}]px")

    obs_table, obs_radii = prepare_obstacle_table(
        obstacles, obs_t_max, cfg.TIME_STEP)

    if cable_offsets is not None and cfg.OBS_DYN_HEIGHT > 0.0:
        obs_heights = np.full(len(obs_radii), cfg.OBS_DYN_HEIGHT,
                              dtype=np.float64)
        always_on = (h_payload is not None
                     and cfg.OBS_DYN_HEIGHT > float(h_payload.max()))
        tag = "guard always True" if always_on else "height-gated"
        log(f"▸ Dynamic-obs cable check: ON  ({tag}, "
            f"OBS_DYN_HEIGHT={cfg.OBS_DYN_HEIGHT})")
    else:
        obs_heights = None

    # ── 4) Dynamic A* (windowed, decoupled actions) ───────
    log(f"\n▸ Dynamic A* (decoupled, "
        f"max_dθ={cfg.MAX_DTHETA_DEG:.0f}°, max_ds={cfg.MAX_DS_STEPS}, "
        f"margin -{cfg.WINDOW_MARGIN_BEFORE}/+{cfg.WINDOW_MARGIN_AFTER}, "
        f"backward={'yes' if cfg.ALLOW_BACKWARD else 'no'}) …")
    t_plan = time.perf_counter()
    plan, n_exp_dyn, window = dynamic_astar_windowed(
        waypoints_px, wp_configs, offsets, cum_dist,
        static_free, obs_table, obs_radii,
        cfg.RB, n_config, cfg.N_THETA, cfg.N_S, s_values, periods,
        rf=cfg.RF,
        w_path=cfg.W_PATH, w_rot=cfg.W_ROT_DYN,
        w_scale=cfg.W_SCALE_DYN, w_time=cfg.W_TIME,
        max_dtheta_deg=cfg.MAX_DTHETA_DEG,
        max_ds_steps=cfg.MAX_DS_STEPS,
        window_margin_before=cfg.WINDOW_MARGIN_BEFORE,
        window_margin_after=cfg.WINDOW_MARGIN_AFTER,
        auto_window=cfg.AUTO_WINDOW,
        allow_backward=cfg.ALLOW_BACKWARD,
        time_budget_factor=cfg.TIME_BUDGET_FACTOR,
        time_budget_extra=cfg.TIME_BUDGET_EXTRA,
        time_step=cfg.TIME_STEP,
        max_exp=cfg.MAX_EXP_DYN,
        wa_epsilon=cfg.WA_EPSILON,
        decoupled_actions=True,
        dt_move=cfg.DT_MOVE, dt_rot=cfg.DT_ROT,
        dt_scale=cfg.DT_SCALE, dt_wait=cfg.DT_WAIT,
        dt_backward_extra=0, dt_turn_extra=0,
        wp_dt_offsets=wp_dt_offsets,
        obs_heights=obs_heights,
        cable_offsets=cable_offsets,
        cable_counts=cable_counts,
        h_payload=h_payload,
        verbose=verbose,
    )
    dt_plan = time.perf_counter() - t_plan
    if plan is None:
        print(f"  ✗ No plan found ({n_exp_dyn} expansions, {dt_plan:.2f}s)")
        return None
    arrival_t = plan[-1][4]
    naive_t = int(wp_dt_offsets[-1])
    delay = arrival_t - naive_t
    log(f"  ✓ Plan: {len(plan)} frames, arrival t={arrival_t} "
        f"(naive {naive_t}, delay +{delay})  "
        f"{n_exp_dyn:,} expansions  ({dt_plan:.2f}s)")
    if window:
        log(f"  Window: [{window[0]}, {window[1]}]")

    return dict(
        start=start, goal=goal,
        spatial_ms=dt_spatial * 1000.0, plan_ms=dt_plan * 1000.0,
        waypoints_px=waypoints_px, wp_configs=wp_configs,
        wp_dt_offsets=wp_dt_offsets, cum_dist=cum_dist,
        plan=plan, window=window,
        formations_rad=formations_rad, clusters=clusters,
        s_values=s_values, periods=periods,
        obstacles=obstacles, obs_infos=obs_infos,
        obs_table=obs_table, obs_radii=obs_radii,
        cable_offsets=cable_offsets, cable_counts=cable_counts,
        h_payload=h_payload,
        arrival_t=arrival_t, naive_t=naive_t,
    )


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="DA_astar + dynamic decoupled re-planning demo")
    ap.add_argument('--start', type=_parse_cell, default=None,
                    metavar='IX,IY',
                    help='start grid cell — skips the interactive '
                         'selection (use together with --goal)')
    ap.add_argument('--goal', type=_parse_cell, default=None,
                    metavar='IX,IY', help='goal grid cell')
    ap.add_argument('--no-animate', action='store_true',
                    help='skip the final animation (headless runs)')
    args = ap.parse_args(argv)

    print("Dynamic A* — decoupled actions, dt = 1 per primitive")
    print(f"  Map: {cfg.MAP_PATH}")

    img, occ, dist_map = load_map(cfg.MAP_PATH, cfg.OBS_THRESH)
    start, goal = resolve_start_goal(img, dist_map, args.start, args.goal)
    if start is None:
        print("  Selection cancelled.")
        return

    sc = build_scenario(img, dist_map, start, goal, verbose=True)
    if sc is None:
        return

    # ── 5) Animate ────────────────────────────────────────
    if args.no_animate:
        return
    animate_dynamic(img, sc['waypoints_px'], sc['plan'], sc['wp_configs'],
                    sc['wp_dt_offsets'],
                    sc['formations_rad'], sc['clusters'], sc['s_values'],
                    sc['obs_table'], sc['obs_radii'], cfg.TIME_STEP,
                    cable_offsets=sc['cable_offsets'],
                    cable_counts=sc['cable_counts'])


if __name__ == '__main__':
    main()

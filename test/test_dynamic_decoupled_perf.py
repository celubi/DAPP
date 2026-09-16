#!/usr/bin/env python3
"""Dynamic A* (decoupled actions) — performance vs obstacle count.

Measures the dynamic re-planning phase of
:mod:`demo.demo_dynamic_decoupled` over a 3 × 4 grid:

* **3 formation configurations** — full (sym 6), triangular (sym 3),
  bilateral (sym 2);
* **4 obstacle counts** — 1 fixed + {0, 1, 2, 3} random.

12 runs in total.  The fixed obstacle is identical in every run; the
extras are randomised in speed, radius and height from a fixed seed.
The seed depends on the obstacle count only, never on the
configuration, so the three formations meet the same random draw and a
row-to-row comparison isolates the formation itself.

Each run plans its spatial path with a SINGLE formation in
``formations_deg``, so the config named in the table is the one
actually flown.  Each obstacle draws its own height in [0, 100]:
height gates only the cable / payload check on the suspended load —
robot–obstacle contact is purely geometric at any height.

Usage (from the repo root)::

    python -m test.test_dynamic_decoupled_perf
    python -m test.test_dynamic_decoupled_perf --quick   # 1 config
    python -m test.test_dynamic_decoupled_perf --plot    # + freeze-frames
    python -m test.test_dynamic_decoupled_perf --save out.png

``--plot`` freezes, for each configuration, the instant the naïve pass
is hit: the ghost sits where the old path would have put the formation
(colliding), the solid one where the re-plan puts it instead.  It reuses
the runs the table measured — nothing is re-solved.
"""

import argparse
import math
import time

from pathlib import Path

import numpy as np

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
from test._output import resolve_save


# ═══════════════════════════════════════════════════════════
#  TEST PARAMETERS
# ═══════════════════════════════════════════════════════════

_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = str(_ROOT / "random_maps" / "random_map_5.png")
HEIGHT_MAP_PATH = str(_ROOT / "random_maps" / "random_map_5_height.png")

# Grid cells, not pixels: START_PX // XY_STEP (cfg.XY_STEP = 10).
START_PX = (200, 200)
GOAL_PX = (1800, 1800)

# One formation per run — see module docstring.
CONFIGS = [
    ("full",       [[0.0], [60.0], [120.0], [180.0], [240.0], [300.0]]),
    ("triangular", [[-10.0, 10.0], [110.0, 130.0], [230.0, 250.0]]),
    ("bilateral",  [[-20.0, 0.0, 20.0], [160.0, 180.0, 200.0]]),
]

# 1 fixed obstacle + this many random ones.
N_EXTRA_LIST = [0, 1, 2, 3]

# Random obstacle heights, drawn per obstacle from the seeded RNG; the
# fixed obstacle keeps cfg.OBS_DYN_HEIGHT so it is identical in every
# run.
OBS_HEIGHT_MIN = 0.0
OBS_HEIGHT_MAX = 100.0

# Master seed.  Each (config, n_extra) pair derives its own stream from
# it, so a run's obstacles do not depend on what ran before it.
SEED = 76


def _s_mid():
    return cfg.N_S // 2


def _to_grid(px):
    return (px[0] // cfg.XY_STEP, px[1] // cfg.XY_STEP, 0, _s_mid(), 0)


# ═══════════════════════════════════════════════════════════
#  One run
# ═══════════════════════════════════════════════════════════

def run_case(cfg_name, formation_deg, n_extra, img, occ, dist_map,
             verbose=False):
    """Spatial plan + dynamic re-plan for one (config, n_extra) pair."""
    formations_deg = [formation_deg]        # single template on purpose
    formations_rad, clusters, sym_orders = parse_formations(formations_deg)
    n_config = len(formations_rad)
    n_robots = len(formations_rad[0])
    sym_eff = sym_orders if cfg.USE_SYMMETRY else [1] * n_config
    periods = [cfg.N_THETA // k for k in sym_eff]
    s_values = np.linspace(cfg.S_MIN, cfg.S_MAX, cfg.N_S)

    # ── 1) Spatial path (static map, single config) ──────────
    spatial_path, cost, n_exp = find_path_da_from_map(
        MAP_PATH, _to_grid(START_PX), _to_grid(GOAL_PX),
        obs_thresh=cfg.OBS_THRESH,
        rb=cfg.RB, rf=cfg.RF,
        formations_deg=formations_deg,
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
        height_map_path=HEIGHT_MAP_PATH,
        L_pole=cfg.L_POLE, L_rope=cfg.L_ROPE,
        cable_sample_step_px=cfg.CABLE_SAMPLE_STEP_PX,
        height_max=cfg.HEIGHT_MAX,
        verbose=False,
    )
    if spatial_path is None:
        return None

    # ── 2) Waypoints / timing baseline ──────────────────────
    waypoints_px, wp_configs, _ = extract_unique_waypoints(
        spatial_path, cfg.XY_STEP, deduplicate=False)
    cum_dist = compute_cumulative_dist(waypoints_px)
    wp_dt_offsets = compute_wp_dt_offsets(
        waypoints_px, wp_configs,
        dt_move=cfg.DT_MOVE, dt_rot=cfg.DT_ROT, dt_scale=cfg.DT_SCALE)
    n_dt_total = int(wp_dt_offsets[-1])

    offsets = precompute_offsets(formations_rad, cfg.RF, cfg.N_THETA,
                                 s_values, periods, clusters)
    offsets_arr = np.zeros(
        (n_config, cfg.N_THETA, cfg.N_S, n_robots, 2), dtype=np.int32)
    for (ic, it, js), off in offsets.items():
        offsets_arr[ic, it, js] = off

    # ── Payload / cable height structures ───────────────────
    height_map = load_height_map(HEIGHT_MAP_PATH, max_height=cfg.HEIGHT_MAX)
    h_payload, js_admissible = compute_h_payload(
        s_values, cfg.RF, cfg.L_POLE, cfg.L_ROPE)
    cable_offsets, cable_counts = precompute_cable_offsets(
        formations_rad, cfg.RF, cfg.N_THETA, s_values, periods, clusters,
        sample_step_px=cfg.CABLE_SAMPLE_STEP_PX)

    static_free = compute_path_static_free(
        waypoints_px, wp_configs, offsets_arr,
        cfg.RB, dist_map, cfg.N_THETA, cfg.N_S, periods,
        height_map=height_map, h_payload=h_payload,
        cable_offsets=cable_offsets, cable_counts=cable_counts,
        js_admissible=js_admissible)

    # ── 3) Obstacles: 1 fixed + n_extra random ──────────────
    full_naive_secs = (n_dt_total + 1) * cfg.TIME_STEP
    full_horizon_s = (full_naive_secs * cfg.TIME_BUDGET_FACTOR
                      + cfg.TIME_BUDGET_EXTRA)
    obs_t_max = int(math.ceil(full_horizon_s / cfg.TIME_STEP))
    wp_times = np.asarray(wp_dt_offsets, dtype=float) * cfg.TIME_STEP

    # Seed depends on n_extra only, NOT on the config, so every
    # formation faces the same random draw.
    case_seed = SEED + n_extra

    obs_kwargs = dict(
        base_intercept_frac=cfg.INTERCEPT_FRAC,
        base_approach_mode=cfg.APPROACH_MODE,
        base_approach_side=cfg.APPROACH_SIDE,
        base_speed=cfg.OBSTACLE_SPEED,
        base_radius=cfg.OBSTACLE_RADIUS,
        base_travel_before=cfg.OBSTACLE_TRAVEL_BEFORE,
        base_travel_after=cfg.OBSTACLE_TRAVEL_AFTER,
        n_extra=n_extra,
        frac_spread=cfg.EXTRA_OBS_FRAC_SPREAD,
        speed_range=(cfg.EXTRA_OBS_SPEED_MIN, cfg.EXTRA_OBS_SPEED_MAX),
        radius_range=(cfg.EXTRA_OBS_RADIUS_MIN, cfg.EXTRA_OBS_RADIUS_MAX),
        travel_after_range=(cfg.EXTRA_OBS_TRAVEL_AFTER_MIN,
                            cfg.EXTRA_OBS_TRAVEL_AFTER_MAX),
        seed=case_seed,
    )
    if cfg.TARGET_ROBOT is None:
        obstacles, obs_infos = generate_obstacle_cluster(
            waypoints_px, wp_times, **obs_kwargs)
    else:
        obstacles, obs_infos = generate_obstacle_cluster_to_robot(
            waypoints_px, wp_times,
            wp_configs=wp_configs, offsets=offsets,
            target_robot=cfg.TARGET_ROBOT,
            **obs_kwargs)

    obs_table, obs_radii = prepare_obstacle_table(
        obstacles, obs_t_max, cfg.TIME_STEP)

    # Heights: obstacle 0 (the fixed one) keeps the config value; the
    # extras draw their own from a separate RNG stream.
    rng = np.random.default_rng(case_seed)
    obs_heights = np.empty(len(obs_radii), dtype=np.float64)
    obs_heights[0] = cfg.OBS_DYN_HEIGHT
    if len(obs_radii) > 1:
        obs_heights[1:] = rng.uniform(OBS_HEIGHT_MIN, OBS_HEIGHT_MAX,
                                      size=len(obs_radii) - 1)

    # ── 4) Dynamic A* — the measured phase ──────────────────
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

    # ── Window geometry ─────────────────────────────────────
    # `window` is (p_start, p_end) in WAYPOINT indices; convert to the
    # quantities the table reports.
    win_dur_s = float('nan')
    win_len_px = float('nan')
    if window:
        p0, p1 = int(window[0]), int(window[1])
        p1c = min(p1, len(cum_dist) - 1)
        win_len_px = float(cum_dist[p1c] - cum_dist[p0])
        win_dur_s = float(wp_dt_offsets[p1c] - wp_dt_offsets[p0]) \
            * cfg.TIME_STEP

    return dict(
        config=cfg_name,
        n_obs=len(obs_radii),
        heights=obs_heights.copy(),
        radii=np.asarray(obs_radii, dtype=float).copy(),
        speeds=np.array([oi['speed'] for oi in obs_infos], dtype=float),
        win_dur_s=win_dur_s,
        win_len_px=win_len_px,
        dt_plan=dt_plan,
        n_exp_dyn=n_exp_dyn,
        solved=plan is not None,
        arrival=(plan[-1][4] if plan else None),
        naive=n_dt_total,
        spatial_steps=len(spatial_path),
        n_wp=len(waypoints_px),
        # Everything the freeze-frame plot needs, so it never re-solves.
        plan=plan,
        window=window,
        waypoints_px=waypoints_px,
        wp_configs=wp_configs,
        wp_dt_offsets=wp_dt_offsets,
        offsets=offsets,
        offsets_arr=offsets_arr,
        obs_table=obs_table,
        obs_radii=np.asarray(obs_radii, dtype=float),
        obs_heights_vec=obs_heights.copy(),
        formations_rad=formations_rad,
        clusters=clusters,
        s_values=s_values,
        h_payload=h_payload,
    )


# ═══════════════════════════════════════════════════════════
#  Freeze-frame plot: colliding ghost vs safe re-plan
# ═══════════════════════════════════════════════════════════

def _formation_positions(x, y, theta, scale, config_rad, rf, cluster_def):
    """Cluster-aware robot placement (same geometry as the demo)."""
    n = len(config_rad)
    TWO_PI = 2.0 * math.pi
    slots = np.zeros(n)
    chord = np.zeros(n)
    side = np.zeros(n)
    for grp in cluster_def:
        grp = np.asarray(grp, dtype=int)
        angles = np.asarray(config_rad, dtype=float)[grp]
        slot = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
        for k in grp:
            d_ang = ((config_rad[k] - slot + math.pi) % TWO_PI) - math.pi
            slots[k] = slot
            chord[k] = 2.0 * rf * abs(math.sin(d_ang / 2.0))
            side[k] = 1.0 if d_ang >= 0.0 else -1.0
    r = rf * scale
    ratio = np.clip(chord / (2.0 * r), -1.0, 1.0)
    a = slots + theta + side * 2.0 * np.arcsin(ratio)
    return np.column_stack([x + r * np.cos(a), y + r * np.sin(a)])


def _first_hit_per_obstacle(r):
    """For each obstacle, the first naïve-pass collision with IT.

    Robot–obstacle contact is purely geometric (no height guard),
    mirroring ``dynamic_collision``; height gates only the cable /
    payload check.

    Returns ``{obs_index: (p_hit, t_hit)}``; obstacles that never reach
    the ghost are absent.
    """
    wp = r['waypoints_px']
    offsets = r['offsets']
    dto = r['wp_dt_offsets']
    ot, orad = r['obs_table'], r['obs_radii']
    t_max = ot.shape[1]

    out = {}
    for i in range(len(orad)):
        for p in range(len(wp)):
            it, js, ic = r['wp_configs'][p]
            t = int(dto[p])
            if t < 0 or t >= t_max:
                continue
            ox, oy = ot[i, t, 0], ot[i, t, 1]
            if math.isnan(ox):
                continue
            offs = offsets[(int(ic), int(it), int(js))]
            thr2 = (cfg.RB + orad[i]) ** 2
            hit = False
            for k in range(offs.shape[0]):
                dx = wp[p, 0] + offs[k, 0] - ox
                dy = wp[p, 1] + offs[k, 1] - oy
                if dx * dx + dy * dy <= thr2:
                    hit = True
                    break
            if hit:
                out[i] = (p, t)
                break
    return out


def _plan_state_at_t(plan, t):
    """Where the re-plan is at clock time ``t`` (last state with t'<=t)."""
    best = None
    for st in plan:
        if st[4] <= t:
            best = st
        else:
            break
    return best


def _draw_panel(ax, img, r, i_obs, p_hit, t_hit):
    """One panel: the encounter with obstacle ``i_obs`` at time t_hit.

    Ghost and re-plan are both drawn at the same instant t_hit; the
    displacement between them is the dodge.
    """
    import matplotlib.pyplot as plt

    wp = r['waypoints_px']
    th_unit = 2.0 * math.pi / cfg.N_THETA
    fr, cl, sv = r['formations_rad'], r['clusters'], r['s_values']
    ot, orad = r['obs_table'], r['obs_radii']

    ax.imshow(img, cmap='gray', origin='upper')

    # Ghost: naïve pose at the colliding waypoint.
    git, gjs, gic = r['wp_configs'][p_hit]
    gpos = _formation_positions(wp[p_hit, 0], wp[p_hit, 1],
                                git * th_unit, sv[gjs],
                                fr[gic], cfg.RF, cl[gic])

    # Re-plan sampled at the SAME instant (not the same waypoint).
    st = _plan_state_at_t(r['plan'], t_hit) if r['plan'] else None
    npos = None
    if st is not None:
        p_n, ith_n, isc_n, ic_n, _t_n = st
        npos = _formation_positions(wp[p_n, 0], wp[p_n, 1],
                                    ith_n * th_unit, sv[isc_n],
                                    fr[ic_n], cfg.RF, cl[ic_n])

    ax.plot(wp[:, 0], wp[:, 1], '-', color='deepskyblue', lw=1.0,
            alpha=0.45, zorder=2)

    # Obstacles at t_hit: the one under test in red (labelled), the
    # others grey.
    for j in range(len(orad)):
        ox, oy = ot[j, t_hit, 0], ot[j, t_hit, 1]
        if math.isnan(ox):
            continue
        if j == i_obs:
            fc_, ec_, al, z = 'red', 'darkred', 0.50, 6
        else:
            fc_, ec_, al, z = '0.55', '0.30', 0.40, 5
        ax.add_patch(plt.Circle((ox, oy), orad[j], color=fc_, alpha=al,
                                lw=0, zorder=z))
        ax.add_patch(plt.Circle((ox, oy), orad[j], fill=False, ec=ec_,
                                lw=1.6, zorder=z + 1))
        if j == i_obs:
            ax.annotate(f"h={r['obs_heights_vec'][j]:.0f}", (ox, oy),
                        color='white', fontsize=8, ha='center',
                        va='center', zorder=z + 2, fontweight='bold')

    # The red obstacle's track through t_hit (the table is NaN outside
    # its lifetime, so scan for real samples).
    ts = [t for t in range(ot.shape[1]) if not math.isnan(ot[i_obs, t, 0])]
    if len(ts) >= 2:
        track = np.array([[ot[i_obs, t, 0], ot[i_obs, t, 1]] for t in ts])
        ax.plot(track[:, 0], track[:, 1], ':', color='darkred', lw=1.6,
                alpha=0.85, zorder=4)
        # Arrow head at the far end of the track, pointing forward.
        t_i = ts.index(t_hit) if t_hit in ts else len(ts) // 2
        j0 = max(t_i - 1, 0)
        j1 = min(t_i + 1, len(ts) - 1)
        if j1 > j0:
            p_from = track[j0]
            d = track[j1] - track[j0]
            n = math.hypot(d[0], d[1])
            if n > 1e-6:
                d = d / n * (2.0 * orad[i_obs])
                ax.annotate("", xy=(p_from[0] + d[0], p_from[1] + d[1]),
                            xytext=(p_from[0], p_from[1]),
                            arrowprops=dict(arrowstyle='-|>', lw=1.8,
                                            color='darkred', ls=':',
                                            shrinkA=0, shrinkB=0),
                            zorder=5)

    n_rob = len(fr[0])
    for k in range(n_rob):
        ax.add_patch(plt.Circle(gpos[k], cfg.RB, color='orange', alpha=0.45,
                                lw=0, zorder=9))
        ax.add_patch(plt.Circle(gpos[k], cfg.RB, fill=False, ec='darkorange',
                                lw=1.4, alpha=0.95, zorder=9))
        if npos is not None:
            ax.add_patch(plt.Circle(npos[k], cfg.RB, color='#1f77b4',
                                    alpha=0.95, ec='k', lw=0.5, zorder=11))
    ax.add_patch(plt.Circle((wp[p_hit, 0], wp[p_hit, 1]), cfg.RF * sv[gjs],
                            fill=False, ec='orange', ls='--', lw=1.0,
                            alpha=0.55, zorder=8))
    if npos is not None:
        ax.add_patch(plt.Circle((wp[p_n, 0], wp[p_n, 1]),
                                cfg.RF * sv[isc_n], fill=False, ec='#1f77b4',
                                ls='--', lw=1.0, alpha=0.7, zorder=10))

    # Crop on the collision (ghost + its obstacle); the re-plan is
    # included only when it is close enough to fit.
    focus = np.vstack([gpos,
                       np.array([[ot[i_obs, t_hit, 0],
                                  ot[i_obs, t_hit, 1]]])])
    cx_ = 0.5 * (focus[:, 0].min() + focus[:, 0].max())
    cy_ = 0.5 * (focus[:, 1].min() + focus[:, 1].max())
    half = 0.5 * max(np.ptp(focus[:, 0]), np.ptp(focus[:, 1])) \
        + orad[i_obs] + 0.7 * cfg.RF
    if npos is not None:
        nc = npos.mean(axis=0)
        if (abs(nc[0] - cx_) < 2.5 * half) and (abs(nc[1] - cy_) < 2.5 * half):
            half = max(half,
                       abs(nc[0] - cx_) + cfg.RB * 2,
                       abs(nc[1] - cy_) + cfg.RB * 2)
    # Clamp the centre, never the half-size, to keep the box square.
    H, W = img.shape[:2]
    cx_ = min(max(cx_, half), max(W - half, half))
    cy_ = min(max(cy_, half), max(H - half, half))
    ax.set_xlim(cx_ - half, cx_ + half)
    ax.set_ylim(cy_ + half, cy_ - half)
    ax.set_aspect('equal')
    ax.set_axis_off()

    dodge = ""
    if npos is not None:
        d = math.hypot(wp[p_n, 0] - wp[p_hit, 0], wp[p_n, 1] - wp[p_hit, 1])
        x0v, x1v = ax.get_xlim()
        y1v, y0v = ax.get_ylim()
        nc = npos.mean(axis=0)
        inside = (x0v <= nc[0] <= x1v) and (y0v <= nc[1] <= y1v)
        dodge = (f"  ·  re-plan {d:.0f} px away"
                 + ("" if inside else ", outside this crop"))
    ax.set_title(f"obstacle {i_obs}  (r={orad[i_obs]:.0f}, "
                 f"h={r['obs_heights_vec'][i_obs]:.0f}, "
                 f"v={r['speeds'][i_obs]:.1f})\n"
                 f"t={t_hit} ({t_hit * cfg.TIME_STEP:.1f}s), "
                 f"ghost at p={p_hit}{dodge}",
                 fontsize=9)


def plot_case(r, img, save=None):
    """One panel per obstacle: each collision frozen at its own instant.

    Failed runs get a blank FAIL panel rather than an invented pose.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_obs = len(r['obs_radii'])
    hits = _first_hit_per_obstacle(r)

    # One panel per real collision, in chronological order.
    if r['solved']:
        panels = sorted(hits.keys(), key=lambda i: hits[i][1])
    else:
        panels = []

    if not panels:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.set_axis_off()
        if not r['solved']:
            msg, col = "FAIL\nno dynamic plan", 'red'
        else:
            msg = ("no collision at all\nthe naïve pass is never "
                   "reached by any obstacle")
            col = '0.35'
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center',
                va='center', fontsize=17, fontweight='bold', color=col)
        fig.suptitle(f"{r['config']} — {n_obs} obstacles", fontsize=13)
    else:
        ncol = min(len(panels), 4)
        nrow = int(math.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(5.0 * ncol, 5.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for slot, i in enumerate(panels):
            p_hit, t_hit = hits[i]
            _draw_panel(axes[slot], img, r, i, p_hit, t_hit)
        for j in range(len(panels), len(axes)):
            axes[j].set_axis_off()

    # Obstacles that never reach the ghost get a suptitle note instead
    # of a panel.
    skipped = [i for i in range(n_obs) if i not in hits]
    skip_note = ""
    if skipped and r['solved']:
        tags = ", ".join(f"#{i}" for i in skipped)
        skip_note = (f"\nnot shown — never reach the naïve pass: {tags}")

    if panels:
        handles = [
            Line2D([], [], marker='o', ls='none', mfc='orange',
                   mec='darkorange', alpha=0.7, ms=10,
                   label='ghost — old path, COLLIDES'),
            Line2D([], [], marker='o', ls='none', color='#1f77b4', ms=10,
                   label='re-plan — same instant, clear'),
            Line2D([], [], marker='o', ls='none', mfc='red', mec='darkred',
                   alpha=0.5, ms=10, label='the obstacle it collides with'),
            Line2D([], [], ls=':', color='darkred', lw=1.8,
                   label='its trajectory'),
            Line2D([], [], marker='o', ls='none', mfc='0.55', mec='0.30',
                   alpha=0.5, ms=10, label='other obstacles'),
        ]
        delay = f"delay +{r['arrival'] - r['naive']}"
        fig.suptitle(
            f"{r['config']} — {n_obs} obstacles  (solved, {delay})  ·  "
            f"{len(panels)} real collision(s)\n"
            f"each panel freezes ONE collision at its own instant: ghost "
            f"on the old path vs the re-plan at the same clock time"
            f"{skip_note}",
            fontsize=13)
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
        fig.tight_layout(rect=[0, 0.07, 1, 0.85 if skip_note else 0.89])
    else:
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    if save:
        fig.savefig(save, dpi=125, bbox_inches='tight')
        print(f"    figure → {save}")
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def _fmt_vec(v, prec=0):
    if v is None or len(v) == 0:
        return "[]"
    return "[" + " ".join(f"{x:.{prec}f}" for x in v) + "]"


def main(quick=False, plot=False, save=None):
    configs = CONFIGS[:1] if quick else CONFIGS
    extras = N_EXTRA_LIST[:1] if quick else N_EXTRA_LIST

    print("Dynamic A* (decoupled) — performance vs obstacle count")
    print(f"  Map:          {MAP_PATH}")
    print(f"  Height map:   {HEIGHT_MAP_PATH}")
    print(f"  Start / Goal: {START_PX} → {GOAL_PX} px")
    print(f"  Configs:      {[c for c, _ in configs]} (one per run)")
    print(f"  Obstacles:    1 fixed + {extras} random")
    print(f"  Obs heights:  fixed={cfg.OBS_DYN_HEIGHT}, "
          f"random ∈ [{OBS_HEIGHT_MIN:.0f}, {OBS_HEIGHT_MAX:.0f}]")
    print(f"  Obs speeds:   fixed={cfg.OBSTACLE_SPEED:.0f}, "
          f"random ∈ [{cfg.EXTRA_OBS_SPEED_MIN:.0f}, "
          f"{cfg.EXTRA_OBS_SPEED_MAX:.0f}] px/s")
    print(f"  Seed:         {SEED} + n_extra — SAME across configs, so "
          f"full/triangular/bilateral\n                face the same "
          f"random draw (heights, radii, speeds)")
    print(f"  TIME_STEP:    {cfg.TIME_STEP}s   "
          f"W_TIME={cfg.W_TIME}  WA_EPSILON={cfg.WA_EPSILON}")
    print()

    img, occ, dist_map = load_map(MAP_PATH, cfg.OBS_THRESH)

    # Warm up the JIT once (compile time is not measured).
    print("  Warming up JIT …")
    run_case(configs[0][0], configs[0][1], 0, img, occ, dist_map)

    rows = []
    for cfg_name, formation in configs:
        for n_extra in extras:
            print(f"\n▸ {cfg_name:<11} 1 fixed + {n_extra} random …")
            r = run_case(cfg_name, formation, n_extra, img, occ, dist_map)
            if r is None:
                print(f"  ✗ no spatial path for config '{cfg_name}' — "
                      f"check START_PX / GOAL_PX.")
                continue
            if not r['solved']:
                print(f"  ✗ dynamic A* found no plan "
                      f"({r['n_exp_dyn']:,} expansions, {r['dt_plan']:.2f}s)")
            else:
                print(f"  ✓ arrival t={r['arrival']} (naive {r['naive']}, "
                      f"delay +{r['arrival'] - r['naive']})  "
                      f"{r['n_exp_dyn']:,} exp  {r['dt_plan']:.3f}s")
            print(f"    heights={_fmt_vec(r['heights'])}  "
                  f"radii={_fmt_vec(r['radii'])}  "
                  f"speeds={_fmt_vec(r['speeds'], 1)}")
            print(f"    window: {r['win_dur_s']:.1f}s  "
                  f"{r['win_len_px']:.0f} px")
            rows.append(r)

    if not rows:
        print("\n  No results.")
        return

    # ── Summary ─────────────────────────────────────────────
    print("\n\n── Summary ──")
    hdr = (f"  {'config':<11} {'n_obs':>5}  {'heights':<22} "
           f"{'radii':<20} {'speeds':<20} "
           f"{'win(s)':>7} {'win(px)':>8} {'plan(s)':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) + 6))
    for r in rows:
        flag = "" if r['solved'] else "  ✗no-plan"
        print(f"  {r['config']:<11} {r['n_obs']:>5}  "
              f"{_fmt_vec(r['heights']):<22} "
              f"{_fmt_vec(r['radii']):<20} "
              f"{_fmt_vec(r['speeds'], 1):<20} "
              f"{r['win_dur_s']:>7.1f} {r['win_len_px']:>8.0f} "
              f"{r['dt_plan']:>8.3f}{flag}")

    print("\n  config    — formation template (single, fixed per run)")
    print("  n_obs     — 1 fixed + N random")
    print("  heights   — obstacle heights; [0] is the fixed one "
          f"({cfg.OBS_DYN_HEIGHT}), rest random in "
          f"[{OBS_HEIGHT_MIN:.0f}, {OBS_HEIGHT_MAX:.0f}]")
    print("  win(s)    — replanning window duration along the baseline "
          "schedule")
    print("  win(px)   — replanning window length along the path")
    print("  plan(s)   — dynamic-phase planning time (the measured "
          "quantity)")
    print(f"\n  An obstacle blocks the payload only where its height "
          f"exceeds the\n  payload altitude (h_payload ∈ "
          f"[{_h_payload_range()}]); lower ones are flown over.")

    if plot:
        # One freeze-frame per configuration, at the busiest case.
        n_plot = max(N_EXTRA_LIST) + 1
        print(f"\n── Freeze-frame plots ({n_plot} obstacles) ──")
        for r in rows:
            # Unsolved cases are plotted too (FAIL panel).
            if r['n_obs'] != n_plot:
                continue
            out = None
            if save:
                stem, _, ext = save.rpartition('.')
                out = (f"{stem}_{r['config']}.{ext}" if stem
                       else f"{save}_{r['config']}")
            plot_case(r, img, save=out)


def _h_payload_range():
    s_values = np.linspace(cfg.S_MIN, cfg.S_MAX, cfg.N_S)
    hp, _ = compute_h_payload(s_values, cfg.RF, cfg.L_POLE, cfg.L_ROPE)
    finite = hp[np.isfinite(hp)]
    if len(finite) == 0:
        return "n/a"
    return f"{finite.min():.0f}, {finite.max():.0f}"


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='one config / one obstacle count (smoke test)')
    ap.add_argument('--plot', action='store_true',
                    help='after the table, draw one freeze-frame per '
                         'configuration at the busiest obstacle count')
    ap.add_argument('--save', type=str, default=None,
                    help='save the figures instead of showing them; the '
                         'config name is appended per file '
                         '(out.png → out_full.png, out_triangular.png, …); '
                         'relative paths land in test_output/')
    args = ap.parse_args()
    main(quick=args.quick,
         plot=args.plot or args.save is not None,
         save=resolve_save(args.save))

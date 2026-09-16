"""Dynamic obstacle generation.

A *dynamic obstacle* here is a constant-velocity disc whose trajectory
is sampled at the same time grid as the formation path.  Trajectories
are constructed so that the obstacle disc passes through (or near) a
chosen waypoint of the formation path — useful for stress-testing the
dynamic planner.

These utilities are formation-agnostic: they only depend on the
formation centre positions over time.
"""

import math
import numpy as np


def _path_tangent(positions, step):
    """Unit tangent of the path at ``step`` (central differences)."""
    n = len(positions)
    if n < 2:
        return np.array([1.0, 0.0])
    lo = max(0, step - 1)
    hi = min(n - 1, step + 1)
    d = positions[hi] - positions[lo]
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return d / norm


def compute_approach_angle(positions, step, mode="perpendicular",
                           manual_deg=0.0, side="left"):
    """Approach angle (radians) for an obstacle velocity vector.

    Parameters
    ----------
    positions : (N, 2) ndarray — formation centre path.
    step      : int            — index along the path where the
                                 obstacle should intercept.
    mode      : ``"manual"`` | ``"perpendicular"`` | ``"head_on"``
    manual_deg: angle in degrees, only used when ``mode == "manual"``.
    side      : ``"left"`` or ``"right"`` of the path tangent.
    """
    if mode == "manual":
        return math.radians(manual_deg)

    tangent = _path_tangent(positions, step)
    t_angle = math.atan2(tangent[1], tangent[0])

    if mode == "head_on":
        base = t_angle + math.pi
        offset = math.radians(15) * (1 if side == "left" else -1)
        return base + offset

    if side == "left":
        return t_angle - math.pi / 2
    return t_angle + math.pi / 2


def compute_intercept_trajectory(path_positions, path_times,
                                 intercept_step, approach_angle_rad,
                                 speed, radius=10.0,
                                 travel_before=0.0, travel_after=0.0):
    """Constant-velocity trajectory through the formation centre at
    ``intercept_step``.

    ``travel_before`` / ``travel_after`` (px) bound the obstacle's
    active window relative to the intercept point.  ``0`` (or negative)
    means "no bound" on that side — the obstacle exists for the entire
    formation traversal time.
    """
    n = len(path_positions)
    intercept_step = max(0, min(intercept_step, n - 1))
    tx, ty = path_positions[intercept_step]
    t_int = path_times[intercept_step]

    vx = speed * math.cos(approach_angle_rad)
    vy = speed * math.sin(approach_angle_rad)

    x0 = tx - vx * t_int
    y0 = ty - vy * t_int

    if travel_before and travel_before > 0 and speed > 0:
        t_start = t_int - travel_before / speed
    else:
        t_start = path_times[0]

    if travel_after and travel_after > 0 and speed > 0:
        t_end = t_int + travel_after / speed
    else:
        t_end = path_times[-1]

    mask = (path_times >= t_start) & (path_times <= t_end)
    obs_times = path_times[mask]
    obs_positions = np.column_stack([x0 + vx * obs_times,
                                     y0 + vy * obs_times])

    if len(obs_times) > 1:
        actual_before = speed * (t_int - obs_times[0])
        actual_after = speed * (obs_times[-1] - t_int)
    else:
        actual_before = 0.0
        actual_after = 0.0

    info = dict(
        intercept_step=intercept_step,
        intercept_time=t_int,
        target_xy=(tx, ty),
        x0=x0, y0=y0, vx=vx, vy=vy,
        speed=speed, radius=radius,
        approach_angle_deg=math.degrees(approach_angle_rad),
        travel_before=actual_before,
        travel_after=actual_after,
    )
    return obs_positions, obs_times, info


def _robot_target_xy(path_positions, wp_configs, offsets, step, target_robot):
    """Pixel position of the ``target_robot`` at the *baseline* pose of
    the spatial planner at ``step``.

    ``offsets`` is the precompute_offsets dict — ``{(ic, iθ, is): (n_robots, 2)}`` —
    or any object indexable as ``offsets[(ic, iθ, is)]``.  Returns
    ``(tx, ty)`` in the same pixel frame as ``path_positions``.
    """
    cx, cy = path_positions[step]
    it, js, ic = wp_configs[step]
    off = offsets[(int(ic), int(it), int(js))]
    return float(cx + off[target_robot, 0]), float(cy + off[target_robot, 1])


def compute_intercept_trajectory_to_robot(
        path_positions, path_times,
        wp_configs, offsets, target_robot,
        intercept_step, approach_angle_rad,
        speed, radius=10.0,
        travel_before=0.0, travel_after=0.0):
    """Constant-velocity trajectory through a *specific robot* of the
    formation at ``intercept_step``.

    Same back-propagation logic as
    :func:`compute_intercept_trajectory`, but the target point is the
    pixel position of ``target_robot`` under the spatial planner's
    nominal pose at ``intercept_step`` instead of the formation
    centre.  ``offsets`` is the per-config offsets dict produced by
    :func:`fapp.core.formations.precompute_offsets`.

    The approach angle is unchanged — it can still be computed from
    the path tangent (perpendicular / head-on) so the obstacle comes
    "from the side" of the path even though its impact point is one
    of the robots.
    """
    n = len(path_positions)
    intercept_step = max(0, min(intercept_step, n - 1))
    tx, ty = _robot_target_xy(
        path_positions, wp_configs, offsets, intercept_step, target_robot)
    t_int = path_times[intercept_step]

    vx = speed * math.cos(approach_angle_rad)
    vy = speed * math.sin(approach_angle_rad)

    x0 = tx - vx * t_int
    y0 = ty - vy * t_int

    if travel_before and travel_before > 0 and speed > 0:
        t_start = t_int - travel_before / speed
    else:
        t_start = path_times[0]

    if travel_after and travel_after > 0 and speed > 0:
        t_end = t_int + travel_after / speed
    else:
        t_end = path_times[-1]

    mask = (path_times >= t_start) & (path_times <= t_end)
    obs_times = path_times[mask]
    obs_positions = np.column_stack([x0 + vx * obs_times,
                                     y0 + vy * obs_times])

    if len(obs_times) > 1:
        actual_before = speed * (t_int - obs_times[0])
        actual_after = speed * (obs_times[-1] - t_int)
    else:
        actual_before = 0.0
        actual_after = 0.0

    info = dict(
        intercept_step=intercept_step,
        intercept_time=t_int,
        target_robot=int(target_robot),
        target_xy=(tx, ty),
        x0=x0, y0=y0, vx=vx, vy=vy,
        speed=speed, radius=radius,
        approach_angle_deg=math.degrees(approach_angle_rad),
        travel_before=actual_before,
        travel_after=actual_after,
    )
    return obs_positions, obs_times, info


def generate_obstacle_cluster_to_robot(
        path_positions, path_times,
        wp_configs, offsets, target_robot,
        base_intercept_frac, base_approach_mode,
        base_approach_side, base_speed, base_radius,
        n_extra=0, frac_spread=0.08,
        speed_range=(20.0, 40.0),
        radius_range=(30.0, 50.0),
        base_travel_before=0.0, base_travel_after=0.0,
        travel_after_range=(200.0, 500.0),
        seed=42):
    """Robot-targeted counterpart of :func:`generate_obstacle_cluster`.

    Every obstacle in the cluster aims at the position of
    ``target_robot`` under the *baseline* pose of the spatial planner
    at its intercept step.  Extras share the same target robot but
    keep the randomised intercept fractions, speeds and radii of the
    base generator.
    """
    n_wp = len(path_positions)
    rng = np.random.default_rng(seed)

    int_step = int(round(base_intercept_frac * (n_wp - 1)))
    angle_rad = compute_approach_angle(
        path_positions, int_step,
        mode=base_approach_mode, side=base_approach_side)
    pos, t, info = compute_intercept_trajectory_to_robot(
        path_positions, path_times,
        wp_configs, offsets, target_robot,
        intercept_step=int_step,
        approach_angle_rad=angle_rad,
        speed=base_speed, radius=base_radius,
        travel_before=base_travel_before,
        travel_after=base_travel_after)
    obstacles = [{'positions': pos, 'times': t, 'radius': base_radius}]
    infos = [info]

    sides = ["left", "right"]
    modes = ["perpendicular", "head_on"]

    for k in range(n_extra):
        frac_off = rng.uniform(-frac_spread, frac_spread)
        frac_k = float(np.clip(base_intercept_frac + frac_off, 0.02, 0.98))
        step_k = int(round(frac_k * (n_wp - 1)))

        mode_k = modes[k % len(modes)]
        side_k = sides[k % len(sides)]

        speed_k = float(rng.uniform(*speed_range))
        radius_k = float(rng.uniform(*radius_range))
        tafter_k = float(rng.uniform(*travel_after_range))

        angle_k = compute_approach_angle(
            path_positions, step_k, mode=mode_k, side=side_k)
        pos_k, t_k, info_k = compute_intercept_trajectory_to_robot(
            path_positions, path_times,
            wp_configs, offsets, target_robot,
            intercept_step=step_k,
            approach_angle_rad=angle_k,
            speed=speed_k, radius=radius_k,
            travel_after=tafter_k)
        obstacles.append({'positions': pos_k, 'times': t_k,
                          'radius': radius_k})
        infos.append(info_k)

    return obstacles, infos


def generate_obstacle_cluster(
        path_positions, path_times,
        base_intercept_frac, base_approach_mode,
        base_approach_side, base_speed, base_radius,
        n_extra=2, frac_spread=0.08,
        speed_range=(20.0, 40.0),
        radius_range=(30.0, 50.0),
        base_travel_before=0.0, base_travel_after=0.0,
        travel_after_range=(200.0, 500.0),
        seed=42):
    """Generate ``1 + n_extra`` obstacles intercepting near the same point.

    The first obstacle uses the *base* parameters verbatim.  Extras
    are randomised within the supplied ranges; their intercept fraction
    is jittered around ``base_intercept_frac`` by up to ``frac_spread``.

    Returns
    -------
    obstacles : list of dict
        Each ``{'positions': (M, 2), 'times': (M,), 'radius': float}``.
    infos : list of dict
        Per-obstacle metadata (intercept step, angle, speed, …).
    """
    n_wp = len(path_positions)
    rng = np.random.default_rng(seed)

    int_step = int(round(base_intercept_frac * (n_wp - 1)))
    angle_rad = compute_approach_angle(
        path_positions, int_step,
        mode=base_approach_mode, side=base_approach_side)
    pos, t, info = compute_intercept_trajectory(
        path_positions, path_times,
        intercept_step=int_step,
        approach_angle_rad=angle_rad,
        speed=base_speed, radius=base_radius,
        travel_before=base_travel_before,
        travel_after=base_travel_after)
    obstacles = [{'positions': pos, 'times': t, 'radius': base_radius}]
    infos = [info]

    sides = ["left", "right"]
    modes = ["perpendicular", "head_on"]

    for k in range(n_extra):
        frac_off = rng.uniform(-frac_spread, frac_spread)
        frac_k = float(np.clip(base_intercept_frac + frac_off, 0.02, 0.98))
        step_k = int(round(frac_k * (n_wp - 1)))

        mode_k = modes[k % len(modes)]
        side_k = sides[k % len(sides)]

        speed_k = float(rng.uniform(*speed_range))
        radius_k = float(rng.uniform(*radius_range))
        tafter_k = float(rng.uniform(*travel_after_range))

        angle_k = compute_approach_angle(
            path_positions, step_k, mode=mode_k, side=side_k)
        pos_k, t_k, info_k = compute_intercept_trajectory(
            path_positions, path_times,
            intercept_step=step_k,
            approach_angle_rad=angle_k,
            speed=speed_k, radius=radius_k,
            travel_after=tafter_k)
        obstacles.append({'positions': pos_k, 'times': t_k,
                          'radius': radius_k})
        infos.append(info_k)

    return obstacles, infos


def generate_obstacles_distributed(
        path_positions, path_times,
        n_obstacles,
        frac_min=0.10, frac_max=0.95,
        speed_range=(2.0, 20.0),
        radius_range=(20.0, 60.0),
        travel_before_range=(0.0, 0.0),
        travel_after_range=(0.0, 0.0),
        modes=("perpendicular", "head_on"),
        sides=("left", "right"),
        seed=0):
    """Generate ``n_obstacles`` intercepts uniformly spread along the path.

    Companion to :func:`generate_obstacle_cluster` but with intercept
    fractions sampled uniformly on ``[frac_min, frac_max]`` instead of
    being jittered around a single point.

    Approach mode and side cycle through the given tuples so that
    successive obstacles alternate; speed / radius / travel-window are
    drawn independently from their ranges.

    Returns ``(obstacles, infos)`` in the same shape as
    :func:`generate_obstacle_cluster`.
    """
    n_wp = len(path_positions)
    rng = np.random.default_rng(seed)

    fracs = rng.uniform(frac_min, frac_max, size=n_obstacles)
    fracs.sort()

    obstacles, infos = [], []
    for k in range(n_obstacles):
        frac_k = float(np.clip(fracs[k], 0.02, 0.98))
        step_k = int(round(frac_k * (n_wp - 1)))

        mode_k = modes[k % len(modes)]
        side_k = sides[k % len(sides)]

        speed_k = float(rng.uniform(*speed_range))
        radius_k = float(rng.uniform(*radius_range))
        tbefore_k = float(rng.uniform(*travel_before_range))
        tafter_k = float(rng.uniform(*travel_after_range))

        angle_k = compute_approach_angle(
            path_positions, step_k, mode=mode_k, side=side_k)
        pos_k, t_k, info_k = compute_intercept_trajectory(
            path_positions, path_times,
            intercept_step=step_k,
            approach_angle_rad=angle_k,
            speed=speed_k, radius=radius_k,
            travel_before=tbefore_k,
            travel_after=tafter_k)
        obstacles.append({'positions': pos_k, 'times': t_k,
                          'radius': radius_k})
        infos.append(info_k)

    return obstacles, infos


def check_collisions(path_positions, path_offsets,
                     obs_positions, rb, obs_radius):
    """Brute-force diagnostic — every step where any robot disc
    overlaps the obstacle disc.

    Returns list of ``(step, robot_index, distance)``.
    """
    out = []
    threshold = rb + obs_radius
    n = min(len(path_positions), len(obs_positions))
    for i in range(n):
        ox, oy = obs_positions[i]
        rpos = path_offsets[i] + path_positions[i]
        for k in range(rpos.shape[0]):
            d = math.hypot(rpos[k, 0] - ox, rpos[k, 1] - oy)
            if d < threshold:
                out.append((i, k, d))
    return out

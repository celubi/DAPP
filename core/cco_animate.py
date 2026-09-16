"""Animation for the CCO_planner A* output.

``animate_path`` takes a path of ``(mL_xy, mR_xy)`` pairs and draws
the 6-robot bilateral formation moving along the wall map, with
online collision / chord-length sanity checks (bad robots flash red).
"""

from __future__ import annotations

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .cco_obstacle import _circle_free, get_cluster_positions


def animate_path(path_AB, wall_img, BAR, robot_r, intra_robot_dist,
                 clearance, title="CCO_planner — Formation Path"):
    rf = 0.5 * BAR
    cont = []
    for A, B in path_AB:
        cx_p = 0.5 * (A[0] + B[0])
        cy_p = 0.5 * (A[1] + B[1])
        d = math.hypot(B[0] - A[0], B[1] - A[1])
        scale = d / BAR if BAR > 0 else 1.0
        theta = math.atan2(B[1] - A[1], B[0] - A[0])
        cont.append((cx_p, cy_p, theta, scale, A, B))

    map_rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(map_rgb, cv2.COLOR_BGR2RGB),
              cmap="gray", origin="upper")
    ax.set_axis_off()

    centres = np.array([(c[0], c[1]) for c in cont])
    ax.plot(centres[:, 0], centres[:, 1], "-",
            color="deepskyblue", linewidth=1.5, alpha=0.6)
    ax.plot(centres[0, 0], centres[0, 1], "o",
            color="lime", markersize=8, zorder=5)
    ax.plot(centres[-1, 0], centres[-1, 1], "o",
            color="red", markersize=8, zorder=5)

    n_robots = 6
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_robots, 3)))[:n_robots]
    circles = []
    for k in range(n_robots):
        c = plt.Circle((0, 0), robot_r, color=colors[k], alpha=0.85,
                       zorder=10)
        ax.add_patch(c)
        circles.append(c)

    fc = plt.Circle((0, 0), rf, fill=False, edgecolor="white",
                    linestyle="--", linewidth=1, alpha=0.6, zorder=9)
    ax.add_patch(fc)
    cdot, = ax.plot([], [], "x", color="white", markersize=7,
                    zorder=11)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                  fontsize=10, color="white", va="top",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="black",
                            alpha=0.6))

    ON_CIRCLE_TOL, CHORD_TOL = 1.5, 1.5
    reported: set = set()

    def _check_frame(frame, pts):
        cx_p, cy_p, _, scale, A, B = cont[frame]
        rs = rf * scale
        bad = [False] * n_robots
        issues = []
        for k, p in enumerate(pts):
            if not _circle_free(float(p[0]), float(p[1]),
                                float(robot_r), clearance):
                bad[k] = True
                issues.append(f"R{k} collides")
        for k, p in enumerate(pts):
            err = abs(math.hypot(p[0] - cx_p, p[1] - cy_p) - rs)
            if err > ON_CIRCLE_TOL:
                bad[k] = True
                issues.append(f"R{k} off-circle ({err:.2f})")
        for cluster_idx, base in enumerate((0, 3)):
            main = pts[base]
            for sat_off in (1, 2):
                sat = pts[base + sat_off]
                chord = math.hypot(sat[0] - main[0], sat[1] - main[1])
                err = abs(chord - intra_robot_dist)
                if err > CHORD_TOL:
                    bad[base] = True
                    bad[base + sat_off] = True
                    issues.append(f"cluster{cluster_idx} chord off "
                                  f"({chord:.2f}, err {err:.2f})")
        if issues and frame not in reported:
            reported.add(frame)
            print(f"  [frame {frame}] " + "; ".join(issues))
        return bad

    def _update(frame):
        cx_p, cy_p, theta, scale, A, B = cont[frame]
        cluster_a, cluster_b = get_cluster_positions(A, B,
                                                      intra_robot_dist)
        # Round each robot centre to the nearest integer pixel *before*
        # both the sanity check and the draw, mirroring DA A*'s
        # collision model (offsets are integer, so the checked pixel and
        # the drawn disc coincide).  ``_circle_free`` rounds internally
        # anyway, so this only aligns the *drawn* disc to the checked
        # pixel — it removes the sub-pixel gap where the disc appeared to
        # graze an obstacle the check had already cleared.
        pts = [(float(round(p[0])), float(round(p[1])))
               for p in (cluster_a + cluster_b)]
        bad = _check_frame(frame, pts)
        n_bad = sum(bad)
        for k in range(n_robots):
            circles[k].center = (pts[k][0], pts[k][1])
            if bad[k]:
                circles[k].set_facecolor("red")
                circles[k].set_edgecolor("red")
            else:
                circles[k].set_facecolor(colors[k])
                circles[k].set_edgecolor(colors[k])
        fc.center = (cx_p, cy_p)
        fc.set_radius(rf * scale)
        cdot.set_data([cx_p], [cy_p])
        status = (f"  [BAD: {n_bad}]" if n_bad else "  [ok]")
        txt.set_text(
            f"step {frame}/{len(cont)-1}  "
            f"θ={math.degrees(theta):+.0f}°  s={scale:.2f}  "
            f"|AB|={math.hypot(B[0]-A[0], B[1]-A[1]):.0f}px"
            f"{status}")
        return circles + [fc, cdot, txt]

    ax.set_title(title, fontsize=14)
    anim = FuncAnimation(fig, _update, frames=len(cont),
                         interval=150, repeat=True, blit=True)
    cur = [0]

    def _on_key(event):
        if event.key == "right":
            cur[0] = (cur[0] + 1) % len(cont)
        elif event.key == "left":
            cur[0] = (cur[0] - 1) % len(cont)
        elif event.key == " ":
            if anim.event_source is not None:
                if getattr(anim, "_running", True):
                    anim.pause(); anim._running = False
                else:
                    anim.resume(); anim._running = True
            return
        else:
            return
        _update(cur[0])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show()

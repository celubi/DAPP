"""Configuration for the CCO_planner (Critical Crossable Obstacle) demo.

The formation- and cluster-defining parameters are **derived from**
:mod:`config.bilateral_only` so the CCO and DA_astar planners describe
the same physical formation:

    BAR              = 2 · RF            main↔main separation = circle Ø
    TOL_FRAC         = (S_MAX − S_MIN)/2 scale band, symmetric about s=1
    INTRA_ROBOT_DIST = 2 · RF · sin(φ/2) chord main↔satellite at s=1,
                       where φ is the angular half-spread of a cluster
                       in DA_astar's bilateral formation
    ROBOT_R          = RB                robot body radius

Scene paths, polygon sampling, inflation radius and thread count are
CCO-specific and overridable through the demo's argparse interface.
"""

import math
from pathlib import Path

from config import bilateral_only as _la

_ROOT = Path(__file__).resolve().parent.parent

# ─── Scene (CCO specific) ───────────────────────────────────
# The wall map must be the OPEN variant of the obstacle: the closed
# variant's tight enclosing border leaves the boundary anchor pairs no
# clearance, and start/goal resolution fails at the obstacle ends.
MAP_PATH  = str(_ROOT / "maps" / "mp_narrow_flip.png")
WALL_PATH = str(_ROOT / "maps" / "wall_narrow_flip_open.png")


# ─── Derivation helpers ─────────────────────────────────────

def _bilateral_half_spread_rad(formations_deg):
    """Angular half-spread of a cluster in DA_astar's bilateral formation.

    By convention the bilateral (2-cluster) formation is the **last**
    entry of ``formations_deg`` — that is the one the CCO_planner models.
    Returns the max angular deviation (radians) of a robot from its
    cluster's slot angle, i.e. the spread used to size the intra-cluster
    chord.  (A singleton cluster yields 0 → chord 0.)
    """
    TWO_PI = 2.0 * math.pi
    formation = formations_deg[-1]

    half_spread = 0.0
    for grp in formation:
        ang = [math.radians(a) for a in grp]
        slot = math.atan2(sum(math.sin(a) for a in ang),
                          sum(math.cos(a) for a in ang))
        for a in ang:
            d = ((a - slot + math.pi) % TWO_PI) - math.pi
            half_spread = max(half_spread, abs(d))
    return half_spread


# ─── Anchor / formation geometry (derived from DA_astar) ────
# main↔main separation = diameter of the formation circle at s=1.
BAR = 2.0 * _la.RF

# Scale band. DA_astar's |AB| = 2·RF·s sweeps [2·RF·S_MIN, 2·RF·S_MAX];
# the CCO planner's |AB| sweeps BAR·[1−TOL_FRAC, 1+TOL_FRAC]. With
# BAR = 2·RF the two coincide iff the band is symmetric about s=1 and
# TOL_FRAC = (S_MAX − S_MIN)/2.
_s_mid = 0.5 * (_la.S_MIN + _la.S_MAX)
if abs(_s_mid - 1.0) > 1e-6:
    raise ValueError(
        f"DA_astar scale band [{_la.S_MIN}, {_la.S_MAX}] is not symmetric "
        f"about s=1 (midpoint {_s_mid:.3f}). The CCO planner's symmetric "
        f"TOL_FRAC cannot reproduce it; adjust S_MIN/S_MAX or set TOL_FRAC "
        f"by hand.")
TOL_FRAC = 0.5 * (_la.S_MAX - _la.S_MIN)

# Intra-cluster chord (main↔satellite) at s=1, from the bilateral
# formation's angular spread: chord = 2·RF·sin(half_spread/2).
INTRA_ROBOT_DIST = 2.0 * _la.RF * math.sin(
    0.5 * _bilateral_half_spread_rad(_la.FORMATIONS_DEG))

# Robot body radius — same physical robots.
ROBOT_R = float(_la.RB)

# Collision-check safety margin (px) added to ROBOT_R only in the
# planning kernel's clearance test (not for drawing): ``_circle_free``
# accepts exact tangency (``>=``), so the margin guarantees a real gap
# between the drawn disc and the obstacle.  0.0 disables it.
CLEARANCE_MARGIN = 0.1

# ─── Payload / cable height check (derived from DA_astar) ───
# Same payload model as DA_astar, inherited from config.bilateral_only.
# The CCO planner has no separate height image, so the wall map doubles
# as the height map.  PAYLOAD_CHECK is the master on/off switch.
PAYLOAD_CHECK        = True
HEIGHT_MAP_PATH      = WALL_PATH
L_POLE               = float(_la.L_POLE)
L_ROPE               = float(_la.L_ROPE)
CABLE_SAMPLE_STEP_PX = float(_la.CABLE_SAMPLE_STEP_PX)
HEIGHT_MAX           = int(_la.HEIGHT_MAX)

# ─── CCO specific (no DA_astar equivalent) ──────────────────
STEP_SIZE = 10         # sampling step on the inflated polygon (px)
R_INFL    = 50.0       # obstacle inflation radius (px)

# ─── Numba threads (0 = use all cores) ──────────────────────
THREADS = 0

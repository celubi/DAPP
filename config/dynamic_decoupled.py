"""Configuration for the decoupled-actions dynamic A* demo.

Used exclusively by :mod:`demo.demo_dynamic_decoupled` (one axis per
action: move / rotate / scale / wait).  The per-axis dt's below are
applied uniformly to the spatial baseline and the dynamic search via
``compute_wp_dt_offsets``.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ─── Map ────────────────────────────────────────────────────
MAP_PATH   = str(_ROOT / "random_maps" / "random_map_5.png")
OBS_THRESH = 128

# ─── Formation (cluster-based schema) ───────────────────────
FORMATIONS_DEG = [
    #[[0.0], [60.0], [120.0], [180.0], [240.0], [300.0]],
    [[-20.0, 0.0, 20.0], [160.0, 180.0, 200.0]],
]

# ─── Robot / formation geometry (pixels) ────────────────────
RB = 10.0
RF = 100.0

# ─── Payload / cable height check ───────────────────────────
HEIGHT_MAP_PATH      = str(_ROOT / "random_maps" / "random_map_5_height.png")  # None to disable
L_POLE               = 200.0
L_ROPE               = 180.0
CABLE_SAMPLE_STEP_PX = 10
HEIGHT_MAX           = 200

# ─── State discretisation ───────────────────────────────────
XY_STEP = 10
N_THETA = 144
S_MIN, S_MAX, N_S = 0.7, 1.3, 20

# ─── Spatial cost weights ───────────────────────────────────
W_MOVE   = 1.0
W_ROT    = 1.0
W_SCALE  = 1.0
W_CONFIG = 1.0

# ─── Adaptive branching ─────────────────────────────────────
C_DEFORM = None

# ─── Reconfiguration collision check ────────────────────────
RECONFIG_CHECK = 'sampling'
N_ARC_SAMPLES  = 8

# ─── Goal relaxation ────────────────────────────────────────
FREE_THETA  = False
FREE_S      = False
FREE_CONFIG = False

# ─── Symmetry-based θ-axis pruning ──────────────────────────
USE_SYMMETRY = False

# ─── Animation ──────────────────────────────────────────────
FPS      = 30
INTERVAL = 1000 // FPS

# ═══════════════════════════════════════════════════════════
#  Dynamic A* (time-aware re-planning around moving obstacles)
# ═══════════════════════════════════════════════════════════

# ─── Discretisation & timing ────────────────────────────────
TIME_STEP = 0.5     # seconds per planner time step

# Temporal horizon for the dynamic search:
#     horizon = naive_secs * TIME_BUDGET_FACTOR + TIME_BUDGET_EXTRA
# where ``naive_secs`` is the time to traverse the *inflated* spatial
# window at one waypoint per TIME_STEP.
TIME_BUDGET_FACTOR = 2.0
TIME_BUDGET_EXTRA  = 20.0

# ─── Cost weights (dynamic phase) ───────────────────────────
W_PATH      = 1.0
W_ROT_DYN   = 1.0
W_SCALE_DYN = 1.0
W_TIME      = 10

# ─── Window & bounded offsets ───────────────────────────────
AUTO_WINDOW          = True
WINDOW_MARGIN_BEFORE = 30
WINDOW_MARGIN_AFTER  = 10
MAX_DTHETA_DEG       = 120.0
MAX_DS_STEPS         = 10

# ─── Search behaviour ───────────────────────────────────────
ALLOW_BACKWARD = True
WA_EPSILON     = 1.0
MAX_EXP_DYN    = 2_000_000

# ─── Per-axis dt's (time-step cost of each primitive) ──────
DT_MOVE  = 1
DT_ROT   = 1
DT_SCALE = 3
DT_WAIT  = 1

# ─── Dynamic-obstacle height (for payload / cable check) ────
OBS_DYN_HEIGHT = 40.0

# ─── Main intercepting obstacle ─────────────────────────────
INTERCEPT_FRAC         = 0.5
APPROACH_MODE          = "perpendicular"
APPROACH_SIDE          = "left"
OBSTACLE_SPEED         = 10.0
OBSTACLE_RADIUS        = 20.0
OBSTACLE_TRAVEL_BEFORE = 0.0
OBSTACLE_TRAVEL_AFTER  = 0.0

# ─── Target selection ───────────────────────────────────────
# When None the obstacle cluster aims at the *centre* of the
# formation at each intercept step.  When set to an integer k in
# [0, n_robots) the cluster aims at the position of robot k.
TARGET_ROBOT = 0

# ─── Extra obstacle cluster ─────────────────────────────────
N_EXTRA_OBSTACLES          = 3
EXTRA_OBS_FRAC_SPREAD      = 0.08
EXTRA_OBS_SPEED_MIN        = 2.0
EXTRA_OBS_SPEED_MAX        = 20.0
EXTRA_OBS_RADIUS_MIN       = 20.0
EXTRA_OBS_RADIUS_MAX       = 40.0
EXTRA_OBS_TRAVEL_AFTER_MIN = 0.0
EXTRA_OBS_TRAVEL_AFTER_MAX = 0.0
EXTRA_OBS_SEED             = 76

"""Configuration for the DA_astar demo (single bilateral formation).

Every parameter that affects planner output lives here.  Path values
are resolved relative to the repository root so the project is
portable; override them if you want to point at a different map.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ─── Map ────────────────────────────────────────────────────
MAP_PATH   = str(_ROOT / "maps" / "wall_narrow_flip_close.png")
OBS_THRESH = 128

# ─── Formation (cluster-based schema) ───────────────────────
# Each top-level entry is one formation the planner may switch into.
# Each formation is a list of clusters; each cluster is a list of
# robot angles (degrees) on the formation circle.
_DELTA = 13.3995 # For a chord of approx. 35cm

FORMATIONS_DEG = [
    # Nominal: 6 singletons evenly spaced, sym=6
    #[[0.0], [60.0], [120.0], [180.0], [240.0], [300.0]],
    # Triangular: 3 pairs at 0°/120°/240°, sym=3
    #[[-10.0, 10.0], [110.0, 130.0], [230.0, 250.0]],
    # Bilateral: 2 triples at 0°/180°, sym=2
    [[-_DELTA, 0.0, _DELTA],[180.0 - _DELTA, 180.0, 180.0 + _DELTA]],
]

# ─── Robot / formation geometry (pixels) ────────────────────
RB = 15.0         # robot body radius
RF = 150.0        # formation circle radius (at scale s=1)

# ─── Payload / cable height check ───────────────────────────
HEIGHT_MAP_PATH      = None#MAP_PATH    # set to None to disable
L_POLE               = 300.0
L_ROPE               = 280.0
CABLE_SAMPLE_STEP_PX = 20
HEIGHT_MAX           = 200

# ─── State discretisation ───────────────────────────────────
# 8 px: at coarser steps the enclosed corridor on MAP_PATH becomes
# infeasible.
XY_STEP = 8
N_THETA = 144     # must be divisible by every sym order
S_MIN, S_MAX, N_S = 0.8, 1.2, 40

# ─── Cost weights ───────────────────────────────────────────
W_MOVE   = 1.0
W_ROT    = 1.0
W_SCALE  = 1.0
W_CONFIG = 1.0

# ─── Adaptive branching ─────────────────────────────────────
C_DEFORM = ((RF + RB) * S_MAX) + XY_STEP

# ─── Reconfiguration collision check ────────────────────────
RECONFIG_CHECK = 'sampling'
N_ARC_SAMPLES  = 8

# ─── Goal relaxation ────────────────────────────────────────
FREE_THETA  = False
FREE_S      = False
FREE_CONFIG = False

# ─── Symmetry-based θ-axis pruning ──────────────────────────
# True: fold each formation's θ axis to period = N_THETA // sym_order.
# False: explore the full θ axis (sym_orders forced to 1).
USE_SYMMETRY = True

# ─── Animation ──────────────────────────────────────────────
FPS      = 30
INTERVAL = 1000 // FPS

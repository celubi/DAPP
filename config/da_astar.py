"""Default configuration for the DA_astar planner (3-formation set).

Every parameter that affects planner output lives here.  Path values
are resolved relative to the repository root so the project is
portable; override them if you want to point at a different map.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ─── Map ────────────────────────────────────────────────────
MAP_PATH   = str(_ROOT / "random_maps" / "random_map_4.png")
OBS_THRESH = 128

# ─── Formation (cluster-based schema) ───────────────────────
# Each top-level entry is one formation the planner may switch into.
# Each formation is a list of clusters; each cluster is a list of
# robot angles (degrees) on the formation circle.
FORMATIONS_DEG = [
    # Nominal: 6 singletons evenly spaced, sym=6
    [[0.0], [60.0], [120.0], [180.0], [240.0], [300.0]],
    # Triangular: 3 pairs at 0°/120°/240°, sym=3
    [[-10.0, 10.0], [110.0, 130.0], [230.0, 250.0]],
    # Bilateral: 2 triples at 0°/180°, sym=2
    [[-20.0, 0.0, 20.0], [160.0, 180.0, 200.0]],
]

# ─── Robot / formation geometry (pixels) ────────────────────
RB = 10.0         # true robot body radius (what the videos draw)

# Radius the *planner* uses.  Inflating the body by a couple of
# centimetres is the standard way of buying back the margin the
# discretisation eats (grid step, rounded robot centres, cable
# sampling): the plan is then feasible for the real 10 cm robots even
# where a state sits between two cells.  Drawing keeps RB, so the
# video shows the machines, not the margin.
RB_PLAN = 10.0
RF = 100.0        # formation circle radius (at scale s=1)

# ─── Payload / cable height check ───────────────────────────
HEIGHT_MAP_PATH      = str(_ROOT / "random_maps" / "random_map_4_height.png")
L_POLE               = 200.0
L_ROPE               = 180.0
CABLE_SAMPLE_STEP_PX = 5
HEIGHT_MAX           = 100

# ─── State discretisation ───────────────────────────────────
XY_STEP = 16
N_THETA = 72     # must be divisible by every sym order
S_MIN, S_MAX, N_S = 0.7, 1.3, 10

# ─── Cost weights ───────────────────────────────────────────
W_MOVE   = 1.0
W_ROT    = 1.0
W_SCALE  = 1.0
W_CONFIG = 1.0

# ─── Adaptive branching ─────────────────────────────────────
# Built from the planning radius, the one the collision check uses:
# with RB, a band of cells would count as open space (no scaling or
# reconfiguring) for a formation that is not guaranteed to fit there.
C_DEFORM = (RF * S_MAX) + RB_PLAN + XY_STEP

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

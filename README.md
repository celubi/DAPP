# Deformation-aware Path Planning

Path planning for a multi-robot formation carrying a cable-suspended
payload.  The formation moves on a 2-D occupancy map with a 5-D state
`(x, y, θ, s, k)` — position, orientation, scale, and formation
template — and can optionally check the suspended payload against an
obstacle *height* map.  All planners are Numba-JIT compiled.

## Demo video

<!-- TODO: replace fHGQPHXpJRI (5 occurrences below) with the YouTube video id -->
**▶ [Watch the demo video on YouTube](https://www.youtube.com/watch?v=fHGQPHXpJRI)**

The video follows the paper: Deformation-aware A*, its two extensions
for moving obstacles and unknown environments, then the critical
crossable-obstacle planner.  Each part opens with a title card.

| Starts at | Paper | Planner | Code |
|---|---|---|---|
| [0:11](https://www.youtube.com/watch?v=fHGQPHXpJRI&t=11s) | §5.3 | Deformation-aware A* | `core/da_astar.py` |
| [1:10](https://www.youtube.com/watch?v=fHGQPHXpJRI&t=70s) | §5.4 | ↳ Dynamic extension | `core/dynamic.py` |
| [1:45](https://www.youtube.com/watch?v=fHGQPHXpJRI&t=105s) | §5.5 | ↳ Unknown environments extension | `core/receding_horizon.py` |
| [3:04](https://www.youtube.com/watch?v=fHGQPHXpJRI&t=184s) | §5.6 | Critical crossable-obstacle planner | `core/cco_planner.py` |

## The three planners

### 1. DA_astar — Deformation-Aware A*  (`core/da_astar.py`)

A* over the 5-D configuration space: neighbours are generated
on demand, so memory scales with expanded nodes, not with the grid.
Entry points: `find_path_da`, `find_path_da_from_map`.

Two **optional pruning strategies**:

- **Loose-space pruning** (`c_deform`): scale and reconfiguration
  edges are generated only where the formation-centre clearance is
  below the threshold — in open ("loose") space only spatial moves are produced. 
- **Symmetry pruning** (`use_symmetry`): each template's θ axis is
  folded to its fundamental symmetry period, and reconfiguration
  edges branch over the distinct orbit representatives, with the
  folded rotation treated as free relabelling.  With symmetry off
  the planner degenerates exactly to the unfolded full-space search.

Two **extensions** build on DA_astar:

- **Dynamic extension** (`core/dynamic.py` + `core/dynamic_jit.py`):
  given a DA_astar spatial path and moving obstacles, a windowed
  time-aware A* re-plans timing plus bounded θ/scale offsets with
  decoupled, one-axis-at-a-time actions (`dynamic_astar_windowed`).
  Moving-obstacle scenario generation lives in `core/obstacles.py`.
- **Unknown environments extension** (`core/receding_horizon.py`):
  receding-horizon online planning on a progressively-revealed map —
  sense a disc, plan optimistically (unknown = free), commit only the
  prefix inside the observed region, repeat (`plan_receding_horizon`).

### 2. baseline_planner  (`core/baseline_planner.py`)

The reference technique of Liu et al. (arXiv:2210.03340): full
validity mapping of the whole configuration space, complete graph
materialisation (CSR), BFS feasibility gate, then Dijkstra with no
heuristic.  It reuses DA_astar's expansion kernel, so it minimises the
same cost function — `baseline == DA-full` costs are the fairness
check of the benchmark.  Entry point: `plan_baseline_from_map`.

### 3. CCO_planner — Critical Crossable Obstacle Planner
(`core/cco_planner.py` driver; `core/cco_kernels.py`,
`core/cco_astar.py`, `core/cco_obstacle.py`)

Plans the bilateral formation *across* a single elongated
obstacle: the obstacle boundary is inflated and split into LEFT/RIGHT
anchor chains (`cco_obstacle.Obstacle`), a parallel kernel optimises
the robot slots for every anchor pair (`optimise_anchors_batch`), and
a small A* recovers the path over the implicit anchor graph
(`astar_small_jit`).  Start and goal are derived automatically at the
obstacle's two ends via `find_start`.  The high-level entry points
live in `core/cco_planner.py` — `prepare_cco_scene` +
`build_cco_chains` (preprocessing) and `find_path_cco` (the timed
find-path phase) — and are shared by the demo and the benchmark, so
both report the same timing quantity by construction.

## Layout

```
core/                the three planners + extensions + shared utilities
config/              parameter modules shared by the demos and benchmarks
demo/                4 interactive demos (see below)
test/                9 benchmark / scaling studies
scripts/             generate_random_map.py (random occupancy+height maps)
maps/                obstacle / wall maps for the CCO demo and benchmark
random_maps/         generated maps (occupancy + _height twin) for the
                     DA_astar-based demos and benchmarks
experimental_maps/   scaling_map.png for the resolution / formation-set studies,
                     pruning_map.png for the pruning matrix
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.10, numba 0.65
```

**Always run from the repo root with `python -m …`** (the packages
`core/`, `config/`, `demo/`, `test/` are resolved from the cwd; running
a file by path breaks imports).  The first run of each planner pays a
one-time Numba compilation cost (the on-disk cache makes later runs
fast).  Every test, and the DA_astar and CCO demos, warm up the JIT
before timing anything; the dynamic-decoupled and receding-horizon
demos do not, so their first timings include compilation.

## Demos

| Command | What it shows |
|---|---|
| `python -m demo.demo_da_astar` | DA_astar; click START/GOAL on the map, watch the formation |
| `python -m demo.demo_dynamic_decoupled` | DA_astar + dynamic decoupled re-planning around moving obstacles (click START/GOAL) |
| `python -m demo.demo_receding_horizon` | DA_astar + receding horizon on a progressively-revealed map (click START/GOAL) |
| `python -m demo.demo_cco_planner` | CCO_planner; start/goal auto-derived at the obstacle's two ends |

The three click-demos also accept `--start IX,IY --goal IX,IY` (grid
cells) and `--no-animate` for non-interactive runs; the CCO demo has
`--no-animate` and exposes every parameter via flags.

## Tests / benchmarks

| Command | Measures |
|---|---|
| `python -m test.test_baseline_vs_da_astar` | baseline vs DA-full vs DA-pruning (symmetry + `c_deform`) on one map; includes the `baseline == DA-full` fairness check |
| `python -m test.test_pruning_matrix` | DA_astar 2×2 pruning matrix (symmetry × `c_deform`) on one query: cost, expansions and time vs the no-pruning reference |
| `python -m test.test_cco_vs_da_astar` | CCO_planner vs DA_astar (pruning) on a crossable obstacle, search-time tables + comparison figures |
| `python -m test.test_different_maps` | DA_astar expansions, cost and time across maps × start/goal pairs × formation sets |
| `python -m test.test_xy_resolution` | DA_astar scaling vs XY grid resolution |
| `python -m test.test_rot_resolution` | DA_astar scaling vs θ resolution |
| `python -m test.test_s_resolution` | DA_astar scaling vs scale resolution |
| `python -m test.test_dynamic_decoupled_perf` | dynamic extension vs obstacle count (`--quick`, `--plot`) |
| `python -m test.test_receding_horizon_perf` | receding horizon vs fully-known-map baseline (`--quick`, `--plot`) |

`scripts/generate_random_map.py` (run by path, not `-m`) generates new
`random_maps/random_map_<n>.png` + `_height` pairs.

## Reference

The models and design choices behind this code are described in:

> *Deformation-aware path planning for cable-tethered multi-robot payload transportation*

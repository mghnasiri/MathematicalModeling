# CLAUDE.md

## Project Overview

Operations Research problem repository featuring mathematical formulations, algorithm implementations, and benchmarks. Educational/research-focused with academic rigor (references, complexity analysis, scheduling notation). Currently in **Phase 1** (Scheduling problems), with future phases planned for Routing, Packing, Location, and Stochastic problems.

**Author**: Mohammad Ghafourian Nasiri
**License**: MIT
**Python**: 3.10+ required (uses `from __future__ import annotations`, `|` union syntax)

## Repository Structure

```
MathematicalModeling/
├── CLAUDE.md
├── README.md              # Project overview + taxonomy
├── CONTRIBUTING.md         # Guidelines for adding problems/algorithms
├── requirements.txt        # numpy, scipy, matplotlib, pytest, pandas
├── docs/
│   └── taxonomy.md         # Full problem classification (9 families)
├── shared/                 # Reusable infrastructure
│   ├── parsers/
│   │   └── taillard_parser.py   # Taillard benchmark downloader/parser
│   ├── exact/              # (placeholder for shared exact method components)
│   ├── metaheuristics/     # (placeholder for generic metaheuristic frameworks)
│   └── visualization/      # (placeholder for visualization tools)
└── problems/
    └── scheduling/
        ├── flow_shop/      # FULLY IMPLEMENTED
        │   ├── instance.py             # FlowShopInstance, FlowShopSolution dataclasses
        │   ├── benchmark_runner.py     # CLI for evaluating algorithms on Taillard instances
        │   ├── exact/
        │   │   ├── johnsons_rule.py    # Optimal for F2||Cmax, O(n log n)
        │   │   ├── mip_formulation.py  # SciPy HiGHS + OR-Tools CP-SAT solvers
        │   │   └── branch_and_bound.py # Taillard lower bound, NEH warm-start
        │   ├── heuristics/
        │   │   ├── palmers_slope.py    # Slope index, O(n*m + n log n)
        │   │   ├── guptas_algorithm.py # Priority-based, O(n*m + n log n)
        │   │   ├── dannenbring.py      # Rapid Access weighted Johnson's, O(n*m + n log n)
        │   │   ├── cds.py             # Campbell-Dudek-Smith, O(m * n log n)
        │   │   ├── neh.py             # Nawaz-Enscore-Ham, O(n^2 * m), best constructive
        │   │   └── lr_heuristic.py    # Multi-candidate greedy, O(n^3 * m)
        │   ├── metaheuristics/
        │   │   ├── iterated_greedy.py      # Ruiz & Stuetzle (2007), state-of-the-art
        │   │   ├── simulated_annealing.py  # Osman & Potts (1989), classic SA
        │   │   ├── genetic_algorithm.py    # Reeves (1995), OX crossover + insertion mutation
        │   │   ├── tabu_search.py          # Nowicki & Smutnicki (1996), fast TS
        │   │   ├── ant_colony.py           # Stützle (1998), ACO with MMAS pheromone bounds
        │   │   └── local_search.py         # Swap, insertion, or-opt, VND neighborhoods
        │   ├── variants/
        │   │   ├── no_wait/           # Fm | prmu, no-wait | Cmax
        │   │   │   ├── instance.py    # NoWaitFlowShopInstance, delay matrix computation
        │   │   │   ├── heuristics.py  # NN, NEH-NW, Gangadharan-Rajendran
        │   │   │   ├── metaheuristics.py  # Iterated Greedy for no-wait
        │   │   │   └── README.md
        │   │   ├── blocking/          # Fm | prmu, blocking | Cmax
        │   │   │   ├── instance.py    # BlockingFlowShopInstance, departure times
        │   │   │   ├── heuristics.py  # NEH-B, Profile Fitting
        │   │   │   ├── metaheuristics.py  # Iterated Greedy for blocking
        │   │   │   └── README.md
        │   │   └── setup_times/       # Fm | prmu, Ssd | Cmax
        │   │       ├── instance.py    # SDSTFlowShopInstance, setup time matrices
        │   │       ├── heuristics.py  # NEH-SDST, GRASP-SDST
        │   │       ├── metaheuristics.py  # Iterated Greedy for SDST
        │   │       └── README.md
        │   └── tests/
        │       ├── test_flow_shop.py       # 57 tests, original PFSP algorithms
        │       ├── test_new_algorithms.py  # 38 tests, new algorithms + variants
        │       └── test_ts_aco_sdst.py     # 35 tests, TS, ACO, SDST variant
        ├── parallel_machine/ # FULLY IMPLEMENTED (7 Python files, 43-test suite)
        │   ├── instance.py              # ParallelMachineInstance (Pm, Qm, Rm)
        │   ├── exact/
        │   │   └── mip_makespan.py      # MIP formulation via SciPy HiGHS
        │   ├── heuristics/
        │   │   ├── lpt.py               # LPT (4/3 approx) + SPT for ΣCj
        │   │   ├── multifit.py          # MULTIFIT (1.22 approx), FFD + binary search
        │   │   └── list_scheduling.py   # Greedy list scheduling (2-1/m approx)
        │   ├── metaheuristics/
        │   │   └── genetic_algorithm.py # GA with integer-vector encoding
        │   └── tests/
        │       └── test_parallel_machine.py  # 43 tests, 8 test classes
        ├── single_machine/   # FULLY IMPLEMENTED (7 Python files, 55-test suite)
        │   ├── instance.py              # SingleMachineInstance, objective functions
        │   ├── exact/
        │   │   ├── dynamic_programming.py  # Bitmask DP for 1||ΣTj, O(2^n * n)
        │   │   └── branch_and_bound.py     # B&B for 1||ΣwjTj, ATC warm-start
        │   ├── heuristics/
        │   │   ├── dispatching_rules.py    # SPT, WSPT, EDD, LPT — O(n log n)
        │   │   ├── moores_algorithm.py     # 1||ΣUj — O(n log n)
        │   │   └── apparent_tardiness_cost.py  # ATC for 1||ΣwjTj — O(n²)
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py  # SA for ΣwjTj and ΣTj
        │   └── tests/
        │       └── test_single_machine.py  # 55 tests, 12 test classes
        ├── parallel_machine/ # Scaffolding only
        ├── job_shop/         # Scaffolding only
        ├── flexible_job_shop/# Scaffolding only
        └── rcpsp/            # Scaffolding only
```

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all scheduling tests (228 tests)
python -m pytest problems/scheduling/ -v

# Run all flow shop tests (130 tests)
python -m pytest problems/scheduling/flow_shop/tests/ -v

# Run parallel machine tests (43 tests)
python -m pytest problems/scheduling/parallel_machine/tests/ -v

# Run single machine tests (55 tests)
python -m pytest problems/scheduling/single_machine/tests/ -v

# Run specific test class
python -m pytest problems/scheduling/flow_shop/tests/test_flow_shop.py::TestNEH -v

# Run benchmarks (example: 20 jobs, 5 machines)
python problems/scheduling/flow_shop/benchmark_runner.py --class 20_5 --all
```

### Dependencies

**Core**: numpy (>=1.24), scipy (>=1.10), pandas (>=2.0), matplotlib (>=3.7), pytest (>=7.3)
**Optional**: ortools (>=9.6), gurobipy (>=10.0), pyomo (>=6.5)

## Flow Shop Problem Family

### Standard Permutation Flow Shop (Fm | prmu | Cmax)

All n jobs must be processed on m machines in the same order. The objective
is to find the job permutation that minimizes the makespan (completion time
of the last job on the last machine). NP-hard for m >= 3.

**Exact methods:**
- Johnson's Rule — optimal for F2||Cmax in O(n log n)
- Branch & Bound — Taillard (1993) lower bounds, NEH warm-start, practical for n <= ~20
- MIP — position-based formulation via SciPy HiGHS or OR-Tools CP-SAT

**Constructive heuristics** (fast, single-pass):
- Palmer's Slope Index (1965) — weakest, O(n*m + n log n)
- Gupta's Algorithm (1971) — bottleneck-aware priority, O(n*m + n log n)
- Dannenbring's Rapid Access (1977) — weighted Johnson's reduction, O(n*m + n log n)
- CDS (Campbell-Dudek-Smith, 1970) — m-1 virtual 2-machine sub-problems, O(m * n log n)
- NEH (Nawaz-Enscore-Ham, 1983) — best constructive heuristic, O(n^2 * m)
- LR (Liu & Reeves, 2001) — multi-candidate composite index, O(n^3 * m)

**Metaheuristics** (iterative improvement):
- Local Search — swap, insertion, or-opt, VND neighborhoods
- Simulated Annealing (Osman & Potts, 1989) — Boltzmann acceptance, insertion neighborhood
- Genetic Algorithm (Reeves, 1995) — OX crossover, insertion mutation, steady-state
- Tabu Search (Nowicki & Smutnicki, 1996) — short-term memory, aspiration criterion
- Ant Colony Optimization (Stützle, 1998) — pheromone trails, MMAS bounds, elitist update
- Iterated Greedy (Ruiz & Stuetzle, 2007) — state-of-the-art destroy-and-repair

### No-Wait Flow Shop (Fm | prmu, no-wait | Cmax)

Jobs cannot wait between consecutive machines — processing must be contiguous.
Reduces to an asymmetric TSP on the inter-job delay matrix. NP-hard for m >= 3.

**Applications:** Steel manufacturing, chemical processing, food processing.

**Algorithms:** Nearest Neighbor, NEH-NW, Gangadharan-Rajendran heuristic,
Iterated Greedy (IG-NW).

### Blocking Flow Shop (Fm | prmu, blocking | Cmax)

No intermediate buffers between machines — a completed job blocks its machine
until the next machine is available. Uses departure time recursion instead of
standard completion times. NP-hard for m >= 3.

**Applications:** Manufacturing with limited buffers, robotic cells, paint shops.

**Algorithms:** NEH-B, Profile Fitting, Iterated Greedy (IG-B).

### Sequence-Dependent Setup Times Flow Shop (Fm | prmu, Ssd | Cmax)

Setup times depend on both the current and preceding job on each machine.
Models real-world changeover times that vary by product sequence. NP-hard
for m >= 2.

**Applications:** Printing (color changeovers), chemical processing, automotive
manufacturing, semiconductor fabrication, food processing.

**Algorithms:** NEH-SDST (setup-aware workload sorting), GRASP-SDST (randomized
greedy with local search), Iterated Greedy (IG-SDST).

## Parallel Machine Problem Family

### Problem Definition (Pm | β | γ)

n jobs must be assigned to m parallel machines. Three machine environments:
- **Identical (Pm)**: all machines have equal speed
- **Uniform (Qm)**: machine i has speed s_i
- **Unrelated (Rm)**: processing time p_ij depends on both job and machine

Primary objective: Cmax (makespan). NP-hard even for P2||Cmax (reduces to PARTITION).

**Exact methods:**
- MIP — assignment-based formulation via SciPy HiGHS

**Constructive heuristics:**
- LPT (Longest Processing Time) — 4/3 - 1/(3m) approximation for Cmax
- SPT (Shortest Processing Time) — optimal for Pm||ΣCj with round-robin
- MULTIFIT (Coffman, Garey & Johnson, 1978) — FFD + binary search, 1.22 approximation
- List Scheduling (Graham, 1966) — 2 - 1/m approximation

**Metaheuristics:**
- Genetic Algorithm — integer-vector encoding, uniform crossover, load-balancing LS

## Single Machine Problem Family

### Problem Definition (1 | β | γ)

n jobs processed on one machine. The schedule is fully determined by the
processing order. Multiple objectives supported, ranging from polynomial-time
solvable to strongly NP-hard.

### Tractable Objectives (polynomial-time optimal rules)

| Objective | Rule | Complexity | Reference |
|-----------|------|------------|-----------|
| 1 \|\| ΣCj | SPT (Shortest Processing Time) | O(n log n) | Conway et al. (1967) |
| 1 \|\| ΣwjCj | WSPT (Smith's Rule: sort by pj/wj) | O(n log n) | Smith (1956) |
| 1 \|\| Lmax | EDD (Earliest Due Date) | O(n log n) | Jackson (1955) |
| 1 \|\| ΣUj | Moore's Algorithm | O(n log n) | Moore (1968) |

### NP-Hard Objectives

| Objective | Methods | Reference |
|-----------|---------|-----------|
| 1 \|\| ΣTj | DP (bitmask, exact for n ≤ 20), SA | Lawler (1977), Du & Leung (1990) |
| 1 \|\| ΣwjTj | B&B (ATC warm-start), ATC heuristic, SA | Potts & Van Wassenhove (1985) |

**Constructive heuristics:**
- SPT, WSPT, EDD, LPT — optimal dispatching rules for tractable objectives
- Moore's Algorithm — greedy EDD-based with longest-job removal for ΣUj
- ATC (Apparent Tardiness Cost) — composite dispatching rule for ΣwjTj, O(n²)

**Exact methods:**
- Dynamic Programming — bitmask DP for 1 || ΣTj, O(2^n × n), practical for n ≤ 20
- Branch and Bound — DFS with EDD lower bounds for 1 || ΣwjTj, ATC warm-start

**Metaheuristics:**
- Simulated Annealing — swap/insertion neighborhood for ΣwjTj and ΣTj

## Code Conventions

### Architecture Pattern

Every problem follows this structure:

1. **`instance.py`** — Problem and Solution dataclasses at problem root
2. **`exact/`** — Optimal solvers (B&B, MIP, DP, polynomial algorithms)
3. **`heuristics/`** — Constructive heuristics (fast, approximate)
4. **`metaheuristics/`** — Improvement methods (local search, population-based)
5. **`tests/`** — pytest suite
6. **`benchmark_runner.py`** — CLI to evaluate on standard benchmarks
7. **`variants/`** — Problem variants with modified constraints

### File Template

Each algorithm file must contain:
- **Module docstring** with algorithm name, problem notation (alpha | beta | gamma), complexity, and references with DOI links
- **Dataclass-based** instances and solutions (using `@dataclass`)
- **Main solve function** as the entry point (e.g., `neh()`, `simulated_annealing()`)
- **`if __name__ == "__main__"` block** with example usage and comparisons
- **Type hints** on all function signatures

### Naming & Style

- **PEP 8**: snake_case for functions/variables, PascalCase for classes, CONSTANT_CASE for constants
- **Numpy arrays** for numerical data (processing times, completion matrices)
- **Docstrings**: Google-style with Args, Returns, Raises sections
- **No global state**: all state passed explicitly via function parameters
- **Determinism**: random algorithms must accept a `seed` parameter
- **Warm-starts**: exact methods should warm-start with best constructive heuristic (typically NEH)

### Import Conventions

- Relative imports within same problem directory: `from instance import FlowShopInstance`
- Absolute imports for shared modules: `from shared.parsers.taillard_parser import load_taillard_instance`
- `sys.path.insert()` used for nested directory access (established pattern in this repo)
- **Variant modules** use `importlib.util` for explicit file-path imports to avoid name collisions with parent `instance.py`
- Optional imports wrapped in try/except (e.g., `ortools`, `gurobipy`)

### Data Structure Conventions

```python
@dataclass
class ProblemInstance:
    n: int                          # primary dimension
    param: np.ndarray               # numpy for numerical data

    @classmethod
    def from_file(cls, filepath: str) -> 'ProblemInstance': ...

    @classmethod
    def random(cls, **kwargs) -> 'ProblemInstance': ...

@dataclass
class Solution:
    objective: int | float
    representation: list[int]       # e.g., permutation

    def __repr__(self) -> str: ...
```

### Testing Conventions

- **pytest** framework with fixtures and parametrization
- Run tests with `python -m pytest` to ensure correct module resolution
- Test categories per algorithm: correctness on small handcrafted instances, comparison against known optima, edge cases (single job/machine), benchmark validation on Taillard instances
- Test class naming: `TestAlgorithmName` (e.g., `TestNEH`, `TestJohnsonsRule`)
- Variant tests use `importlib.util` based module loading to avoid import collisions
- Large instance tests should be marked or have reasonable timeouts

### Documentation Per Problem

Each problem folder should contain:
- **README.md**: Problem definition, mathematical formulation, complexity
- **BENCHMARKS.md**: Standard benchmark instances with URLs
- **LITERATURE.md**: Key papers with DOI links and one-sentence summaries

## Key Domain Concepts

- **Scheduling notation**: alpha | beta | gamma (machine environment | constraints | objective)
- **PFSP**: Permutation Flow Shop Problem — all jobs follow same machine order
- **NWFSP**: No-Wait Flow Shop — jobs proceed without waiting between machines
- **BFSP**: Blocking Flow Shop — no intermediate buffers between machines
- **Makespan (Cmax)**: Completion time of last job — primary objective
- **RPD**: Relative Percentage Deviation from best known solution — primary benchmark metric
- **Taillard benchmarks**: 120 standard PFSP instances across 12 size classes (20-500 jobs, 5-20 machines)
- **Delay matrix**: For NWFSP, asymmetric matrix D[j][k] giving minimum start-to-start gap between jobs
- **Departure time**: For BFSP, time when a job leaves a machine (may be after processing completes due to blocking)
- **Tardiness**: Tj = max(0, Cj - dj) — lateness clipped at zero
- **Weighted tardiness**: ΣwjTj — strongly NP-hard single machine objective
- **SPT/WSPT/EDD**: Optimal dispatching rules for tractable single machine objectives
- **ATC**: Apparent Tardiness Cost — composite dispatching rule combining WSPT ratio with due date urgency

## Adding New Problems

Follow the pattern established by `problems/scheduling/flow_shop/`:

1. Create the problem directory under appropriate family
2. Implement `instance.py` with dataclass-based Instance and Solution
3. Add algorithms in `exact/`, `heuristics/`, `metaheuristics/` subdirectories
4. Write comprehensive tests in `tests/`
5. Add benchmark runner if standard benchmarks exist
6. Include README.md, BENCHMARKS.md, LITERATURE.md documentation
7. For variants of an existing problem, use the `variants/` subdirectory pattern
8. See `CONTRIBUTING.md` for the full template and guidelines

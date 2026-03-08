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
        ├── flow_shop/      # FULLY IMPLEMENTED (13 Python files, 668-line test suite)
        │   ├── instance.py             # FlowShopInstance, FlowShopSolution dataclasses
        │   ├── benchmark_runner.py     # CLI for evaluating algorithms on Taillard instances
        │   ├── exact/
        │   │   ├── johnsons_rule.py    # Optimal for F2||Cmax, O(n log n)
        │   │   ├── mip_formulation.py  # SciPy HiGHS + OR-Tools CP-SAT solvers
        │   │   └── branch_and_bound.py # Taillard lower bound, NEH warm-start
        │   ├── heuristics/
        │   │   ├── palmers_slope.py    # Slope index, O(n*m + n log n)
        │   │   ├── guptas_algorithm.py # Priority-based, O(n*m + n log n)
        │   │   ├── cds.py             # Campbell-Dudek-Smith, O(m * n log n)
        │   │   ├── neh.py             # Nawaz-Enscore-Ham, O(n^2 * m), best constructive
        │   │   └── lr_heuristic.py    # Multi-candidate greedy, O(n^3 * m)
        │   ├── metaheuristics/
        │   │   ├── iterated_greedy.py  # Ruiz & Stützle (2007), state-of-the-art for PFSP
        │   │   └── local_search.py     # Swap, insertion, or-opt, VND neighborhoods
        │   └── tests/
        │       └── test_flow_shop.py   # 12 test classes, comprehensive coverage
        ├── single_machine/   # Scaffolding only (docs, no implementations)
        ├── parallel_machine/ # Scaffolding only
        ├── job_shop/         # Scaffolding only
        ├── flexible_job_shop/# Scaffolding only
        └── rcpsp/            # Scaffolding only
```

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest problems/scheduling/flow_shop/tests/test_flow_shop.py -v

# Run specific test class
pytest problems/scheduling/flow_shop/tests/test_flow_shop.py::TestNEH -v

# Run benchmarks (example: 20 jobs, 5 machines)
python problems/scheduling/flow_shop/benchmark_runner.py --class 20_5 --all
```

### Dependencies

**Core**: numpy (>=1.24), scipy (>=1.10), pandas (>=2.0), matplotlib (>=3.7), pytest (>=7.3)
**Optional**: ortools (>=9.6), gurobipy (>=10.0), pyomo (>=6.5)

## Code Conventions

### Architecture Pattern

Every problem follows this structure:

1. **`instance.py`** — Problem and Solution dataclasses at problem root
2. **`exact/`** — Optimal solvers (B&B, MIP, DP, polynomial algorithms)
3. **`heuristics/`** — Constructive heuristics (fast, approximate)
4. **`metaheuristics/`** — Improvement methods (local search, population-based)
5. **`tests/`** — pytest suite
6. **`benchmark_runner.py`** — CLI to evaluate on standard benchmarks

### File Template

Each algorithm file must contain:
- **Module docstring** with algorithm name, problem notation (α | β | γ), complexity, and references
- **Dataclass-based** instances and solutions (using `@dataclass`)
- **`solve()` function** as the main entry point
- **`if __name__ == "__main__"` block** with example usage
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
- Test categories per algorithm: correctness on small handcrafted instances, comparison against known optima, edge cases (single job/machine), benchmark validation on Taillard instances
- Test class naming: `TestAlgorithmName` (e.g., `TestNEH`, `TestJohnsonsRule`)
- Large instance tests should be marked or have reasonable timeouts

### Documentation Per Problem

Each problem folder should contain:
- **README.md**: Problem definition, mathematical formulation, complexity
- **BENCHMARKS.md**: Standard benchmark instances with URLs
- **LITERATURE.md**: Key papers with DOI links and one-sentence summaries

## Key Domain Concepts

- **Scheduling notation**: α | β | γ (machine environment | constraints | objective)
- **PFSP**: Permutation Flow Shop Problem — all jobs follow same machine order
- **Makespan (Cmax)**: Completion time of last job — primary objective
- **RPD**: Relative Percentage Deviation from best known solution — primary benchmark metric
- **Taillard benchmarks**: 120 standard PFSP instances across 12 size classes (20-500 jobs, 5-20 machines)

## Adding New Problems

Follow the pattern established by `problems/scheduling/flow_shop/`:

1. Create the problem directory under appropriate family
2. Implement `instance.py` with dataclass-based Instance and Solution
3. Add algorithms in `exact/`, `heuristics/`, `metaheuristics/` subdirectories
4. Write comprehensive tests in `tests/`
5. Add benchmark runner if standard benchmarks exist
6. Include README.md, BENCHMARKS.md, LITERATURE.md documentation
7. See `CONTRIBUTING.md` for the full template and guidelines

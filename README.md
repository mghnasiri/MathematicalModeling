# Mathematical Modeling — Operations Research

A structured repository of Operations Research problems with mathematical formulations, algorithm implementations, benchmark references, and literature links.

Built as a learning and research companion for OR practitioners and researchers.

---

## Problem Taxonomy

```
Operations Research Problems
│
├── 1. Scheduling  ◄── Phase 1 (Active)
│   ├── Single Machine Scheduling
│   ├── Parallel Machine Scheduling
│   ├── Flow Shop Scheduling (FSP / PFSP)
│   ├── Job Shop Scheduling (JSP)
│   ├── Flexible Job Shop Scheduling (FJSP)
│   └── Resource-Constrained Project Scheduling (RCPSP)
│
├── 2. Routing
│   ├── Traveling Salesman Problem (TSP)
│   ├── Vehicle Routing Problem (VRP / CVRP / VRPTW)
│   └── Arc Routing (CARP)
│
├── 3. Packing & Cutting
│   ├── Knapsack (0-1, Bounded, Multidimensional)
│   ├── Bin Packing (1D, 2D, 3D)
│   └── Cutting Stock
│
├── 4. Location & Layout
│   ├── Facility Location (UFL, CFL)
│   ├── p-Median / p-Center
│   ├── Hub Location
│   └── Quadratic Assignment (QAP)
│
├── 5. Network Optimization
│   ├── Shortest Path / SPPRC
│   ├── Max Flow / Min Cut
│   ├── Min Cost Flow
│   └── Network Design
│
├── 6. Assignment & Matching
│   ├── Linear Assignment (LAP)
│   ├── Generalized Assignment (GAP)
│   └── Bipartite / Weighted Matching
│
├── 7. Set & Covering
│   ├── Set Covering / Set Partitioning
│   ├── Graph Coloring
│   └── Maximum Clique / Independent Set
│
├── 8. Supply Chain & Inventory
│   ├── Lot Sizing (CLSP, DLSP)
│   ├── Inventory Routing
│   └── Supply Chain Network Design
│
└── 9. Stochastic & Robust Optimization
    ├── Two-Stage Stochastic Programming
    ├── Robust Optimization
    └── Chance-Constrained Programming
```

---

## Repository Structure

```
MathematicalModeling/
├── problems/
│   └── {family}/
│       └── {problem}/
│           ├── README.md          # Definition, formulation, complexity
│           ├── BENCHMARKS.md      # Standard test instances & sources
│           ├── LITERATURE.md      # Key papers & recent articles (links)
│           ├── exact/             # Exact algorithms
│           ├── heuristics/        # Constructive & improvement heuristics
│           ├── metaheuristics/    # GA, SA, Tabu, ACO, etc.
│           └── tests/             # Validation against benchmarks
├── shared/
│   ├── metaheuristics/            # Generic metaheuristic frameworks
│   ├── exact/                     # Reusable exact method components
│   ├── parsers/                   # Benchmark file parsers
│   └── visualization/            # Solution visualization tools
└── docs/
    ├── taxonomy.md                # Full problem classification
    ├── complexity_reference.md    # Complexity classes & reductions
    └── solver_guide.md           # Setting up Gurobi, CPLEX, OR-Tools
```

---

## Each Problem Includes

| Section | Content |
|---------|---------|
| **Definition** | Plain-English description + mathematical formulation (LaTeX) |
| **Complexity** | NP-hardness, polynomial special cases, approximation ratios |
| **Solution Approaches** | Table of exact, heuristic, and metaheuristic methods |
| **Benchmarks** | Links to standard test instance libraries |
| **Literature** | Foundational papers + recent high-impact articles |
| **Implementations** | Python (primary), with C++ and GAMS for selected problems |

---

## Scheduling Notation

This project uses the standard **three-field notation** $\alpha \mid \beta \mid \gamma$ introduced by Graham et al. (1979):

| Field | Meaning | Examples |
|-------|---------|---------|
| $\alpha$ | Machine environment | $1$ (single), $P_m$ (parallel identical), $F_m$ (flow shop), $J_m$ (job shop) |
| $\beta$ | Job characteristics & constraints | $r_j$ (release dates), $p_{ij}$ (machine-dependent), $prec$ (precedence) |
| $\gamma$ | Objective function | $C_{\max}$ (makespan), $\sum C_j$ (total completion), $\sum w_j T_j$ (weighted tardiness) |

---

## Languages

- **Python** — primary implementation language (all problems)
- **C++** — performance-critical algorithms (selected problems)
- **GAMS** — mathematical programming formulations (LP/MILP models)

---

## Getting Started

```bash
# Clone
git clone https://github.com/mghnasiri/MathematicalModeling.git
cd MathematicalModeling

# Install Python dependencies
pip install -r requirements.txt

# Run a specific algorithm
python problems/scheduling/job_shop/heuristics/dispatching_rules.py

# Run tests against benchmarks
pytest problems/scheduling/job_shop/tests/
```

---

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding problems and algorithms.

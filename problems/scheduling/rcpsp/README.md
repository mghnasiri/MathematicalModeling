# Resource-Constrained Project Scheduling (RCPSP)

## Problem Definition

Given a project consisting of $n$ activities (jobs) with precedence constraints and resource requirements, schedule all activities to minimize the project duration (makespan). At any point in time, the total resource usage of concurrent activities must not exceed the available capacity for each resource type.

RCPSP generalizes many scheduling problems and is widely used in project management, construction, manufacturing, and software development.

---

## Variants

| Variant | Notation | Description |
|---------|----------|-------------|
| Basic RCPSP | RCPSP | Renewable resources, minimize makespan |
| Multi-mode RCPSP | MRCPSP | Each activity has multiple execution modes (time/resource trade-off) |
| RCPSP with generalized precedence | RCPSP/GPR | Min/max time lags between activities |
| Multi-skill RCPSP | MS-RCPSP | Workers have different skills |
| Stochastic RCPSP | SRCPSP | Uncertain activity durations |
| RCPSP/max | — | Generalized temporal constraints |
| Preemptive RCPSP | PRCPSP | Activities can be interrupted |

---

## Mathematical Formulation

### Parameters
- $n$ — number of (non-dummy) activities
- $K$ — number of renewable resource types
- $R_k$ — capacity of resource $k$
- $p_j$ — duration of activity $j$
- $r_{jk}$ — resource $k$ requirement of activity $j$
- $E$ — set of precedence pairs $(i, j)$: $i$ must finish before $j$ starts
- Activities $0$ and $n+1$ are dummy source/sink with $p_0 = p_{n+1} = 0$

### Decision Variables
- $s_j$ — start time of activity $j$
- $C_{\max} = s_{n+1}$ — project makespan

### Integer Programming Formulation (time-indexed)

$$\min\ s_{n+1}$$

**Precedence constraints**:

$$s_j \geq s_i + p_i \quad \forall (i, j) \in E$$

**Resource constraints**:

$$\sum_{j: s_j \leq t < s_j + p_j} r_{jk} \leq R_k \quad \forall k, \forall t$$

> In the time-indexed formulation, binary variables $x_{jt} = 1$ if activity $j$ starts at time $t$:

$$\sum_{t \in \mathcal{T}_j} x_{jt} = 1 \quad \forall j$$

$$\sum_{j=1}^{n} r_{jk} \sum_{\tau=\max(0,t-p_j+1)}^{t} x_{j\tau} \leq R_k \quad \forall k, \forall t$$

---

## Complexity

- **Strongly NP-hard** — even with 2 resource types
- Special case of $1$ resource with unit requirements = parallel machine scheduling
- With precedence only (no resource constraints) = longest path (polynomial)
- Decision version (is makespan $\leq T$?) is NP-complete

---

## Solution Approaches

### Exact Methods
| Method | Notes |
|--------|-------|
| Branch & Bound | Demeulemeester & Herroelen (1992) — effective up to ~60 activities |
| Constraint Programming | CP-SAT handles cumulative constraints well |
| SAT / Lazy Clause Generation | Very competitive for RCPSP |
| MIP (time-indexed) | Good LP relaxation, weak for large time horizons |

### Schedule Generation Schemes (SGS)
The core building blocks for RCPSP heuristics:

| Scheme | Description | Property |
|--------|-------------|----------|
| **Serial SGS** | Schedule activities one at a time, earliest feasible start | Generates active schedules |
| **Parallel SGS** | At each time step, schedule all feasible activities | Generates non-delay schedules |

### Priority Rules (used with SGS)
| Rule | Description |
|------|-------------|
| LFT | Latest Finish Time |
| LST | Latest Start Time |
| MTS | Most Total Successors |
| GRPW | Greatest Rank Positional Weight |
| WCS | Worst Case Slack |
| IRSM | Improved Resource-based Slack Method |

### Metaheuristics
| Method | Key Reference | Notes |
|--------|---------------|-------|
| Genetic Algorithm | Hartmann (1998), Kolisch & Hartmann (2006) | Activity-list or random-key encoding |
| Simulated Annealing | Bouleimen & Lecocq (2003) | |
| Tabu Search | Various | |
| Scatter Search | Debels et al. (2006) | |
| Ant Colony Optimization | Merkle et al. (2002) | |
| Iterated Local Search | Various | |

### Lower Bounds
| Method | Notes |
|--------|-------|
| Critical Path | Ignores resource constraints |
| Destructive improvement | Iteratively tighten using binary search |
| LP relaxation | Time-indexed gives strong bounds |
| Energy-based reasoning | From CP |

---

## Implementations in This Repo

```
rcpsp/
├── exact/
│   ├── mip_time_indexed.py        # Time-indexed MIP formulation
│   └── constraint_programming.py   # CP-SAT with cumulative
├── heuristics/
│   ├── serial_sgs.py              # Serial Schedule Generation Scheme
│   ├── parallel_sgs.py            # Parallel Schedule Generation Scheme
│   └── priority_rules.py          # LFT, MTS, GRPW, etc.
├── metaheuristics/
│   ├── genetic_algorithm.py       # GA with activity-list encoding
│   └── simulated_annealing.py     # SA
└── tests/
    └── test_rcpsp.py
```

---

## Key Insight

> The **Schedule Generation Scheme** is RCPSP's secret weapon. Unlike JSP/FSP where solutions are permutations, RCPSP solutions must respect both precedence and resource constraints simultaneously. The SGS decodes a priority list into a feasible schedule — making it possible to apply any metaheuristic to RCPSP by simply searching over priority orderings and feeding them through SGS.

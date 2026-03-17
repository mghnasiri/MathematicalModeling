# Flexible Job Shop Scheduling (FJSP)

## Problem Definition

The FJSP extends the classical Job Shop by allowing each operation to be processed on any machine from a **set of eligible machines**. This introduces a **routing sub-problem** (which machine for each operation) on top of the **sequencing sub-problem** (in what order on each machine).

---

## Variants

| Variant | Description |
|---------|-------------|
| **Total FJSP (T-FJSP)** | Every operation can be processed on any machine |
| **Partial FJSP (P-FJSP)** | Each operation has a subset of eligible machines |
| **FJSP with sequence-dependent setup** | Setup times depend on consecutive operations |
| **Multi-objective FJSP** | Optimize makespan + workload balance + total workload |

---

## Mathematical Formulation

### Parameters
- $n$ — number of jobs, $m$ — number of machines
- $O_{jk}$ — the $k$-th operation of job $j$
- $M_{jk} \subseteq \{1, \ldots, m\}$ — set of eligible machines for $O_{jk}$
- $p_{jki}$ — processing time of $O_{jk}$ on machine $i \in M_{jk}$

### Decision Variables
- $x_{jki} \in \{0, 1\}$ — 1 if operation $O_{jk}$ is assigned to machine $i$
- $s_{jk}$ — start time of operation $O_{jk}$
- $C_{\max}$ — makespan

### Formulation

$$\min\ C_{\max}$$

**Machine assignment** — exactly one machine per operation:

$$\sum_{i \in M_{jk}} x_{jki} = 1 \quad \forall j, k$$

**Precedence** — operations within a job:

$$s_{j,k+1} \geq s_{jk} + \sum_{i \in M_{jk}} p_{jki} \cdot x_{jki} \quad \forall j, k$$

**Disjunctive** — no overlap on machines:

$$s_{jk} + p_{jki} \leq s_{j'k'} + M(1 - y_{jk,j'k'}) \quad \text{if both assigned to machine } i$$

**Makespan**:

$$C_{\max} \geq s_{jk} + \sum_{i \in M_{jk}} p_{jki} \cdot x_{jki} \quad \forall j, k$$

---

## Complexity

- **NP-hard** — generalizes the classical JSP (which is already NP-hard)
- The routing sub-problem alone (without sequencing) is equivalent to the unrelated parallel machine problem
- Even $FJ_2 \mid\mid C_{\max}$ is NP-hard

---

## Solution Approaches

### Exact Methods
| Method | Notes |
|--------|-------|
| MIP | Practical for small instances only |
| Constraint Programming | CP-SAT handles medium instances well |
| Branch & Bound | Limited scalability |

### Hierarchical (Decomposition) Approaches
| Step | Sub-problem | Method |
|------|------------|--------|
| 1. Routing | Assign operations to machines | Local assignment rules, LP relaxation |
| 2. Sequencing | Schedule operations on assigned machines | JSP algorithms |

### Integrated Approaches
| Method | Key Reference | Notes |
|--------|---------------|-------|
| Tabu Search | Mastrolilli & Gambardella (2000) | Two neighborhood structures (assignment + sequencing) |
| Genetic Algorithm | Pezzella et al. (2008) | Integrated routing + sequencing encoding |
| Particle Swarm Optimization | Various | Continuous-to-discrete mapping |
| Simulated Annealing | Various | Combined neighborhoods |
| NSGA-II | For multi-objective variants | Pareto-optimal front |

### Key Neighborhoods
- **Assignment neighborhood**: Move an operation to a different eligible machine
- **Sequencing neighborhood**: Swap/insert operations on the same machine (critical path based)
- **Combined**: Apply both in alternation or jointly

---

## Implementations in This Repo

```
flexible_job_shop/
├── exact/
│   └── mip_fjsp.py                # MIP formulation
├── heuristics/
│   ├── hierarchical.py            # Route-then-sequence decomposition
│   └── dispatching_rules.py       # Adapted rules for FJSP
├── metaheuristics/
│   ├── tabu_search.py             # TS with dual neighborhoods
│   ├── genetic_algorithm.py       # GA (Pezzella-style encoding)
│   └── nsga2.py                   # Multi-objective NSGA-II
└── tests/
    └── test_fjsp.py
```

---

## Key Insight

> The FJSP's difficulty comes from the **interaction between routing and sequencing**: assigning an operation to a faster machine may create a bottleneck for other operations. This coupling is why decomposition approaches (solve routing, then sequencing) often produce suboptimal results, and integrated methods that simultaneously address both decisions tend to perform better.

# Job Shop Scheduling (JSP)

## Problem Definition

Given $n$ jobs and $m$ machines, each job consists of a sequence of operations, where each operation must be processed on a specific machine for a given duration. Different jobs may visit machines in different orders. Determine the start time of each operation to optimize the objective.

The JSP is one of the most computationally challenging scheduling problems and is considered a classic benchmark for combinatorial optimization.

---

## Mathematical Formulation

### Parameters
- $n$ — number of jobs, $m$ — number of machines
- $O_{jk}$ — the $k$-th operation of job $j$
- $\mu_{jk}$ — machine required by operation $O_{jk}$
- $p_{jk}$ — processing time of operation $O_{jk}$

### Decision Variables
- $s_{jk}$ — start time of operation $O_{jk}$
- $C_{\max}$ — makespan

### Disjunctive Formulation

$$\min\ C_{\max}$$

**Precedence constraints** (operations within a job):

$$s_{j,k+1} \geq s_{jk} + p_{jk} \quad \forall j, k$$

**Disjunctive constraints** (no two operations on the same machine overlap):

$$s_{jk} + p_{jk} \leq s_{j'k'} \quad \text{OR} \quad s_{j'k'} + p_{j'k'} \leq s_{jk}$$

for all operations $O_{jk}$, $O_{j'k'}$ sharing the same machine.

**Makespan**:

$$C_{\max} \geq s_{jk} + p_{jk} \quad \forall j, k$$

> The disjunctive graph representation (Roy & Sussmann, 1964) is the standard model: conjunctive arcs enforce job precedence, disjunctive arcs represent machine conflicts to be resolved.

---

## Complexity Analysis

| Problem | Complexity | Notes |
|---------|-----------|-------|
| $J_2 \mid\mid C_{\max}$ | NP-hard | Even with 2 machines |
| $J_m \mid\mid C_{\max}$ | NP-hard (strongly) | Among the hardest CO problems |
| $J_2 \mid n=2 \mid C_{\max}$ | $O(n \log n)$ | Jackson's rule |
| $F_m \mid\mid C_{\max}$ | Special case of JSP | Same route for all jobs |

### Notable Complexity Results
- The 10×10 instance `ft10` (Fisher & Thompson, 1963) remained unsolved for **26 years** (optimal = 930, proven in 1989)
- JSP is in the class of problems where even finding a feasible schedule is NP-complete when deadlines are tight

---

## Solution Approaches

### Exact Methods
| Method | Best For | Notes |
|--------|----------|-------|
| Branch & Bound | Small instances ($n \times m \leq 15 \times 15$) | Brucker et al. (1994) |
| Constraint Programming | Medium instances | CP-SAT (Google OR-Tools) very effective |
| MIP (disjunctive) | General | Big-M or indicator constraints |
| Branch & Price | Large structured instances | Column gen on job schedules |

### Constructive Heuristics
| Method | Description | Notes |
|--------|-------------|-------|
| Dispatching Rules | SPT, LPT, MWR, LWR, FIFO | Simple, fast, moderate quality |
| Shifting Bottleneck | Identify + solve bottleneck machine iteratively | Adams, Balas & Zawack (1988) |
| G&T algorithm | Priority rule-based | Giffler & Thompson (1960) |

### Metaheuristics
| Method | Key Reference | Notes |
|--------|---------------|-------|
| Tabu Search | Nowicki & Smutnicki (2005) — i-TSAB | State-of-the-art for decades |
| Genetic Algorithm | Bierwirth (1995) — permutation encoding | |
| Simulated Annealing | Van Laarhoven et al. (1992) | |
| Path Relinking | Various | |
| Large Neighborhood Search | Various | Destroy-repair on critical path |

### Critical Path-Based Neighborhoods
The **critical path** on the disjunctive graph defines the makespan. The most effective neighborhoods modify operations on the critical path:
- **N1**: Swap adjacent operations on a critical block
- **N5** (Nowicki & Smutnicki): Swap first/last with neighbors in critical blocks
- **N7**: Extended neighborhood with block moves

---

## Implementations in This Repo

```
job_shop/
├── exact/
│   ├── disjunctive_mip.py         # MIP with disjunctive constraints
│   └── constraint_programming.py   # CP-SAT formulation
├── heuristics/
│   ├── dispatching_rules.py       # SPT, LPT, MWR, LWR, FIFO
│   ├── shifting_bottleneck.py     # Adams, Balas & Zawack
│   └── giffler_thompson.py        # G&T active schedule generation
├── metaheuristics/
│   ├── tabu_search.py             # TS with N5 neighborhood
│   ├── genetic_algorithm.py       # GA with permutation encoding
│   └── simulated_annealing.py     # SA with critical path moves
└── tests/
    └── test_job_shop.py
```

---

## Variant Implementations

| Variant | Directory | Description |
|---------|-----------|-------------|
| [No-Wait Job Shop](variants/no_wait/) | `variants/no_wait/` | Jobs cannot wait between consecutive operations |
| [Weighted Tardiness JSP](variants/weighted_tardiness/) | `variants/weighted_tardiness/` | Minimize total weighted tardiness $\Sigma w_j T_j$ |
| [Flexible Tardiness JSP](variants/flexible_tardiness/) | `variants/flexible_tardiness/` | Flexible machine assignment + minimize weighted tardiness |

---

## Key Insight

> The **disjunctive graph** is the central data structure for JSP. Understanding it unlocks everything: the critical path determines the makespan, critical blocks define effective neighborhoods, and the graph structure enables efficient move evaluation. Nearly all state-of-the-art methods operate directly on this graph.

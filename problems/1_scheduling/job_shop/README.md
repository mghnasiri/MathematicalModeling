# Job Shop Scheduling (JSP)

## 1. Problem Definition

- **Input:**
  - A set of $n$ jobs $J = \{1, \ldots, n\}$ and $m$ machines $M = \{1, \ldots, m\}$
  - Each job $j$ has an ordered sequence of operations $O_{j1}, O_{j2}, \ldots, O_{j,n_j}$
  - Each operation $O_{jk}$ requires machine $\mu_{jk} \in M$ for $p_{jk}$ time units
  - Different jobs may visit machines in different orders (job-specific routing)
- **Decision:** Determine the start time $s_{jk}$ of each operation
- **Objective:** Minimize makespan $C_{\max} = \max_{j,k} (s_{jk} + p_{jk})$
- **Constraints:** (1) Operations within a job are processed in order (precedence). (2) Each machine processes at most one operation at a time (disjunctive). (3) No preemption.
- **Classification:** Strongly NP-hard combinatorial optimization
- **Scheduling notation:** $J_m \mid\mid C_{\max}$

### Complexity

| Problem | Complexity | Reference |
|---------|-----------|-----------|
| $J_2 \mid\mid C_{\max}$ | NP-hard | Lenstra & Rinnooy Kan (1979) |
| $J_m \mid\mid C_{\max}$ | Strongly NP-hard | Garey & Johnson (1979) |
| $J_2 \mid n{=}2 \mid C_{\max}$ | $O(n \log n)$ | Jackson's Rule |
| $F_m \mid\mid C_{\max}$ | Special case (same route) | Garey, Johnson & Sethi (1976) |

The 10x10 instance **ft10** (Fisher & Thompson, 1963) remained unsolved for **26 years** — the optimal makespan of 930 was proven in 1989 by Carlier & Pinson using constraint propagation.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $m$ | Number of machines | $\mathbb{Z}^+$ |
| $O_{jk}$ | The $k$-th operation of job $j$ | — |
| $\mu_{jk}$ | Machine required by $O_{jk}$ | $M$ |
| $p_{jk}$ | Processing time of $O_{jk}$ | $\mathbb{R}_{>0}$ |
| $s_{jk}$ | Start time of $O_{jk}$ | $\mathbb{R}_{\geq 0}$ |

### Disjunctive Formulation

$$\min \quad C_{\max} \tag{1}$$

**Precedence constraints** (operations within a job must be sequential):

$$s_{j,k+1} \geq s_{jk} + p_{jk} \quad \forall\, j,\; k = 1, \ldots, n_j{-}1 \tag{2}$$

**Disjunctive constraints** (no overlap on the same machine):

$$s_{jk} + p_{jk} \leq s_{j'k'} \quad \text{OR} \quad s_{j'k'} + p_{j'k'} \leq s_{jk} \tag{3}$$

for all pairs of operations $(O_{jk}, O_{j'k'})$ sharing machine $\mu_{jk} = \mu_{j'k'}$.

**Makespan definition:**

$$C_{\max} \geq s_{jk} + p_{jk} \quad \forall\, j, k \tag{4}$$

### Disjunctive Graph Model (Roy & Sussmann, 1964)

The central data structure. Nodes = operations. **Conjunctive arcs** (directed) encode job precedence. **Disjunctive arcs** (undirected) connect operations sharing a machine. A schedule is obtained by orienting all disjunctive arcs such that no cycle exists. The **critical path** (longest path from source to sink) equals the makespan.

**Critical block:** A maximal sequence of consecutive operations on the same machine that all lie on the critical path. Effective neighborhoods modify operations at the boundaries of critical blocks.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| No-Wait JSP | `variants/no_wait/` | Operations of a job must be processed back-to-back |
| Weighted Tardiness | `variants/weighted_tardiness/` | Minimize $\sum w_j T_j$ instead of $C_{\max}$ |
| Flexible Tardiness | `variants/flexible_tardiness/` | Flexible machine assignment + weighted tardiness |

### 3.1 No-Wait Job Shop

No idle time between consecutive operations of the same job. Much more constrained than standard JSP. Often modeled by modifying the disjunctive graph with fixed time lags.

### 3.2 Weighted Tardiness JSP

Replace the makespan objective with $\min \sum w_j T_j$. This is a regular measure, meaning it cannot decrease if any operation starts later. Dispatching rules like ATC (from single machine) can be adapted.

---

## 4. Benchmark Instances

### Standard Instances

| Instance | Size ($n \times m$) | Optimal | Status |
|----------|-------------------|---------|--------|
| ft06 | 6 × 6 | 55 | Solved |
| ft10 | 10 × 10 | 930 | Solved (1989) |
| ft20 | 20 × 5 | 1165 | Solved |
| la01–la40 | 10×5 to 15×15 | Known | Lawrence (1984) |
| ta01–ta80 | 15×15 to 100×20 | Many open | Taillard (1993) |
| orb01–orb10 | 10 × 10 | Known | Applegate & Cook (1991) |

**URL:** http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir/

### Small Illustrative Instance (ft06)

6 jobs, 6 machines. Each job has 6 operations. Optimal makespan = 55. This instance is small enough to solve by hand or with any method.

---

## 5. Solution Methods

### 5.1 Exact Methods

- **Branch & Bound:** Carlier & Pinson (1989) — constraint propagation on the disjunctive graph. Solved ft10 after 26 years. Practical for $n \times m \leq 15 \times 15$.
- **Constraint Programming:** CP-SAT (OR-Tools) models JSP with interval variables and no-overlap constraints. Very effective for medium instances.
- **MIP:** Big-M disjunctive constraints or indicator variables. Generally weaker than CP.

*Note: This repository does not currently include exact method implementations for JSP. The focus is on heuristic and metaheuristic methods.*

### 5.2 Constructive Heuristics

#### Dispatching Rules (Giffler & Thompson, 1960)

**Idea:** Build a schedule operation by operation. At each step, identify the set of schedulable operations and select one using a priority rule:

| Rule | Priority | Description |
|------|----------|-------------|
| SPT | Ascending $p_{jk}$ | Shortest processing time first |
| LPT | Descending $p_{jk}$ | Longest processing time first |
| MWR | Descending remaining work | Most Work Remaining |
| LWR | Ascending remaining work | Least Work Remaining |
| FIFO | Ascending release time | First In First Out |

The Giffler-Thompson (G&T) procedure generates **active schedules** (no operation can be started earlier without delaying another). Active schedules contain an optimal schedule.

#### Shifting Bottleneck (Adams, Balas & Zawack, 1988)

**Idea:** Iteratively identify the "bottleneck" machine — the one most constraining the makespan. Solve a single-machine sub-problem on each unscheduled machine, select the one with maximum optimal makespan, fix its sequence, then re-optimize previously sequenced machines.

**Quality:** Produces very good initial solutions, often within 5-10% of optimal. Combines single-machine exact algorithms with iterative refinement.

### 5.3 Metaheuristics

This repository implements **6 metaheuristic/LS methods** for JSP:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Critical-path swap/insertion neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | N1 neighborhood with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | N1/N5 neighborhood with aspiration criterion |
| 4 | Iterated Greedy (IG) | Trajectory | Remove operations + reconstruct via dispatching |
| 5 | Genetic Algorithm (GA) | Population | Permutation-with-repetition encoding |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Systematic N1 → swap → block-move |

### Critical-Path Neighborhoods

The most effective JSP neighborhoods operate on **critical blocks** of the disjunctive graph:

- **N1 (Van Laarhoven et al., 1992):** Swap any two adjacent operations in a critical block. Size: $O(m)$ moves per critical path.
- **N5 (Nowicki & Smutnicki, 1996):** Swap only the first/last operation of each critical block with its neighbor. Smaller than N1, but empirically as effective.

---

## 6. Implementation Guide

### Modeling Tips

- **Disjunctive graph:** Store as adjacency lists with operation nodes. Compute the critical path via topological sort + longest-path DP in $O(V + E)$.
- **Makespan after a swap:** After swapping two adjacent operations on a machine, only the affected subgraph needs re-evaluation. Full critical path recomputation is $O(V + E)$, but incremental evaluation can be $O(m + n)$.
- **Feasibility check:** A swap is feasible (no cycle) if and only if the resulting graph remains a DAG. For adjacent critical operations, this is guaranteed by the N1 structure.

### Common Pitfalls

- **Conjunctive vs. disjunctive arcs:** Conjunctive arcs have fixed orientation (job precedence). Only disjunctive arcs (machine sequencing) are decision variables.
- **Active vs. semi-active schedules:** A semi-active schedule has no unnecessary idle time; an active schedule additionally ensures no operation can start earlier. Active schedules are preferable.

---

## 7. Computational Results Summary

| Method | Gap on ft06 (6×6) | Gap on ft10 (10×10) | Gap on ta (15-100) |
|--------|-------------------|---------------------|---------------------|
| Dispatching (MWR) | 5-15% | 10-20% | 15-30% |
| Shifting Bottleneck | 0-5% | 3-8% | 5-15% |
| SA (N1) | 0% | 0-2% | 2-5% |
| TS (N5) | 0% | 0% | 1-3% |
| GA | 0% | 0-1% | 2-5% |

**State-of-the-art:** Tabu Search with N5 neighborhood (Nowicki & Smutnicki) remains competitive. Recent hybrid methods combining TS with path relinking achieve <1% gap on most Taillard instances.

---

## 8. Implementations in This Repository

```
job_shop/
├── instance.py                        # JobShopInstance, disjunctive graph, ft06/ft10
│
├── heuristics/
│   ├── dispatching_rules.py           # SPT, LPT, MWR, LWR, FIFO (G&T active schedules)
│   └── shifting_bottleneck.py         # Adams-Balas-Zawack (1988)
│
├── metaheuristics/
│   ├── local_search.py                # Critical-path swap/insertion
│   ├── simulated_annealing.py         # SA with N1 neighborhood
│   ├── tabu_search.py                 # TS with N1/N5 + aspiration
│   ├── iterated_greedy.py             # IG: remove + dispatching reconstruct
│   ├── genetic_algorithm.py           # GA: permutation-with-repetition
│   └── vns.py                         # VNS: N1 → swap → block-move
│
├── variants/
│   ├── no_wait/                       # No-wait JSP
│   ├── weighted_tardiness/            # Jm || ΣwjTj
│   └── flexible_tardiness/            # FJSP + ΣwjTj
│
└── tests/                             # 6 test files
    ├── conftest.py                    # Shared fixtures
    ├── test_job_shop.py               # Core algorithms
    ├── test_job_shop_ga.py            # Genetic Algorithm
    ├── test_job_shop_ig.py            # Iterated Greedy
    ├── test_job_shop_vns.py           # VNS
    └── test_jsp_ls.py                 # Local Search
```

**Total:** 2 heuristic methods, 6 metaheuristics/LS, 3 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Fisher, H. & Thompson, G.L. (1963). Probabilistic learning combinations of local job-shop scheduling rules. *Industrial Scheduling*, 225-251.
- Giffler, B. & Thompson, G.L. (1960). Algorithms for solving production-scheduling problems. *Operations Research*, 8(4), 487-503.
- Roy, B. & Sussmann, B. (1964). Les problemes d'ordonnancement avec contraintes disjonctives. *Note DS no. 9 bis, SEMA*, Paris.
- Adams, J., Balas, E. & Zawack, D. (1988). The shifting bottleneck procedure for job shop scheduling. *Management Science*, 34(3), 391-401.
- Carlier, J. & Pinson, E. (1989). An algorithm for solving the job-shop problem. *Management Science*, 35(2), 164-176.

### Key Metaheuristic References

- Van Laarhoven, P.J.M., Aarts, E.H.L. & Lenstra, J.K. (1992). Job shop scheduling by simulated annealing. *Operations Research*, 40(1), 113-125.
- Nowicki, E. & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. *Management Science*, 42(6), 797-813.

### Surveys

- Jain, A.S. & Meeran, S. (1999). Deterministic job-shop scheduling: Past, present and future. *European Journal of Operational Research*, 113(2), 390-434.
- Zhang, J., Ding, G., Zou, Y., Qin, S. & Fu, J. (2019). Review of job shop scheduling research and its new perspectives under Industry 4.0. *Journal of Intelligent Manufacturing*, 30(4), 1809-1830.

### Key Insight

> The **disjunctive graph** is the central data structure for JSP. Understanding it unlocks everything: the critical path determines the makespan, critical blocks define effective neighborhoods, and the graph structure enables efficient move evaluation. Nearly all state-of-the-art methods operate directly on this graph.

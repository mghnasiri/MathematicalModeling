# Single Machine Scheduling (1 | beta | gamma)

## 1. Problem Definition

- **Input:**
  - A set $N = \{1, 2, \ldots, n\}$ of jobs
  - Processing times $p_j > 0$, weights $w_j \geq 0$, due dates $d_j$, release dates $r_j \geq 0$
- **Decision:** Find a permutation $\pi = (\pi(1), \pi(2), \ldots, \pi(n))$ of jobs
- **Objective:** Minimize a scheduling objective (see Section 2)
- **Constraints:** One machine processes one job at a time. No preemption (unless specified). Completion times: $C_{\pi(1)} = r_{\pi(1)} + p_{\pi(1)}$, $C_{\pi(k)} = \max(C_{\pi(k-1)}, r_{\pi(k)}) + p_{\pi(k)}$
- **Classification:** The single machine is the foundational scheduling environment. Complexity ranges from trivial to strongly NP-hard depending on the objective.

### Complexity Landscape

| Problem | Complexity | Optimal Rule / Method | Reference |
|---------|-----------|----------------------|-----------|
| $1 \mid\mid C_{\max}$ | Trivial | Any order ($C_{\max} = \sum p_j$) | — |
| $1 \mid\mid \sum C_j$ | $O(n \log n)$ | SPT (Shortest Processing Time) | Conway et al. (1967) |
| $1 \mid\mid \sum w_j C_j$ | $O(n \log n)$ | WSPT (Smith's Rule: sort by $p_j/w_j$) | Smith (1956) |
| $1 \mid\mid L_{\max}$ | $O(n \log n)$ | EDD (Earliest Due Date) | Jackson (1955) |
| $1 \mid\mid \sum U_j$ | $O(n \log n)$ | Moore's Algorithm | Moore (1968) |
| $1 \mid\mid \sum T_j$ | NP-hard | Bitmask DP $O(2^n \cdot n)$ | Du & Leung (1990) |
| $1 \mid\mid \sum w_j T_j$ | Strongly NP-hard | B&B, ATC heuristic | Potts & Van Wassenhove (1985) |
| $1 \mid r_j \mid \sum C_j$ | NP-hard | B&B | Lenstra et al. (1977) |
| $1 \mid r_j \mid L_{\max}$ | NP-hard (preemptive: poly) | SRPT for preemptive | Schrage (1968) |
| $1 \mid prec \mid \sum w_j C_j$ | NP-hard | B&B | Lawler (1978) |
| $1 \mid s_{jk} \mid C_{\max}$ | NP-hard (reduces to TSP) | TSP methods | — |

The single machine is the **foundation** of scheduling theory — every dispatching rule here (SPT, EDD, WSPT) becomes a subroutine for multi-machine algorithms.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $p_j$ | Processing time of job $j$ | $\mathbb{R}_{>0}$ |
| $w_j$ | Weight (priority) of job $j$ | $\mathbb{R}_{\geq 0}$ |
| $d_j$ | Due date of job $j$ | $\mathbb{R}$ |
| $r_j$ | Release date of job $j$ | $\mathbb{R}_{\geq 0}$ |
| $C_j$ | Completion time of job $j$ | $\mathbb{R}_{>0}$ |
| $T_j$ | Tardiness: $T_j = \max(0, C_j - d_j)$ | $\mathbb{R}_{\geq 0}$ |
| $U_j$ | Unit penalty: $U_j = 1$ if $C_j > d_j$, else 0 | $\{0, 1\}$ |
| $L_j$ | Lateness: $L_j = C_j - d_j$ | $\mathbb{R}$ |

### Weighted Completion Time ($1 \mid\mid \sum w_j C_j$)

Optimality of WSPT: For any two adjacent jobs $j, k$ in the schedule, swapping them improves the objective if and only if $p_j / w_j > p_k / w_k$. Therefore, sorting by increasing $p_j / w_j$ (Smith's ratio rule) is optimal.

### Total Weighted Tardiness ($1 \mid\mid \sum w_j T_j$) — MILP

$$\min \sum_{j=1}^{n} w_j T_j \tag{1}$$

$$\text{s.t.} \quad C_j = \sum_{k=1}^{n} p_j \cdot x_{jk} \cdot k' \quad \text{(linearized via position variables)} \tag{2}$$

$$T_j \geq C_j - d_j \quad \forall j \tag{3}$$

$$T_j \geq 0 \quad \forall j \tag{4}$$

In practice, the position-based MILP is weak; B&B with the ATC heuristic as warm-start is more effective.

---

## 3. Variants

| Variant | Directory | Notation | Key Difference |
|---------|-----------|----------|---------------|
| Preemptive | `variants/preemptive/` | $1 \mid pmtn, r_j \mid \sum C_j$ | Jobs can be interrupted and resumed |
| Batch | `variants/batch/` | $1 \mid batch, s_j \mid \sum w_j C_j$ | Jobs grouped into batches with setup times |

### 3.1 Preemptive Scheduling

Jobs can be interrupted and resumed later without penalty. With release dates, the preemptive case $1 \mid pmtn, r_j \mid L_{\max}$ is solvable in $O(n \log n)$ via Shortest Remaining Processing Time (SRPT), while the non-preemptive case is NP-hard.

### 3.2 Batch Scheduling

Jobs are grouped into batches. A setup time occurs between consecutive batches of different types. Objective typically involves minimizing total weighted completion time across all batches.

---

## 4. Benchmark Instances

Single machine scheduling benchmarks are typically generated randomly:

- **OR-Library** (Beasley, 1990): Weighted tardiness instances with $n = 40, 50, 100$.
- **Random generation:** Processing times $p_j \sim U[1, 100]$, due dates $d_j \sim U[P(1-\tau-R/2), P(1-\tau+R/2)]$ where $P = \sum p_j$, $\tau$ = tardiness factor, $R$ = due date range.

### Small Illustrative Instance

```
5 jobs: p = [4, 3, 7, 2, 6], w = [5, 2, 1, 4, 3], d = [8, 12, 10, 5, 15]
SPT order: [4, 2, 1, 5, 3] (sort by p_j)
WSPT order: [4, 1, 5, 2, 3] (sort by p_j/w_j = [0.8, 1.5, 7.0, 0.5, 2.0])
EDD order: [4, 1, 3, 2, 5] (sort by d_j)
```

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Polynomial-Time Dispatching Rules

| Rule | Sorts By | Optimizes | Complexity |
|------|----------|-----------|-----------|
| SPT | Ascending $p_j$ | $\sum C_j$ | $O(n \log n)$ |
| WSPT | Ascending $p_j/w_j$ | $\sum w_j C_j$ | $O(n \log n)$ |
| EDD | Ascending $d_j$ | $L_{\max}$ | $O(n \log n)$ |
| LPT | Descending $p_j$ | Load balancing (used in parallel machine) | $O(n \log n)$ |

#### Moore's Algorithm ($1 \mid\mid \sum U_j$)

**Idea:** Process jobs in EDD order. Whenever a job is tardy, remove the longest job processed so far (it goes to the "late" set). The remaining jobs are on time.

**Complexity:** $O(n \log n)$ using a max-heap for the removal step.

```
ALGORITHM Moore(p[1..n], d[1..n])
  Sort jobs by ascending d_j → sequence S
  on_time ← [], late ← [], t ← 0
  FOR j in S:
    on_time.append(j), t ← t + p[j]
    IF t > d[j]:
      longest ← argmax p[k] for k in on_time
      Remove longest from on_time, add to late
      t ← t - p[longest]
  RETURN on_time + late   (on_time in EDD, late in any order)
```

#### Apparent Tardiness Cost (ATC) — Heuristic for $1 \mid\mid \sum w_j T_j$

**Idea:** Composite dispatching rule combining WSPT ratio with due-date urgency. At each step, schedule the job maximizing:

$$I_j(t) = \frac{w_j}{p_j} \exp\left(-\frac{\max(d_j - p_j - t, 0)}{K \bar{p}}\right)$$

where $K$ is a look-ahead parameter (typically $K \in [1, 3]$) and $\bar{p}$ is the average processing time of remaining jobs.

**Complexity:** $O(n^2)$. Quality: typically within 1-5% of optimal for $n \leq 100$.

#### Dynamic Programming ($1 \mid\mid \sum T_j$)

**Idea:** Bitmask DP. State: $(S, t)$ where $S$ is the set of scheduled jobs and $t$ is the current time. Transition: add any unscheduled job.

**Complexity:** $O(2^n \cdot n)$. **Practical limit:** $n \leq 20$.

#### Branch and Bound ($1 \mid\mid \sum w_j T_j$)

DFS with ATC warm-start upper bound. Lower bound: relaxation of due date constraints or WSPT on remaining jobs. Effective up to $n \approx 50{-}100$ depending on due date tightness.

### 5.2 Metaheuristics (for NP-hard objectives)

This repository implements **6 metaheuristics** for single machine:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Swap, insertion neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | Swap/insertion with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Swap neighborhood with recency tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove + reinsert jobs via ATC |
| 5 | Genetic Algorithm (GA) | Population | Permutation encoding, OX crossover |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Swap → insertion → block-move |

---

## 6. Implementation Guide

### Modeling Tips

- **Completion time computation:** Given a permutation $\pi$, completion times are computed in $O(n)$: $C_{\pi(1)} = p_{\pi(1)}$, $C_{\pi(k)} = C_{\pi(k-1)} + p_{\pi(k)}$.
- **Tardiness:** $T_j = \max(0, C_j - d_j)$. Pre-compute completion times first, then tardiness.
- **ATC parameter $K$:** Start with $K = 2$. Lower values increase greediness (good for tight due dates); higher values make ATC approach WSPT.

### Common Pitfalls

- **WSPT ties:** When $p_j/w_j = p_k/w_k$, either order is optimal for $\sum w_j C_j$. But for $\sum w_j T_j$, tie-breaking matters — use EDD among tied jobs.
- **Moore's Algorithm correctness:** The removed job must be the longest *among all on-time jobs so far*, not the current job. Using a max-heap is essential.

---

## 7. Computational Results Summary

| Method | Objective | Gap (n=20) | Gap (n=100) |
|--------|-----------|-----------|-------------|
| SPT/WSPT/EDD | Polynomial objectives | 0% | 0% |
| Moore's | $\sum U_j$ | 0% | 0% |
| ATC | $\sum w_j T_j$ | 1-5% | 2-8% |
| DP (bitmask) | $\sum T_j$ | 0% | Infeasible |
| B&B + ATC | $\sum w_j T_j$ | 0% | 0% (minutes) |
| SA | $\sum w_j T_j$ | <1% | 1-3% |
| GA | $\sum w_j T_j$ | <1% | 1-3% |

---

## 8. Implementations in This Repository

```
single_machine/
├── instance.py                        # SingleMachineInstance, objective functions
│
├── exact/
│   ├── dynamic_programming.py         # Bitmask DP for 1||ΣTj — O(2^n × n)
│   └── branch_and_bound.py            # B&B for 1||ΣwjTj, ATC warm-start
│
├── heuristics/
│   ├── dispatching_rules.py           # SPT, WSPT, EDD, LPT
│   ├── moores_algorithm.py            # Moore's for 1||ΣUj — O(n log n)
│   └── apparent_tardiness_cost.py     # ATC for 1||ΣwjTj — O(n²)
│
├── metaheuristics/
│   ├── local_search.py                # Swap, insertion neighborhoods
│   ├── simulated_annealing.py         # SA for ΣwjTj and ΣTj
│   ├── tabu_search.py                 # TS with recency tabu
│   ├── iterated_greedy.py             # IG: remove + ATC reinsert
│   ├── genetic_algorithm.py           # GA: permutation encoding, OX
│   └── vns.py                         # VNS: swap → insertion → block-move
│
├── variants/
│   ├── preemptive/                    # 1 | pmtn, rj | ΣCj
│   └── batch/                         # 1 | batch, sj | ΣwjCj
│
└── tests/                             # 6 test files
    ├── test_single_machine.py         # Core algorithms
    ├── test_sm_tabu_search.py         # Tabu Search
    ├── test_sm_ga.py                  # Genetic Algorithm
    ├── test_sm_ig.py                  # Iterated Greedy
    ├── test_sm_ls.py                  # Local Search
    └── test_sm_vns.py                 # VNS
```

**Total:** 2 exact methods, 4 dispatching rules + Moore's + ATC, 6 metaheuristics/LS, 2 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Smith, W.E. (1956). Various optimizers for single-stage production. *Naval Research Logistics Quarterly*, 3(1-2), 59-66.
- Moore, J.M. (1968). An $n$ job, one machine sequencing algorithm for minimizing the number of late jobs. *Management Science*, 15(1), 102-109.
- Jackson, J.R. (1955). Scheduling a production line to minimize maximum tardiness. *Management Science Research Project, Research Report 43, UCLA*.
- Du, J. & Leung, J.Y.T. (1990). Minimizing total tardiness on one machine is NP-hard. *Mathematics of Operations Research*, 15(3), 483-495.

### Key Methods

- Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound algorithm for the total weighted tardiness problem. *Operations Research*, 33(2), 363-377.
- Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035-1047.

### Textbook

- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.

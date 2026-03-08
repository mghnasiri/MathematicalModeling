# Single Machine Scheduling

## Problem Definition

Given $n$ jobs to be processed on a single machine, determine the processing order that optimizes a given objective. The machine can process only one job at a time and preemption may or may not be allowed.

---

## Mathematical Formulation

### Parameters
- $n$ — number of jobs
- $p_j$ — processing time of job $j$
- $w_j$ — weight (priority) of job $j$
- $d_j$ — due date of job $j$
- $r_j$ — release date of job $j$

### Decision Variables
- $C_j$ — completion time of job $j$
- $x_{jk} \in \{0, 1\}$ — 1 if job $j$ is in position $k$

### Common Objectives

**Minimize makespan** $1 \mid\mid C_{\max}$:

$$\min\ C_{\max} = \sum_{j=1}^{n} p_j$$

> Trivial — any sequence gives the same makespan.

**Minimize total weighted completion** $1 \mid\mid \sum w_j C_j$:

$$\min \sum_{j=1}^{n} w_j C_j$$

**Minimize maximum lateness** $1 \mid\mid L_{\max}$:

$$\min\ \max_{j} (C_j - d_j)$$

**Minimize total tardiness** $1 \mid\mid \sum T_j$:

$$\min \sum_{j=1}^{n} \max(0, C_j - d_j)$$

**Minimize number of tardy jobs** $1 \mid\mid \sum U_j$:

$$\min \sum_{j=1}^{n} U_j, \quad U_j = \begin{cases} 1 & \text{if } C_j > d_j \\ 0 & \text{otherwise} \end{cases}$$

---

## Complexity Analysis

| Problem | Complexity | Optimal Rule / Algorithm |
|---------|-----------|------------------------|
| $1 \mid\mid C_{\max}$ | Trivial | Any order |
| $1 \mid\mid \sum C_j$ | $O(n \log n)$ | SPT (Shortest Processing Time) |
| $1 \mid\mid \sum w_j C_j$ | $O(n \log n)$ | WSPT (Weighted SPT: sort by $p_j/w_j$) |
| $1 \mid\mid L_{\max}$ | $O(n \log n)$ | EDD (Earliest Due Date) |
| $1 \mid\mid \sum U_j$ | $O(n \log n)$ | Moore's Algorithm |
| $1 \mid\mid \sum T_j$ | NP-hard | DP, Branch & Bound |
| $1 \mid\mid \sum w_j T_j$ | NP-hard (strongly) | Branch & Bound, metaheuristics |
| $1 \mid r_j \mid \sum C_j$ | NP-hard | Branch & Bound |
| $1 \mid r_j \mid L_{\max}$ | NP-hard (preemptive: $O(n \log n)$ via SRPT) | Branch & Bound |
| $1 \mid prec \mid \sum w_j C_j$ | NP-hard | Branch & Bound |
| $1 \mid s_{jk} \mid C_{\max}$ | NP-hard (reduces to TSP) | TSP heuristics |

---

## Solution Approaches

### Exact Methods
| Method | Best For | Notes |
|--------|----------|-------|
| Dynamic Programming | $1 \mid\mid \sum T_j$ (moderate $n$) | Pseudo-polynomial for some variants |
| Branch & Bound | $1 \mid\mid \sum w_j T_j$ | Effective up to $n \approx 50$-$100$ |
| MIP | All NP-hard variants | General but slow for large $n$ |

### Polynomial-Time Rules
| Rule | Notation | Optimizes |
|------|----------|-----------|
| SPT | Sort ascending by $p_j$ | $\sum C_j$ |
| WSPT | Sort ascending by $p_j / w_j$ | $\sum w_j C_j$ |
| EDD | Sort ascending by $d_j$ | $L_{\max}$ |
| Moore's Algorithm | Greedy + rejection | $\sum U_j$ |
| SRPT | Preemptive, shortest remaining | $1 \mid r_j, pmtn \mid \sum C_j$ |

### Metaheuristics (for NP-hard variants)
| Method | Typical Target |
|--------|---------------|
| Simulated Annealing | $\sum w_j T_j$ |
| Tabu Search | $\sum w_j T_j$ |
| Genetic Algorithm | General |
| VNS | $\sum T_j$ |

---

## Implementations in This Repo

```
single_machine/
├── exact/
│   ├── branch_and_bound.py       # B&B for weighted tardiness
│   └── dynamic_programming.py    # DP for total tardiness
├── heuristics/
│   ├── dispatching_rules.py      # SPT, WSPT, EDD, LPT
│   ├── moores_algorithm.py       # Minimize number of tardy jobs
│   └── apparent_tardiness_cost.py # ATC composite rule
├── metaheuristics/
│   ├── simulated_annealing.py    # SA for weighted tardiness
│   └── tabu_search.py            # Tabu search
└── tests/
    └── test_single_machine.py    # Validation suite
```

---

## Key Insight

> Single machine problems are the **foundation** of scheduling theory. Every polynomial-time dispatching rule you learn here (SPT, EDD, WSPT) becomes a building block for multi-machine heuristics — they're used as subroutines in flow shop, job shop, and RCPSP algorithms.

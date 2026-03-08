# Parallel Machine Scheduling

## Problem Definition

Given $n$ jobs and $m$ identical parallel machines, assign each job to a machine and determine the processing order on each machine to optimize a given objective. Each job must be processed on exactly one machine (non-preemptive) or can be split (preemptive).

---

## Variants

| Notation | Environment | Description |
|----------|------------|-------------|
| $P_m$ | Identical parallel | All machines have the same speed |
| $Q_m$ | Uniform parallel | Machine $i$ has speed $s_i$; processing time $= p_j / s_i$ |
| $R_m$ | Unrelated parallel | Processing time $p_{ij}$ depends on both job and machine |

---

## Mathematical Formulation

### Parameters
- $n$ — number of jobs, $m$ — number of machines
- $p_j$ — processing time of job $j$ (identical machines)
- $p_{ij}$ — processing time of job $j$ on machine $i$ (unrelated)
- $w_j$ — weight of job $j$
- $d_j$ — due date of job $j$

### Decision Variables
- $x_{ij} \in \{0, 1\}$ — 1 if job $j$ assigned to machine $i$
- $C_j$ — completion time of job $j$

### MIP for $P_m \mid\mid C_{\max}$

$$\min\ C_{\max}$$

subject to:

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j$$

$$C_{\max} \geq \sum_{j=1}^{n} p_j \cdot x_{ij} \quad \forall i$$

$$x_{ij} \in \{0, 1\}$$

> This is equivalent to the **Multiprocessor Scheduling Problem** (identical to the number partitioning problem for $m = 2$).

---

## Complexity Analysis

| Problem | Complexity | Notes |
|---------|-----------|-------|
| $P_2 \mid\mid C_{\max}$ | NP-hard (weakly) | Reduces to PARTITION |
| $P_m \mid\mid C_{\max}$ | NP-hard (strongly) for variable $m$ | Reduces to 3-PARTITION |
| $P_m \mid pmtn \mid C_{\max}$ | $O(n)$ | McNaughton's rule |
| $P_m \mid\mid \sum C_j$ | $O(n \log n)$ | SPT + round-robin |
| $P_m \mid\mid \sum w_j C_j$ | NP-hard | WSPT is approx. |
| $Q_m \mid\mid C_{\max}$ | NP-hard | |
| $Q_m \mid pmtn \mid C_{\max}$ | $O(n \log n)$ | Horvath, Lam, Sethi (1977) |
| $R_m \mid\mid C_{\max}$ | NP-hard (strongly) | |

### Approximation Results
- LPT for $P_m \mid\mid C_{\max}$: ratio $\leq 4/3 - 1/(3m)$
- PTAS exists for fixed $m$ (Hochbaum & Shmoys, 1987)
- MULTIFIT: ratio $\leq 1.22$ (Coffman, Garey & Johnson, 1978)

---

## Solution Approaches

### Exact Methods
| Method | Best For | Notes |
|--------|----------|-------|
| Branch & Bound | $P_m \mid\mid C_{\max}$ up to $n \approx 50$ | Dell'Amico & Martello (1995) |
| Branch & Price | Large instances | Column generation on machine schedules |
| MIP | General | Tight for moderate $n$ |

### Heuristics
| Rule | Description | Target |
|------|-------------|--------|
| LPT | Longest Processing Time first | $C_{\max}$ — 4/3 approx |
| SPT | Shortest Processing Time first | $\sum C_j$ — optimal |
| MULTIFIT | Binary search + First Fit Decreasing | $C_{\max}$ — 1.22 approx |
| WSPT | Weighted SPT | $\sum w_j C_j$ |
| List scheduling | Assign next job to least loaded machine | $C_{\max}$ — 2 approx |

### Metaheuristics
| Method | Notes |
|--------|-------|
| Genetic Algorithm | Good for unrelated machines with complex constraints |
| Simulated Annealing | Effective for $\sum w_j T_j$ |
| Tabu Search | Strong for $R_m$ variants |

---

## Implementations in This Repo

```
parallel_machine/
├── exact/
│   └── mip_makespan.py           # MIP formulation for Pm||Cmax
├── heuristics/
│   ├── lpt.py                    # Longest Processing Time
│   ├── multifit.py               # MULTIFIT algorithm
│   └── list_scheduling.py        # List scheduling (greedy)
├── metaheuristics/
│   └── genetic_algorithm.py      # GA for Rm||Cmax
└── tests/
    └── test_parallel_machine.py
```

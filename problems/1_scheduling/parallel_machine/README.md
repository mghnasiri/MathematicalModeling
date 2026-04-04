# Parallel Machine Scheduling (Pm | beta | gamma)

## 1. Problem Definition

- **Input:**
  - A set $N = \{1, \ldots, n\}$ of jobs with processing times $p_j$ (or $p_{ij}$ for unrelated machines), weights $w_j$, due dates $d_j$
  - A set $M = \{1, \ldots, m\}$ of parallel machines
- **Decision:** Assign each job to a machine and determine the processing order on each machine
- **Objective:** Minimize a scheduling criterion (typically makespan $C_{\max}$ or total weighted completion $\sum w_j C_j$)
- **Constraints:** Each job processed on exactly one machine (non-preemptive). Each machine processes one job at a time.
- **Classification:** NP-hard for most objectives (reduces to PARTITION for $P_2 \mid\mid C_{\max}$)

### Machine Environments

| Notation | Type | Speed Model |
|----------|------|------------|
| $P_m$ | Identical | All machines have speed 1 |
| $Q_m$ | Uniform | Machine $i$ has speed $s_i$; time $= p_j / s_i$ |
| $R_m$ | Unrelated | Processing time $p_{ij}$ depends on both job and machine |

### Complexity

| Problem | Complexity | Optimal / Approximation |
|---------|-----------|------------------------|
| $P_2 \mid\mid C_{\max}$ | NP-hard (weakly) | Reduces to PARTITION |
| $P_m \mid\mid C_{\max}$ | Strongly NP-hard (var. $m$) | 3-PARTITION |
| $P_m \mid pmtn \mid C_{\max}$ | $O(n)$ | McNaughton's Rule |
| $P_m \mid\mid \sum C_j$ | $O(n \log n)$ | SPT + round-robin |
| LPT for $P_m \mid\mid C_{\max}$ | $O(n \log n)$ | $4/3 - 1/(3m)$ approximation |
| MULTIFIT | $O(n \log n \cdot \log W)$ | 1.22 approximation |
| PTAS | Polynomial (fixed $m$) | Hochbaum & Shmoys (1987) |

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $m$ | Number of machines | $\mathbb{Z}^+$ |
| $p_j$ | Processing time of job $j$ (identical) | $\mathbb{R}_{>0}$ |
| $p_{ij}$ | Processing time of job $j$ on machine $i$ (unrelated) | $\mathbb{R}_{>0}$ |
| $x_{ij}$ | 1 if job $j$ assigned to machine $i$ | $\{0, 1\}$ |
| $C_{\max}$ | Makespan | $\mathbb{R}_{>0}$ |

### MIP for $P_m \mid\mid C_{\max}$

$$\min \quad C_{\max} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \quad \text{(each job to one machine)} \tag{2}$$

$$C_{\max} \geq \sum_{j=1}^{n} p_j \cdot x_{ij} \quad \forall i \quad \text{(makespan)} \tag{3}$$

$$x_{ij} \in \{0, 1\} \tag{4}$$

Equivalent to the multiprocessor scheduling / number partitioning problem for $m = 2$.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| SDST Parallel Machine | `variants/sdst/` | Sequence-dependent setup times ($R_m \mid S_{sd} \mid C_{\max}$) |
| Unrelated Tardiness | `variants/unrelated_tardiness/` | Minimize tardiness on unrelated machines ($R_m \mid\mid \sum T_j$) |

---

## 4. Benchmark Instances

Standard parallel machine instances are typically generated:
- Processing times $p_j \sim U[1, 100]$
- For unrelated: $p_{ij} \sim U[1, 100]$ independently

### Small Illustrative Instance

```
6 jobs on 3 machines: p = [8, 6, 5, 4, 3, 2]
LPT assignment: M1=[8,3], M2=[6,4], M3=[5,2] → Cmax = 11
Optimal: Cmax = 10 (balance loads as [8,2], [6,4], [5,3])
```

---

## 5. Solution Methods

### 5.1 Exact Methods

**MIP (HiGHS):** Assignment-based formulation (above). Practical for $n \leq 50$ with small $m$.

### 5.2 Constructive Heuristics

| Method | Idea | Approximation |
|--------|------|--------------|
| **LPT** | Assign longest unscheduled job to least-loaded machine | $4/3 - 1/(3m)$ for $C_{\max}$ |
| **SPT** | Sort by ascending $p_j$, round-robin assignment | Optimal for $\sum C_j$ |
| **MULTIFIT** | FFD bin-packing + binary search on makespan | 1.22 for $C_{\max}$ |
| **List Scheduling** | Assign next job to least-loaded machine (any order) | $2 - 1/m$ for $C_{\max}$ |

```
ALGORITHM LPT(p[1..n], m)
  Sort jobs by decreasing p_j
  load[1..m] ← 0
  FOR each job j in sorted order:
    i* ← argmin load[i]
    Assign j to machine i*
    load[i*] ← load[i*] + p[j]
  RETURN assignment
```

### 5.3 Metaheuristics

This repository implements **6 metaheuristics** for parallel machine scheduling:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Reassign, swap between machines |
| 2 | Simulated Annealing (SA) | Trajectory | Reassign/swap with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Job-machine pair tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove jobs + LPT reconstruct |
| 5 | Genetic Algorithm (GA) | Population | Integer-vector encoding (gene = machine) |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Reassign → swap → block-move |

---

## 6. Implementation Guide

- **Encoding:** Solutions are vectors $\sigma \in \{1, \ldots, m\}^n$ where $\sigma_j$ = machine assigned to job $j$. Makespan evaluation: $O(n)$.
- **LPT tie-breaking:** When multiple machines have equal load, use lowest index. This doesn't affect worst-case ratio.
- **MULTIFIT:** Binary search between $\lceil\sum p_j / m\rceil$ (lower bound) and $\max p_j + \sum p_j / m$ (upper bound). Each iteration runs FFD in $O(n \log n)$.

---

## 7. Computational Results Summary

| Method | Gap ($n{=}50$) | Gap ($n{=}200$) |
|--------|-------------|--------------|
| LPT | 1-5% | 1-3% |
| MULTIFIT | <1% | <1% |
| MIP (exact) | 0% | Slow |
| SA | <1% | <1% |
| GA | <1% | 1-2% |

---

## 8. Implementations in This Repository

```
parallel_machine/
├── instance.py                    # ParallelMachineInstance (Pm, Qm, Rm)
├── exact/
│   └── mip_makespan.py            # MIP via SciPy HiGHS
├── heuristics/
│   ├── lpt.py                     # LPT (4/3 approx) + SPT for ΣCj
│   ├── multifit.py                # MULTIFIT (1.22 approx)
│   └── list_scheduling.py         # Greedy list scheduling (2-1/m approx)
├── metaheuristics/
│   ├── local_search.py            # Reassign, swap neighborhoods
│   ├── simulated_annealing.py     # SA with reassign/swap
│   ├── tabu_search.py             # TS with job-machine tabu
│   ├── iterated_greedy.py         # IG: remove + LPT reconstruct
│   ├── genetic_algorithm.py       # GA: integer-vector encoding
│   └── vns.py                     # VNS: reassign → swap → block-move
├── variants/
│   ├── sdst/                      # Rm | Ssd | Cmax
│   └── unrelated_tardiness/       # Rm || ΣTj
└── tests/                         # 6 test files
    ├── test_parallel_machine.py   # Core algorithms
    ├── test_pm_sa.py              # SA
    ├── test_pm_ts.py              # TS
    ├── test_pm_ig.py              # IG
    ├── test_pm_ls.py              # LS
    └── test_pm_vns.py             # VNS
```

**Total:** 1 exact, 3 heuristics, 6 metaheuristics/LS, 2 variants, 6 test files.

---

## 9. Key References

- Graham, R.L. (1969). Bounds on multiprocessing timing anomalies. *SIAM J. Applied Math.*, 17(2), 416-429.
- Coffman, E.G., Garey, M.R. & Johnson, D.S. (1978). An application of bin-packing to multiprocessor scheduling. *SIAM J. Computing*, 7(1), 1-17.
- Hochbaum, D.S. & Shmoys, D.B. (1987). Using dual approximation algorithms for scheduling problems. *JACM*, 34(1), 144-162.
- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.

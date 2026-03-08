# Scheduling Problems

Scheduling is one of the most extensively studied families in Operations Research. The goal is to allocate limited resources (machines, processors, workers) to tasks (jobs) over time, optimizing one or more objectives.

---

## Three-Field Notation: $\alpha \mid \beta \mid \gamma$

All scheduling problems in this repository use the **Graham et al. (1979)** classification:

### $\alpha$ — Machine Environment

| Symbol | Environment | Description |
|--------|------------|-------------|
| $1$ | Single machine | One machine processes all jobs |
| $P_m$ | Identical parallel | $m$ identical machines |
| $Q_m$ | Uniform parallel | Machines have different speeds |
| $R_m$ | Unrelated parallel | Processing times are machine-dependent |
| $F_m$ | Flow shop | $m$ machines in series, all jobs follow same route |
| $J_m$ | Job shop | $m$ machines, each job has its own route |
| $O_m$ | Open shop | $m$ machines, operation order is free |
| $FJ_m$ | Flexible job shop | Job shop + machine flexibility per operation |

### $\beta$ — Job Constraints

| Symbol | Meaning |
|--------|---------|
| $r_j$ | Release dates (job $j$ available at time $r_j$) |
| $d_j$ | Due dates |
| $\bar{d}_j$ | Deadlines (hard due dates) |
| $p_j = p$ | All processing times equal |
| $prec$ | Precedence constraints |
| $s_{jk}$ | Sequence-dependent setup times |
| $pmtn$ | Preemption allowed |
| $M_j$ | Machine eligibility restrictions |

### $\gamma$ — Objective Functions

| Symbol | Name | Formula |
|--------|------|---------|
| $C_{\max}$ | Makespan | $\max_j C_j$ |
| $\sum C_j$ | Total completion time | — |
| $\sum w_j C_j$ | Weighted completion time | — |
| $L_{\max}$ | Maximum lateness | $\max_j (C_j - d_j)$ |
| $\sum T_j$ | Total tardiness | $\sum \max(0, C_j - d_j)$ |
| $\sum w_j T_j$ | Weighted tardiness | — |
| $\sum U_j$ | Number of tardy jobs | $U_j = 1$ if $C_j > d_j$ |

---

## Problem Hierarchy & Complexity

```
                    Single Machine (1)
                    /       |        \
            P_m (parallel)  Q_m      R_m
              |
         Flow Shop (F_m)
              |
         Job Shop (J_m)  ←── most general deterministic
              |
      Flexible Job Shop (FJ_m)
              |
         RCPSP  ←── adds resource constraints + precedence
```

Each level **generalizes** the one above. A flow shop is a special case of a job shop (all jobs share the same route). Understanding this hierarchy helps: an algorithm for $J_m$ works for $F_m$, but a polynomial algorithm for $1$ doesn't extend to $P_m$.

---

## Problems in This Repository

| Problem | Notation | Complexity | Directory |
|---------|----------|------------|-----------|
| [Single Machine](single_machine/) | $1 \mid\mid \gamma$ | Varies (P to NP-hard) | `single_machine/` |
| [Parallel Machine](parallel_machine/) | $P_m \mid\mid C_{\max}$ | NP-hard ($m \geq 2$) | `parallel_machine/` |
| [Flow Shop](flow_shop/) | $F_m \mid\mid C_{\max}$ | NP-hard ($m \geq 3$) | `flow_shop/` |
| [Job Shop](job_shop/) | $J_m \mid\mid C_{\max}$ | NP-hard ($m \geq 2$) | `job_shop/` |
| [Flexible Job Shop](flexible_job_shop/) | $FJ_m \mid\mid C_{\max}$ | NP-hard | `flexible_job_shop/` |
| [RCPSP](rcpsp/) | — | NP-hard (strongly) | `rcpsp/` |

---

## Common Solution Methods Across Scheduling

### Exact Methods
- Branch and Bound
- Mixed-Integer Programming (MIP)
- Constraint Programming (CP)
- Column Generation
- Dynamic Programming (for special cases)

### Constructive Heuristics
- Dispatching Rules (SPT, LPT, EDD, WSPT, ATC)
- NEH (flow shop specific)
- Shifting Bottleneck (job shop specific)

### Metaheuristics
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Tabu Search (TS)
- Iterated Greedy (IG)
- Variable Neighborhood Search (VNS)
- Ant Colony Optimization (ACO)

### Hybrid & Modern
- Matheuristics (MIP + metaheuristic)
- Hyper-heuristics
- Reinforcement Learning for scheduling
- Neural Combinatorial Optimization

---

## Key References

> Links to foundational texts and survey papers on scheduling theory.

- Graham, R.L. et al. (1979). "Optimization and Approximation in Deterministic Sequencing and Scheduling: A Survey" — [DOI](https://doi.org/10.1016/S0167-5060(08)70356-X)
- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.) — [Springer](https://link.springer.com/book/10.1007/978-3-319-26580-3)
- Brucker, P. (2007). *Scheduling Algorithms* (5th ed.) — [Springer](https://link.springer.com/book/10.1007/978-3-540-69516-5)
- Blazewicz, J. et al. (2007). *Handbook on Scheduling* — [Springer](https://link.springer.com/book/10.1007/978-3-540-32220-7)

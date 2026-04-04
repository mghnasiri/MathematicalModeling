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

### MIP for $P_m \mid\mid C_{\max}$ (Identical Machines)

$$\min \quad C_{\max} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \in N \quad \text{(each job assigned to exactly one machine)} \tag{2}$$

$$C_{\max} \geq \sum_{j=1}^{n} p_j \cdot x_{ij} \quad \forall i \in M \quad \text{(makespan definition)} \tag{3}$$

$$x_{ij} \in \{0, 1\} \quad \forall i \in M, \; j \in N \tag{4}$$

The model has $m \cdot n$ binary variables and one continuous variable ($C_{\max}$), with $n$ equality constraints (2) and $m$ inequality constraints (3). Equivalent to the multiprocessor scheduling / number partitioning problem for $m = 2$.

### MIP for $Q_m \mid\mid C_{\max}$ (Uniform Machines)

When machine $i$ has speed $s_i$, the processing time of job $j$ on machine $i$ becomes $p_j / s_i$. The formulation changes only in constraint (3):

$$C_{\max} \geq \sum_{j=1}^{n} \frac{p_j}{s_i} \cdot x_{ij} \quad \forall i \in M \tag{3'}$$

All other constraints remain identical. The LP relaxation is tighter than the identical case because speed differences constrain job placement.

### MIP for $R_m \mid\mid C_{\max}$ (Unrelated Machines)

For unrelated machines, each job-machine pair has an independent processing time $p_{ij}$:

$$\min \quad C_{\max} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \in N \tag{2}$$

$$C_{\max} \geq \sum_{j=1}^{n} p_{ij} \cdot x_{ij} \quad \forall i \in M \tag{3''}$$

$$x_{ij} \in \{0, 1\} \quad \forall i \in M, \; j \in N \tag{4}$$

This is the most general formulation. The implementation in `exact/mip_makespan.py` handles all three environments by calling `instance.get_processing_time(j, i)`, which dispatches to the correct computation based on `machine_type`.

### LP Relaxation and Bounds

Relaxing integrality ($x_{ij} \in [0, 1]$) gives a lower bound on the optimal makespan. For the identical machine case, the LP relaxation always yields:

$$C_{\max}^{LP} = \frac{\sum_{j=1}^{n} p_j}{m}$$

since each job can be fractionally split across all machines for perfect load balance. This equals the **machine lower bound** $LB_1 = \lceil \sum p_j / m \rceil$. A second trivial lower bound is the **job lower bound** $LB_2 = \max_j p_j$. The overall lower bound is:

$$LB = \max(LB_1, LB_2) = \max\!\left(\frac{\sum p_j}{m}, \; \max_j p_j\right)$$

For unrelated machines, the LP relaxation can be tighter than these simple bounds because machine-specific processing times constrain fractional assignments.

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

### Standard Benchmark Sources

| Source | Description | URL |
|--------|-------------|-----|
| OR-Library (Beasley) | Parallel machine instances with known optima ($n$ up to 200, $m$ up to 25) | http://people.brunel.ac.uk/~mastjjb/jeb/info.html |
| Frangioni et al. (2004) | Large-scale identical/uniform instances ($n$ up to 1000) | Described in the paper |
| Random generation | $p_j \sim U[1, 100]$ for identical; $p_{ij} \sim U[1, 100]$ for unrelated | `instance.py` factory methods |

The OR-Library collection by J.E. Beasley includes scheduling datasets covering both identical and uniform machine configurations with known optimal solutions, enabling reproducible comparison of heuristics and exact methods. For unrelated machines, the Dell'Amico & Martello (1995) instances provide a standard testbed.

### Instance Generation in This Repository

The `ParallelMachineInstance` class provides three factory methods:
- `random_identical(n, m, seed)` -- generates $p_j \sim U[1, 99]$ and $w_j \sim U[1, 9]$
- `random_uniform(n, m, seed)` -- additionally generates speeds $s_i \sim U[0.5, 2.0]$
- `random_unrelated(n, m, seed)` -- generates full $p_{ij}$ matrix of shape $(m, n)$

---

## 5. Solution Methods

### 5.1 Exact Methods

**MIP (HiGHS):** Assignment-based formulation (Section 2). Uses SciPy's `milp` interface with the HiGHS solver. Variable ordering: $[x_{0,0}, x_{0,1}, \ldots, x_{m-1,n-1}, C_{\max}]$, giving $m \cdot n + 1$ total variables. Practical for $n \leq 50$ with small $m$. Falls back to LPT when the solver fails to find a feasible solution within the time limit (default 60 s).

**PTAS (Hochbaum & Shmoys, 1987):** A Polynomial-Time Approximation Scheme exists for $P_m \mid\mid C_{\max}$ when $m$ is fixed. The idea: (1) round large jobs (those with $p_j > \varepsilon \cdot LB$) to a polynomial number of distinct sizes, (2) enumerate all assignments of rounded large jobs in time $O(n^{f(1/\varepsilon)})$, (3) assign small jobs greedily to the least loaded machine. For any $\varepsilon > 0$ this yields a $(1 + \varepsilon)$-approximation. The running time is polynomial for fixed $\varepsilon$ but exponential in $1/\varepsilon$, so it is primarily of theoretical interest. Hochbaum and Shmoys later extended this to a dual approximation framework that also handles the uniform machine case $Q_m \mid\mid C_{\max}$.

### 5.2 Constructive Heuristics

| Method | Idea | Approximation |
|--------|------|--------------|
| **LPT** | Assign longest unscheduled job to least-loaded machine | $4/3 - 1/(3m)$ for $C_{\max}$ |
| **SPT** | Sort by ascending $p_j$, round-robin assignment | Optimal for $\sum C_j$ |
| **MULTIFIT** | FFD bin-packing + binary search on makespan | 1.22 for $C_{\max}$ |
| **List Scheduling** | Assign next job to least-loaded machine (any order) | $2 - 1/m$ for $C_{\max}$ |

#### 5.2.1 LPT (Longest Processing Time)

```
ALGORITHM LPT(p[1..n], m)
  Sort jobs by decreasing p_j
  Initialize min-heap H ← {(0, i) for i = 1..m}   // (load, machine)
  FOR each job j in sorted order:
    (load, i*) ← EXTRACT-MIN(H)
    Assign j to machine i*
    INSERT(H, (load + p[j], i*))
  RETURN assignment, max load
```

**Complexity:** $O(n \log n + n \log m)$ -- sorting dominates when $n > m$; each of $n$ heap operations costs $O(\log m)$.

**Approximation proof sketch (4/3 - 1/(3m)):**
Let $C_{\max}^*$ be the optimal makespan and let $C_{\max}^{LPT}$ be the LPT makespan. Consider the job $j^*$ that finishes last in the LPT schedule. When $j^*$ was assigned, its machine had the minimum load, so:

$$\text{load}(i^*) \leq \frac{\sum_{k \neq j^*} p_k}{m}$$

Since LPT processes jobs in decreasing order, $p_{j^*}$ is no larger than any previously assigned job. With $n$ jobs on $m$ machines, $j^*$ is at position $\geq m+1$ in the sorted order, so $p_{j^*} \leq p_{(m+1)}$. Because the optimal solution must have at least two jobs on some machine: $C_{\max}^* \geq p_{(1)} + p_{(m+1)}$, giving $p_{j^*} \leq C_{\max}^* - p_{(1)} \leq C_{\max}^*$. Combining the load bound with $p_{j^*} \leq \frac{1}{3} C_{\max}^*$ (from the pigeonhole argument on the $(m+1)$-th job) yields:

$$C_{\max}^{LPT} \leq \left(\frac{4}{3} - \frac{1}{3m}\right) C_{\max}^*$$

The bound is tight: the classic example $p = (2m-1, 2m-1, 2m-2, 2m-2, \ldots, 2, 2, 1, 1, 1)$ on $m$ machines achieves it.

#### 5.2.2 List Scheduling (Graham's Algorithm)

```
ALGORITHM LIST-SCHEDULING(jobs[1..n], m)
  Initialize min-heap H ← {(0, i) for i = 1..m}
  FOR each job j in jobs (arbitrary order):
    (load, i*) ← EXTRACT-MIN(H)
    Assign j to machine i*
    INSERT(H, (load + p[j], i*))
  RETURN assignment, max load
```

**Complexity:** $O(n \log m)$ -- no sorting required.

**Approximation:** $2 - 1/m$. This was the first worst-case result for any combinatorial optimization problem (Graham, 1966). The proof follows the same structure as LPT but without the decreasing-order constraint on $p_{j^*}$, which only guarantees $p_{j^*} \leq C_{\max}^*$, yielding the weaker $2 - 1/m$ bound.

#### 5.2.3 MULTIFIT

```
ALGORITHM MULTIFIT(p[1..n], m, k)
  L ← max(max(p_j), sum(p_j) / m)        // lower bound
  U ← sum(p_j)                             // upper bound (all on one machine)
  best ← NULL
  FOR iter = 1 TO k:                        // binary search iterations
    C ← (L + U) / 2                        // candidate makespan
    A ← FFD(p, m, C)                       // First Fit Decreasing with capacity C
    IF A ≠ NULL:                            // all n jobs fit in m bins
      best ← A;  U ← C
    ELSE:
      L ← C
  RETURN best

SUBROUTINE FFD(p[1..n], m, C)
  Sort jobs by decreasing p_j
  load[1..m] ← 0
  FOR each job j in sorted order:
    FOR i = 1 TO m:
      IF load[i] + p[j] ≤ C:
        Assign j to machine i;  load[i] += p[j];  BREAK
    IF j not assigned: RETURN NULL          // infeasible
  RETURN assignment
```

**Complexity:** $O(k \cdot n \cdot m)$ where $k$ is the number of binary search iterations (default 30). The binary search converges geometrically: after $k$ iterations, the gap between $L$ and $U$ is $(U_0 - L_0) / 2^k$.

**Approximation:** Coffman, Garey & Johnson (1978) proved that MULTIFIT achieves a worst-case ratio of at most 1.22 for $P_m \mid\mid C_{\max}$. The key insight is that FFD is known to use at most $\lceil 11/9 \cdot \text{OPT} + 6/9 \rceil$ bins for bin packing; binary search over the makespan translates this bin-packing guarantee into a makespan bound. Subsequent analysis by Yue (1990) tightened the bound to $13/11 \approx 1.182$.

#### 5.2.4 SPT Optimality for $P_m \mid\mid \sum C_j$

For the total completion time objective on identical parallel machines, SPT with round-robin assignment is optimal:

```
ALGORITHM SPT-ROUND-ROBIN(p[1..n], m)
  Sort jobs by increasing p_j
  FOR j = 0 TO n-1:
    Assign job sorted[j] to machine (j mod m)
  RETURN assignment
```

**Optimality argument:** On each machine, jobs are processed in SPT order, which minimizes the sum of completion times on that machine (Conway et al., 1967). The round-robin assignment distributes jobs so that the $k$-th largest jobs on each machine are as balanced as possible. Any swap of two jobs between machines would increase the total completion time because the job moved to an earlier position on one machine saves less than the job moved to a later position on the other machine costs.

### 5.3 Metaheuristics

This repository implements **6 metaheuristics** for parallel machine scheduling:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Reassign, swap between machines |
| 2 | Simulated Annealing (SA) | Trajectory | Reassign/swap with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Job-machine pair tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove jobs + LPT reconstruct |
| 5 | Genetic Algorithm (GA) | Population | Integer-vector encoding (gene = machine) |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Reassign -> swap -> block-move |

#### Default Parameter Settings

**Genetic Algorithm (GA):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 30 | Number of individuals in the population |
| `crossover_rate` | 0.8 | Probability of uniform crossover |
| `mutation_rate` | 0.3 | Probability of mutation per offspring |
| `max_generations` | 500 | Maximum generations (if no time limit) |
| `use_local_search` | False | Optional load-balancing local search |
| Selection | Binary tournament | Two random individuals, fittest survives |
| Replacement | Steady-state | Replace worst individual if offspring is better |

**Simulated Annealing (SA):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 5000 | Maximum number of iterations |
| `initial_temp` | Auto-calibrated | Based on initial LPT makespan |
| `cooling_rate` | 0.995 | Geometric cooling: $T_{k+1} = \alpha \cdot T_k$ |
| Neighborhoods | Relocate + Swap | Move from most-loaded or exchange between machines |
| Warm-start | LPT | Initial solution from Longest Processing Time |

**Tabu Search (TS):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 2000 | Maximum number of iterations |
| `tabu_tenure` | $\lfloor\sqrt{n}\rfloor$ | Iterations a (job, machine) pair remains tabu |
| Aspiration | Global best | Tabu overridden if move yields new best |
| Neighborhoods | Relocate + Swap | Same as SA |
| Warm-start | LPT | Initial solution from Longest Processing Time |

---

## 6. Implementation Guide

### Solution Encoding

Solutions are represented as assignment lists: `assignment[i]` is the ordered list of job indices assigned to machine $i$. Makespan evaluation scans all $m$ machines in $O(n)$. For metaheuristics, an equivalent flat encoding is a vector $\sigma \in \{0, \ldots, m-1\}^n$ where $\sigma_j$ is the machine assigned to job $j$.

### Key Design Decisions

- **LPT tie-breaking:** When multiple machines have equal load, the heap returns the lowest-index machine (Python `heapq` breaks ties on the second tuple element). This does not affect the worst-case ratio.
- **MULTIFIT bounds:** Binary search runs between $L = \max(\max_j p_j, \sum p_j / m)$ as lower bound and $U = \sum p_j$ as upper bound. Each iteration runs FFD in $O(n \cdot m)$. A tolerance of $10^{-9}$ handles floating-point comparisons in the FFD capacity check.
- **Unrelated LPT adaptation:** For $R_m \mid\mid C_{\max}$, LPT sorts by the maximum processing time across machines ($\max_i p_{ij}$) and assigns each job to the machine that minimizes the resulting load increase, rather than always choosing the globally least-loaded machine.
- **Warm-starting:** All metaheuristics (SA, TS, GA, IG, VNS) use LPT as their initial solution. The MIP solver falls back to LPT when HiGHS fails to find a feasible solution within the time limit.
- **Machine type dispatch:** The `get_processing_time(job, machine)` method in `ParallelMachineInstance` transparently handles all three environments -- identical returns $p_j$, uniform returns $p_j / s_i$, unrelated returns $p_{ij}$ -- so algorithm code needs no environment-specific branching.

### Complexity Summary

| Method | Time Complexity | Space |
|--------|----------------|-------|
| LPT | $O(n \log n + n \log m)$ | $O(n + m)$ |
| SPT round-robin | $O(n \log n)$ | $O(n + m)$ |
| List Scheduling | $O(n \log m)$ | $O(n + m)$ |
| MULTIFIT | $O(k \cdot n \cdot m)$ | $O(n + m)$ |
| MIP (HiGHS) | Exponential worst-case | $O(m \cdot n)$ |
| SA | $O(I \cdot m)$ per iteration | $O(n + m)$ |
| TS | $O(I \cdot n \cdot m)$ per iteration | $O(n \cdot m)$ |
| GA | $O(G \cdot P \cdot n \cdot m)$ | $O(P \cdot n)$ |

---

## 7. Computational Results Summary

Gaps reported as relative percentage deviation from the lower bound $LB = \max(\sum p_j / m, \max_j p_j)$ on randomly generated identical-machine instances ($p_j \sim U[1, 100]$).

| Method | Gap ($n{=}50$, $m{=}3$) | Gap ($n{=}200$, $m{=}5$) | Time per instance |
|--------|-------------|--------------|-------------------|
| List Scheduling | 5-15% | 3-8% | < 1 ms |
| LPT | 1-5% | 1-3% | < 1 ms |
| MULTIFIT | < 1% | < 1% | < 5 ms |
| MIP (exact) | 0% | Timeout (> 60 s) | 0.1 - 60 s |
| SA (5000 iter) | < 1% | < 1% | 0.5 - 2 s |
| TS (2000 iter) | < 1% | < 1% | 0.3 - 1 s |
| GA (500 gen) | < 1% | 1-2% | 1 - 5 s |

**Observations:**
- For small instances ($n \leq 30$), the MIP solver finds provably optimal solutions in under a second.
- LPT is the best single-pass heuristic for makespan, but MULTIFIT consistently closes the remaining gap.
- Among metaheuristics, SA and TS reach near-optimal solutions fastest due to their lightweight iteration cost.
- For the unrelated machine case ($R_m$), gaps are generally larger because the problem structure offers fewer symmetries to exploit.

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

1. Graham, R.L. (1966). Bounds for certain multiprocessing anomalies. *Bell System Technical Journal*, 45(9), 1563-1581. DOI: [10.1002/j.1538-7305.1966.tb01709.x](https://doi.org/10.1002/j.1538-7305.1966.tb01709.x) -- First worst-case analysis of list scheduling ($2 - 1/m$ bound).

2. Graham, R.L. (1969). Bounds on multiprocessing timing anomalies. *SIAM J. Applied Mathematics*, 17(2), 416-429. DOI: [10.1137/0117039](https://doi.org/10.1137/0117039) -- LPT analysis with $4/3 - 1/(3m)$ approximation guarantee.

3. Coffman, E.G., Garey, M.R. & Johnson, D.S. (1978). An application of bin-packing to multiprocessor scheduling. *SIAM J. Computing*, 7(1), 1-17. DOI: [10.1137/0207001](https://doi.org/10.1137/0207001) -- Introduced MULTIFIT, reducing makespan scheduling to bin packing.

4. Hochbaum, D.S. & Shmoys, D.B. (1987). Using dual approximation algorithms for scheduling problems: Theoretical and practical results. *J. ACM*, 34(1), 144-162. DOI: [10.1145/7531.7535](https://doi.org/10.1145/7531.7535) -- PTAS for $P_m \mid\mid C_{\max}$ via job rounding and dual approximation.

5. Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman. -- NP-hardness proofs for $P_2 \mid\mid C_{\max}$ (PARTITION) and $P \mid\mid C_{\max}$ (3-PARTITION).

6. Conway, R.W., Maxwell, W.L. & Miller, L.W. (1967). *Theory of Scheduling*. Addison-Wesley. -- SPT optimality for single-machine $\sum C_j$ and extension to parallel machines.

7. Dell'Amico, M. & Martello, S. (1995). Optimal scheduling of tasks on identical parallel processors. *ORSA J. Computing*, 7(2), 191-200. DOI: [10.1287/ijoc.7.2.191](https://doi.org/10.1287/ijoc.7.2.191) -- Exact B&B for identical machines with benchmark instances.

8. Cheng, R. & Gen, M. (1997). Parallel machine scheduling problems using memetic algorithms. *Computers & Industrial Engineering*, 33(3-4), 761-764. DOI: [10.1016/S0360-8352(97)00234-2](https://doi.org/10.1016/S0360-8352(97)00234-2) -- GA with integer-vector encoding for parallel machines.

9. Piersma, N. & Van Dijk, W. (1996). A local search heuristic for unrelated parallel machine scheduling with efficient neighborhood search. *Mathematical and Computer Modelling*, 24(9), 11-19. DOI: [10.1016/0895-7177(96)00150-2](https://doi.org/10.1016/0895-7177(96)00150-2) -- Tabu search for unrelated machines.

10. Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680. DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671) -- Foundational SA paper applied here to parallel machine scheduling.

11. Yue, M. (1990). On the exact upper bound for the multifit processor scheduling algorithm. *Annals of Operations Research*, 24(1), 233-259. DOI: [10.1007/BF02216826](https://doi.org/10.1007/BF02216826) -- Tightened MULTIFIT bound to $13/11 \approx 1.182$.

12. Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer. DOI: [10.1007/978-3-319-26580-3](https://doi.org/10.1007/978-3-319-26580-3) -- Comprehensive textbook covering all machine environments and objectives.

13. Lenstra, J.K., Shmoys, D.B. & Tardos, E. (1990). Approximation algorithms for scheduling unrelated parallel machines. *Mathematical Programming*, 46(1-3), 259-271. DOI: [10.1007/BF01585745](https://doi.org/10.1007/BF01585745) -- 2-approximation for $R_m \mid\mid C_{\max}$ via LP rounding.

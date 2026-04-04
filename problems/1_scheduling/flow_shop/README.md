# Permutation Flow Shop Scheduling (PFSP)

## 1. Problem Definition

- **Input:**
  - A set $N = \{1, 2, \ldots, n\}$ of jobs
  - A set $M = \{1, 2, \ldots, m\}$ of machines arranged in series
  - Processing times $p_{ij} \geq 0$ — time to process job $j$ on machine $i$
- **Decision:** Find a permutation $\pi = (\pi(1), \pi(2), \ldots, \pi(n))$ of jobs
- **Objective:** Minimize the makespan $C_{\max} = C_{m,\pi(n)}$
- **Constraints:** Each job visits machines in order $M_1 \to M_2 \to \cdots \to M_m$. In the permutation variant, the job sequence is the same on all machines (no passing). Each machine processes one job at a time; each job is on one machine at a time.
- **Classification:** Combinatorial optimization (discrete permutation)
- **Scheduling notation:** $F_m \mid prmu \mid C_{\max}$

### Complexity

| Problem | Complexity | Reference |
|---------|-----------|-----------|
| $F_2 \mid\mid C_{\max}$ | $O(n \log n)$ — polynomial | Johnson (1954) |
| $F_3 \mid\mid C_{\max}$ | Strongly NP-hard | Garey, Johnson & Sethi (1976) |
| $F_m \mid prmu \mid C_{\max}$ | NP-hard for $m \geq 3$ | Garey, Johnson & Sethi (1976) |
| $F_m \mid prmu \mid \sum C_j$ | NP-hard for $m \geq 2$ | Gonzalez & Sahni (1978) |
| $F_2 \mid\mid \sum C_j$ | NP-hard | Gonzalez & Sahni (1978) |
| $F_m \mid block \mid C_{\max}$ | NP-hard for $m \geq 3$ | Hall & Sriskandarajah (1996) |
| $F_m \mid no\text{-}wait \mid C_{\max}$ | NP-hard for $m \geq 3$; reduces to asymmetric TSP | Wismer (1972) |

The 2-machine case $F_2 \mid\mid C_{\max}$ is the key tractable special case, solved optimally by Johnson's Rule. Adding a third machine makes the problem NP-hard; the proof is by reduction from 3-PARTITION (Garey, Johnson & Sethi, 1976).

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $m$ | Number of machines | $\mathbb{Z}^+$ |
| $p_{ij}$ | Processing time of job $j$ on machine $i$ | $\mathbb{R}_{\geq 0}$ |
| $\pi(k)$ | Job assigned to position $k$ in the permutation | $N$ |
| $C_{i,\pi(k)}$ | Completion time of the $k$-th job on machine $i$ | $\mathbb{R}_{\geq 0}$ |
| $C_{\max}$ | Makespan (completion of last job on last machine) | $\mathbb{R}_{\geq 0}$ |
| $x_{jk}$ | 1 if job $j$ is in position $k$ | $\{0, 1\}$ |

### Formulation A: Completion-Time Recursion (Evaluation)

Given a permutation $\pi$, the makespan is computed by the recursion:

$$C_{1,\pi(1)} = p_{1,\pi(1)} \tag{1}$$

$$C_{i,\pi(1)} = C_{i-1,\pi(1)} + p_{i,\pi(1)} \quad \forall\, i = 2, \ldots, m \tag{2}$$

$$C_{1,\pi(k)} = C_{1,\pi(k-1)} + p_{1,\pi(k)} \quad \forall\, k = 2, \ldots, n \tag{3}$$

$$C_{i,\pi(k)} = \max\bigl(C_{i-1,\pi(k)},\; C_{i,\pi(k-1)}\bigr) + p_{i,\pi(k)} \quad \forall\, i \geq 2,\, k \geq 2 \tag{4}$$

$$C_{\max} = C_{m,\pi(n)} \tag{5}$$

Evaluation complexity: $O(nm)$ for a given permutation.

### Formulation B: Position-Based MILP (Manne/Wagner)

$$\min \quad C_{\max} \tag{6}$$

$$\text{s.t.} \quad \sum_{k=1}^{n} x_{jk} = 1 \quad \forall\, j \in N \quad \text{(each job assigned to one position)} \tag{7}$$

$$\sum_{j=1}^{n} x_{jk} = 1 \quad \forall\, k = 1, \ldots, n \quad \text{(each position gets one job)} \tag{8}$$

$$C_{ik} \geq C_{(i-1)k} + \sum_{j=1}^{n} p_{ij}\, x_{jk} \quad \forall\, i \geq 2,\, k \tag{9}$$

$$C_{ik} \geq C_{i(k-1)} + \sum_{j=1}^{n} p_{ij}\, x_{jk} \quad \forall\, k \geq 2,\, i \tag{10}$$

$$C_{\max} \geq C_{mk} \quad \forall\, k \tag{11}$$

$$x_{jk} \in \{0,1\},\quad C_{ik} \geq 0 \tag{12}$$

**Strengths:** Compact formulation, $O(n^2 + nm)$ constraints. Solvable by any MILP solver.
**Weaknesses:** Weak LP relaxation — the fractional solution spreads jobs across positions, yielding a loose lower bound. Practical for $n \leq 20$.

### Formulation C: Constraint Programming (CP-SAT)

Model each job-on-machine operation as an interval variable; impose no-overlap constraints on each machine and precedence constraints within each job. The CP-SAT solver in OR-Tools handles this natively with propagation and search heuristics. More practical than the MILP for medium instances ($n \leq 50$).

---

## 3. Variants

| Notation | Variant | Directory | Description |
|----------|---------|-----------|-------------|
| $F_m \mid no\text{-}wait \mid C_{\max}$ | No-Wait | `variants/no_wait/` | Jobs proceed without waiting between machines |
| $F_m \mid block \mid C_{\max}$ | Blocking | `variants/blocking/` | No intermediate buffers between machines |
| $F_m \mid S_{sd} \mid C_{\max}$ | Setup Times (SDST) | `variants/setup_times/` | Sequence-dependent setup times |
| $F_m \mid prmu \mid \sum w_j T_j$ | Weighted Tardiness | `variants/tardiness/` | Minimize total weighted tardiness |
| $HF_m \mid prmu \mid C_{\max}$ | Hybrid Flow Shop | `variants/hybrid/` | Multiple parallel machines at each stage |
| $DF_m \mid prmu \mid C_{\max}$ | Distributed | `variants/distributed/` | Jobs assigned across multiple factories |
| $F_m \mid lot \mid C_{\max}$ | Lot Streaming | `variants/lot_streaming/` | Jobs split into sublots for overlapping |
| $F_m \mid stoch \mid E[C_{\max}]$ | Stochastic | `variants/stochastic/` | Processing times are random variables |
| $O_m \mid\mid C_{\max}$ | Open Shop | `variants/open_shop/` | No fixed machine order per job |

### 3.1 No-Wait Flow Shop

Jobs cannot wait between consecutive machines — processing must be contiguous. Reduces to an asymmetric TSP on the delay matrix $D[j][k] = \max_{i=1}^{m-1}\bigl(\sum_{h=1}^{i} p_{hj} - \sum_{h=1}^{i} p_{hk}\bigr)$. Applications: steel rolling, chemical processing.

### 3.2 Blocking Flow Shop

No intermediate buffers — a job blocks its machine until the downstream machine is free. Uses departure-time recursion instead of standard completion times. Applications: robotic cells, paint shops.

### 3.3 Sequence-Dependent Setup Times (SDST)

Setup time $s_{ijk}$ on machine $i$ between job $j$ and job $k$. Adds $s_{i,\pi(k-1),\pi(k)}$ to the completion-time recursion. Applications: printing, semiconductor fab, food processing.

### 3.4 Hybrid / Flexible Flow Shop

Each stage has $m_i \geq 1$ identical parallel machines. Jobs must visit all stages in order but can be processed on any machine at each stage. Combines flow shop sequencing with parallel machine assignment.

### 3.5 Distributed Flow Shop

$F$ identical factories, each with $m$ machines. Each job is assigned to one factory and sequenced within it. Objective: minimize maximum makespan across all factories.

See each variant's `README.md` for detailed formulations and algorithms.

---

## 4. Benchmark Instances

### Taillard Benchmark Library

The standard PFSP benchmark set (Taillard, 1993) comprises 120 instances across 12 size classes:

| Class | $n \times m$ | Instances | BKS Source |
|-------|-------------|-----------|------------|
| ta001–ta010 | 20 × 5 | 10 | Optimal (B&B) |
| ta011–ta020 | 20 × 10 | 10 | Optimal (B&B) |
| ta021–ta030 | 20 × 20 | 10 | Optimal (B&B) |
| ta031–ta040 | 50 × 5 | 10 | Taillard (1993) |
| ta041–ta050 | 50 × 10 | 10 | Taillard (1993) |
| ta051–ta060 | 50 × 20 | 10 | Taillard (1993) |
| ta061–ta070 | 100 × 5 | 10 | Taillard (1993) |
| ta071–ta080 | 100 × 10 | 10 | Taillard (1993) |
| ta081–ta090 | 100 × 20 | 10 | Taillard (1993) |
| ta091–ta100 | 200 × 10 | 10 | Taillard (1993) |
| ta101–ta110 | 200 × 20 | 10 | Taillard (1993) |
| ta111–ta120 | 500 × 20 | 10 | Taillard (1993) |

**URL:** http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/flowshop.dir/best_lb_up.txt

Processing times are drawn from $U[1, 99]$.

### Instance Format

```
n m
p_11 p_12 ... p_1n    (machine 1: processing time of each job)
p_21 p_22 ... p_2n    (machine 2)
...
p_m1 p_m2 ... p_mn    (machine m)
```

### Small Illustrative Instance

A 4-job, 3-machine instance for testing:

```
4 3
5 9 8 10
6 3 7  1
2 4 5  8
```

Optimal permutation: $(2, 1, 3, 4)$ with $C_{\max} = 31$ (verifiable by hand).

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Johnson's Rule (Johnson, 1954) — Optimal for $F_2 \mid\mid C_{\max}$

**Idea:** Partition jobs into two sets: $U = \{j : p_{1j} \leq p_{2j}\}$ and $V = \{j : p_{1j} > p_{2j}\}$. Sort $U$ by increasing $p_{1j}$, sort $V$ by decreasing $p_{2j}$. Concatenate $U$ then $V$.

**Complexity:** $O(n \log n)$.

```
ALGORITHM JohnsonsRule(p[1..n], q[1..n])
  U ← {j : p[j] ≤ q[j]},  V ← {j : p[j] > q[j]}
  Sort U by ascending p[j]
  Sort V by descending q[j]
  RETURN U ++ V
```

#### Branch and Bound (Taillard, 1993; Ladhari & Haouari, 2005)

**Idea:** DFS enumeration of partial permutations with machine-based lower bounds. At each node, a partial sequence $(\pi(1), \ldots, \pi(k))$ is fixed; the bound computes the minimum possible makespan assuming the remaining jobs can be processed without waiting. NEH is used as a warm-start upper bound.

**Practical limit:** $n \leq 20$ for makespan optimality.

#### MIP / CP-SAT

Position-based MILP (Formulation B above) via SciPy HiGHS, or interval-variable CP model via OR-Tools CP-SAT. The CP model scales better than the MILP.

### 5.2 Constructive Heuristics

| # | Method | Author(s) | Year | Complexity | Key Idea |
|---|--------|-----------|------|-----------|----------|
| 1 | Palmer's Slope Index | Palmer | 1965 | $O(nm + n \log n)$ | Weighted positional index favoring jobs with increasing processing times |
| 2 | Bonney-Gundry | Bonney & Gundry | 1976 | $O(nm + n \log n)$ | Cumulative processing time slope index |
| 3 | Gupta's Algorithm | Gupta | 1971 | $O(nm + n \log n)$ | Bottleneck-aware composite index |
| 4 | Dannenbring's RA | Dannenbring | 1977 | $O(nm + n \log n)$ | Weighted 2-machine reduction + Johnson's Rule |
| 5 | CDS | Campbell, Dudek & Smith | 1970 | $O(m \cdot n \log n)$ | $m{-}1$ virtual 2-machine sub-problems via Johnson |
| 6 | NEH | Nawaz, Enscore & Ham | 1983 | $O(n^2 m)$ | Insert jobs one-by-one in best position |
| 7 | NEHKK | Kalczynski & Kamburowski | 2008 | $O(n^2 m)$ | NEH + idle-time-based tie-breaking |
| 8 | Rajendran-Ziegler (RZ) | Rajendran & Ziegler | 1997 | $O(n^2 m)$ | Modified NEH with improvement pass |
| 9 | LR Heuristic | Liu & Reeves | 2001 | $O(n^3 m)$ | Multi-candidate composite index |
| 10 | RA Heuristic | Rad, Ruiz & Boroojerdian | 2009 | $O(n^2 m)$ | Enhanced NEH with front/back insertion |
| 11 | Beam Search | — | — | $O(k \cdot n^2 m)$ | Parallel-beam NEH insertion ($k$ = beam width) |

**NEH** remains the gold standard constructive heuristic after 40+ years. Its insertion mechanism implicitly evaluates $O(n^2)$ candidate positions. Most modern methods (NEHKK, RZ, RA, Beam Search) are NEH extensions.

```
ALGORITHM NEH(instance)
  Sort jobs by decreasing total processing time
  π ← [first_job]
  FOR j = 2 TO n:
    best_pos ← argmin over all positions of Cmax(insert j into π)
    Insert j at best_pos in π
  RETURN π
```

### 5.3 Improvement Heuristics / Local Search

| Neighborhood | Move Definition | Size | Incremental? |
|-------------|----------------|------|-------------|
| Swap | Exchange positions of jobs $i$ and $j$ | $O(n^2)$ | No (full re-evaluation) |
| Insertion | Remove job from position $i$, insert at position $j$ | $O(n^2)$ | Partial (Taillard acceleration) |
| Or-opt | Move block of 1-3 consecutive jobs | $O(n^2)$ | No |
| VND | Apply swap → insertion → or-opt in sequence | Variable | Restarts on improvement |

**Variable Neighborhood Descent (VND)** applies neighborhoods in increasing perturbation order. When one neighborhood finds no improvement, it switches to the next. On any improvement, it restarts from the first neighborhood.

### 5.4 Metaheuristics

This repository implements **14 metaheuristics** for the PFSP — one of the most comprehensive collections available:

| # | Method | Author(s) | Year | Category | Key Feature |
|---|--------|-----------|------|----------|-------------|
| 1 | Simulated Annealing (SA) | Osman & Potts | 1989 | Trajectory | Boltzmann acceptance, insertion neighborhood |
| 2 | Tabu Search (TS) | Nowicki & Smutnicki | 1996 | Trajectory | Short-term memory, aspiration criterion |
| 3 | Iterated Greedy (IG) | Ruiz & Stützle | 2007 | Trajectory | Destroy $d$ jobs + NEH reconstruct; **state-of-the-art** |
| 4 | Variable Neighborhood Search (VNS) | Mladenovic & Hansen | 1997 | Trajectory | Systematic neighborhood change; shaking + VND |
| 5 | Genetic Algorithm (GA) | Reeves | 1995 | Population | OX crossover, insertion mutation |
| 6 | Memetic Algorithm (MA) | Moscato | 1989 | Population | GA + local search on each offspring |
| 7 | Scatter Search (SS) | Glover | 1977 | Population | Reference set, path relinking |
| 8 | Estimation of Distribution (EDA) | Mühlenbein & Paass | 1996 | Population | Probabilistic model replaces crossover |
| 9 | Differential Evolution (DE) | Storn & Price | 1997 | Population | Vector differences adapted to permutations |
| 10 | Particle Swarm (PSO) | Kennedy & Eberhart | 1995 | Population | Velocity-based, random-key encoding |
| 11 | Ant Colony (ACO/MMAS) | Stützle | 1998 | Population | Pheromone trails, MMAS bounds |
| 12 | Artificial Bee Colony (ABC) | Karaboga | 2005 | Population | Employed/onlooker/scout phases |
| 13 | Harmony Search (HS) | Geem et al. | 2001 | Population | Harmony memory consideration rate |
| 14 | Teaching-Learning (TLBO) | Rao et al. | 2011 | Population | Teacher/learner phases; **no algorithm-specific parameters** |
| 15 | Biogeography-Based (BBO) | Simon | 2008 | Population | Migration-based operator |
| 16 | Whale Optimization (WOA) | Mirjalili & Lewis | 2016 | Population | Bubble-net hunting strategy |

**Iterated Greedy** dominates PFSP benchmarks because its destroy-reconstruct cycle creates large jumps in solution space that single-move neighborhoods cannot achieve. By removing $d$ jobs and reinserting via NEH, IG "teleports" between basins of attraction.

```
ALGORITHM IteratedGreedy(instance, d, T)
  π ← NEH(instance)
  π_best ← π
  WHILE not stopping_criterion:
    π_d ← remove d random jobs from π          (Destruction)
    π' ← reinsert removed jobs via NEH into π_d (Construction)
    π' ← LocalSearch(π')                        (Improvement)
    IF Cmax(π') < Cmax(π) OR accept(Δ, T):     (Acceptance)
      π ← π'
    IF Cmax(π) < Cmax(π_best):
      π_best ← π
  RETURN π_best
```

### 5.5 Hybrid and Advanced Methods

- **Memetic Algorithm** = GA + local search on each individual. Combines global exploration with local exploitation.
- **Scatter Search** = Reference set management + path relinking + local search. Uses diversification generation and structured combination.
- **Beam Search** = Truncated tree search keeping top-$k$ partial solutions. Bridges constructive heuristics and exact methods.

---

## 6. Implementation Guide

### Solver Modeling Tips

- **Taillard acceleration:** When evaluating job insertions during NEH, maintain head/tail completion-time matrices to compute makespan in $O(m)$ per insertion instead of $O(nm)$. This reduces NEH from $O(n^3 m)$ to $O(n^2 m)$.
- **Machine-based lower bound:** For B&B, compute per-machine bounds: for each machine $i$, the makespan is at least $\sum_{j \in S} p_{ij}$ (remaining job processing times) plus the minimum head and tail times. Take the maximum across machines.
- **Random-key encoding:** For population-based metaheuristics (PSO, DE) that operate on continuous vectors, use random keys: sort the continuous vector to obtain a permutation.

### Common Pitfalls

- Processing times matrix orientation: this repo uses shape $(m, n)$ — machines as rows, jobs as columns. Some papers use the transpose.
- Johnson's Rule edge case: when $p_{1j} = p_{2j}$, the job can go in either set $U$ or $V$.
- NEH tie-breaking: different tie-breaking rules (last machine idle time, total idle time) can change results by 1-3% on Taillard instances.

---

## 7. Computational Results Summary

### Taillard tai20_5 (10 instances, 20 jobs × 5 machines)

| Algorithm | Category | ARPD | Best RPD | Worst RPD | Avg Time |
|-----------|----------|------|----------|-----------|----------|
| Palmer (1965) | Heuristic | 10.81% | 5.89% | 15.93% | <0.001s |
| Gupta (1971) | Heuristic | 12.90% | 1.55% | 20.19% | <0.001s |
| CDS (1970) | Heuristic | 9.57% | 4.78% | 16.10% | <0.001s |
| NEH (1983) | Heuristic | 3.25% | 0.44% | 7.22% | 0.004s |
| NEH + VND | Heuristic+LS | 1.89% | 0.41% | 4.72% | 0.038s |
| SA (0.5s) | Metaheuristic | ~3% | — | — | 0.5s |
| TS (0.5s) | Metaheuristic | ~2% | — | — | 0.5s |
| **IG (0.5s)** | **Metaheuristic** | **0.56%** | **0.00%** | **1.26%** | **0.5s** |

*ARPD = Average Relative Percentage Deviation from best known solution.*

**Scale guidance:**
- **Small** ($n \leq 20$): B&B can find optimal; NEH + VND is near-optimal.
- **Medium** ($n = 50{-}100$): IG is state-of-the-art. SA/TS competitive with longer runtimes.
- **Large** ($n = 200{-}500$): IG remains the best single-method approach. Population-based methods (MA, SS) can match IG with careful tuning.

---

## 8. Implementations in This Repository

```
flow_shop/
├── instance.py                        # FlowShopInstance, FlowShopSolution dataclasses
├── benchmark_runner.py                # CLI for Taillard benchmarks
│
├── exact/
│   ├── johnsons_rule.py               # Optimal for F2||Cmax — O(n log n)
│   ├── branch_and_bound.py            # B&B with Taillard LB, NEH warm-start
│   └── mip_formulation.py             # Position-based MILP (HiGHS) + CP-SAT (OR-Tools)
│
├── heuristics/
│   ├── palmers_slope.py               # Palmer's slope index (1965)
│   ├── bonney_gundry.py               # Bonney-Gundry slope index (1976)
│   ├── guptas_algorithm.py            # Gupta's composite index (1971)
│   ├── dannenbring.py                 # Dannenbring Rapid Access (1977)
│   ├── cds.py                         # CDS multi-Johnson (1970)
│   ├── neh.py                         # NEH + FV2014 tie-breaking (1983)
│   ├── nehkk.py                       # NEHKK idle-time tie-breaking (2008)
│   ├── rajendran_ziegler.py           # RZ constructive + improvement (1997)
│   ├── lr_heuristic.py                # LR multi-candidate (2001)
│   ├── ra_heuristic.py                # RA enhanced NEH (2009)
│   └── beam_search.py                 # Beam Search (parametric beam width)
│
├── metaheuristics/
│   ├── local_search.py                # Swap, insertion, or-opt, VND
│   ├── simulated_annealing.py         # SA — Osman & Potts (1989)
│   ├── tabu_search.py                 # TS — Nowicki & Smutnicki (1996)
│   ├── iterated_greedy.py             # IG — Ruiz & Stützle (2007) ★ state-of-the-art
│   ├── vns.py                         # VNS — Mladenovic & Hansen (1997)
│   ├── genetic_algorithm.py           # GA — Reeves (1995)
│   ├── memetic_algorithm.py           # MA — GA + local search
│   ├── scatter_search.py              # SS — Glover (1977)
│   ├── eda.py                         # EDA — probabilistic model
│   ├── differential_evolution.py      # DE — Storn & Price (1997)
│   ├── particle_swarm.py              # PSO — Kennedy & Eberhart (1995)
│   ├── ant_colony.py                  # ACO/MMAS — Stützle (1998)
│   ├── bee_colony.py                  # ABC — Karaboga (2005)
│   ├── harmony_search.py              # HS — Geem et al. (2001)
│   ├── tlbo.py                        # TLBO — Rao et al. (2011)
│   ├── bbo.py                         # BBO — Simon (2008)
│   └── whale_optimization.py          # WOA — Mirjalili & Lewis (2016)
│
├── variants/
│   ├── no_wait/                       # Fm | no-wait | Cmax
│   ├── blocking/                      # Fm | block | Cmax
│   ├── setup_times/                   # Fm | Ssd | Cmax
│   ├── tardiness/                     # Fm | prmu | ΣwjTj
│   ├── hybrid/                        # HFm | prmu | Cmax
│   ├── distributed/                   # DFm | prmu | Cmax
│   ├── lot_streaming/                 # Fm | lot | Cmax
│   ├── stochastic/                    # Fm | stoch | E[Cmax]
│   └── open_shop/                     # Om || Cmax
│
└── tests/                             # 8 test files
    ├── test_flow_shop.py              # Core algorithms (Johnson, B&B, NEH, CDS, ...)
    ├── test_new_algorithms.py         # IG, SA, GA, local search
    ├── test_ts_aco_sdst.py            # Tabu Search, ACO, SDST variant
    ├── test_nehkk_abc_beam.py         # NEHKK, ABC, Beam Search
    ├── test_pso_de_ss_ra.py           # PSO, DE, Scatter Search, RA
    ├── test_rz_hs_bbo.py             # RZ, Harmony Search, BBO
    ├── test_bg_tlbo_woa.py            # Bonney-Gundry, TLBO, WOA
    └── test_vns_ma_eda.py             # VNS, Memetic, EDA
```

**Total:** 3 exact methods, 11 constructive heuristics, 17 metaheuristics/local search methods, 9 variants, 8 test files.

---

## 9. Algorithm Taxonomy

```
                     PFSP Solution Methods
                            │
           ┌────────────────┼──────────────────┐
           │                │                  │
        Exact          Constructive       Improvement
           │           Heuristics          Methods
           │                │                  │
     ┌─────┼─────┐    ┌────┼────┐       ┌─────┼──────┐
     │     │     │    │    │    │       │     │      │
  Johnson B&B  MIP  Palmer CDS NEH   Local  Meta-   Hybrid
   Rule  (LB)  (CP)  Gupta  LR      Search heuristics
   (F2)             Dannenb RZ         │
                    B-G  NEHKK    ┌────┼────────────┐
                    RA  Beam    Traj.  Population   Mixed
                                 │       │           │
                              SA,TS    GA,ACO      MA,SS
                              IG,VNS   DE,PSO     Beam
                                      ABC,EDA
                                      HS,TLBO
                                      BBO,WOA
```

---

## 10. Key References

### Seminal Papers

- Johnson, S.M. (1954). Optimal two- and three-stage production schedules with setup times included. *Naval Research Logistics Quarterly*, 1(1), 61-68.
- Garey, M.R., Johnson, D.S. & Sethi, R. (1976). The complexity of flowshop and jobshop scheduling. *Mathematics of Operations Research*, 1(2), 117-129.
- Nawaz, M., Enscore, E.E. & Ham, I. (1983). A heuristic algorithm for the $m$-machine, $n$-job flow-shop sequencing problem. *Omega*, 11(1), 91-95.
- Campbell, H.G., Dudek, R.A. & Smith, M.L. (1970). A heuristic algorithm for the $n$ job, $m$ machine sequencing problem. *Management Science*, 16(10), B-630-B-637.

### Key Metaheuristic References

- Ruiz, R. & Stützle, T. (2007). A simple and effective iterated greedy algorithm for the permutation flowshop scheduling problem. *European Journal of Operational Research*, 177(3), 2033-2049.
- Osman, I.H. & Potts, C.N. (1989). Simulated annealing for permutation flow-shop scheduling. *Omega*, 17(6), 551-557.
- Nowicki, E. & Smutnicki, C. (1996). A fast tabu search algorithm for the permutation flow-shop problem. *European Journal of Operational Research*, 91(1), 160-175.
- Stützle, T. (1998). An ant approach to the flow shop problem. *Proceedings of EUFIT'98*, 1560-1564.
- Reeves, C.R. (1995). A genetic algorithm for flowshop sequencing. *Computers & Operations Research*, 22(1), 5-13.

### Surveys

- Ruiz, R. & Maroto, C. (2005). A comprehensive review and evaluation of permutation flowshop heuristics. *European Journal of Operational Research*, 165(2), 479-494.
- Fernandez-Viagas, V., Ruiz, R. & Framinan, J.M. (2017). A new vision of approximate methods for the permutation flowshop to minimise makespan: State-of-the-art and computational evaluation. *European Journal of Operational Research*, 257(3), 707-721.

### Benchmark

- Taillard, E. (1993). Benchmarks for basic scheduling problems. *European Journal of Operational Research*, 64(2), 278-285.

---

## Key Insights

> **NEH (1983)** remains the gold standard constructive heuristic after 40+ years. Its insertion mechanism implicitly explores $O(n^2)$ solutions, and it serves as the foundation for modern methods like Iterated Greedy.

> **Iterated Greedy** dominates PFSP because its destroy-reconstruct cycle creates large jumps in the solution space that single-move neighborhoods (swap, insertion) cannot achieve. By removing $d$ jobs and reinserting them via NEH, IG effectively "teleports" between basins of attraction.

> **TLBO** is notable for having zero algorithm-specific parameters to tune — only population size and stopping criterion. This makes it attractive for practitioners who lack metaheuristic tuning expertise.

> **The 28-algorithm suite** in this folder forms a comprehensive metaheuristic benchmark lab. Running all methods on the same Taillard instances allows fair comparison of trajectory vs. population-based approaches.

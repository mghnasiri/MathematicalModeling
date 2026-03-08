# Flow Shop Scheduling (FSP)

## Problem Definition

Given $n$ jobs and $m$ machines arranged in series, each job must be processed on all machines in the same order ($M_1 \rightarrow M_2 \rightarrow \ldots \rightarrow M_m$). Determine the job sequence on each machine to optimize the objective. In a **permutation flow shop** (PFSP), the job sequence is the same on all machines.

---

## Variants

| Notation | Name | Description |
|----------|------|-------------|
| $F_m \mid\mid C_{\max}$ | Flow shop makespan | General flow shop |
| $F_m \mid prmu \mid C_{\max}$ | Permutation flow shop | Same sequence on all machines |
| $F_m \mid block \mid C_{\max}$ | Blocking flow shop | No buffer between machines |
| $F_m \mid no\text{-}wait \mid C_{\max}$ | No-wait flow shop | Jobs cannot wait between machines |
| $HF_m$ | Hybrid/flexible flow shop | Multiple machines at each stage |
| $F_m \mid s_{jk} \mid C_{\max}$ | Setup times | Sequence-dependent setups |

---

## Mathematical Formulation

### Parameters
- $n$ — number of jobs, $m$ — number of machines
- $p_{ij}$ — processing time of job $j$ on machine $i$

### Decision Variables
- $C_{ij}$ — completion time of job $j$ on machine $i$
- $x_{jk} \in \{0,1\}$ — 1 if job $j$ is in position $k$ (permutation)

### Completion Time Recursion (Permutation Flow Shop)

Let $\pi$ be the permutation of jobs:

$$C_{1,\pi(1)} = p_{1,\pi(1)}$$

$$C_{i,\pi(1)} = C_{i-1,\pi(1)} + p_{i,\pi(1)} \quad \forall i \geq 2$$

$$C_{1,\pi(k)} = C_{1,\pi(k-1)} + p_{1,\pi(k)} \quad \forall k \geq 2$$

$$C_{i,\pi(k)} = \max(C_{i-1,\pi(k)},\ C_{i,\pi(k-1)}) + p_{i,\pi(k)} \quad \forall i \geq 2, k \geq 2$$

$$\min\ C_{\max} = C_{m,\pi(n)}$$

---

## Complexity Analysis

| Problem | Complexity | Notes |
|---------|-----------|-------|
| $F_2 \mid\mid C_{\max}$ | $O(n \log n)$ | Johnson's Rule (1954) |
| $F_3 \mid\mid C_{\max}$ | NP-hard | Garey, Johnson & Sethi (1976) |
| $F_m \mid prmu \mid C_{\max}$ | NP-hard ($m \geq 3$) | |
| $F_m \mid prmu \mid \sum C_j$ | NP-hard ($m \geq 2$) | |
| $F_2 \mid\mid \sum C_j$ | NP-hard | |
| $F_m \mid block \mid C_{\max}$ | NP-hard ($m \geq 3$) | |
| $F_m \mid no\text{-}wait \mid C_{\max}$ | NP-hard ($m \geq 3$) | Reduces to TSP |

---

## Solution Approaches

### Exact Methods
| Method | Best For | Notes |
|--------|----------|-------|
| Johnson's Rule | $F_2 \mid\mid C_{\max}$ | Optimal in $O(n \log n)$ |
| Branch & Bound | $F_m \mid prmu \mid C_{\max}$ up to $n \approx 20$ | Taillard bound: machine-based lower bound |
| MIP (Manne, Wagner) | General | Position-based or time-indexed |

### Constructive Heuristics
| Method | Description | Quality |
|--------|-------------|---------|
| NEH (Nawaz, Enscore, Ham, 1983) | Insert jobs by decreasing total processing time | Best constructive heuristic |
| CDS (Campbell, Dudek, Smith, 1970) | Generate $m-1$ Johnson sub-problems | Fast, moderate quality |
| Palmer's slope index | Weighted positional heuristic | Simple, low quality |
| Gupta's algorithm | Generalized Johnson for $m > 2$ | |

### Improvement Heuristics
| Method | Neighborhood | Notes |
|--------|-------------|-------|
| 2-opt (swap) | Pairwise interchange | Simple |
| Insert | Remove + reinsert | Better for PFSP |
| Or-opt | Block moves | |

### Metaheuristics
| Method | Key Reference | Notes |
|--------|---------------|-------|
| Iterated Greedy (IG) | Ruiz & Stützle (2007) | State-of-the-art for PFSP |
| Simulated Annealing | Osman & Potts (1989) | Classic |
| Genetic Algorithm | Reeves (1995) | Permutation encoding |
| Tabu Search | Nowicki & Smutnicki (1996) | |
| NEH + local search | Various | NEH as seed + improvement |

---

## Implementations in This Repo

```
flow_shop/
├── instance.py                    # Data structures + makespan evaluation
├── benchmark_runner.py            # CLI benchmarking tool for Taillard instances
├── exact/
│   ├── johnsons_rule.py           # Optimal for F2||Cmax — O(n log n)
│   ├── branch_and_bound.py        # B&B with machine-based lower bounds + NEH warm start
│   └── mip_formulation.py         # MIP (SciPy HiGHS) + CP-SAT (OR-Tools)
├── heuristics/
│   ├── palmers_slope.py           # Palmer's slope index (1965)
│   ├── guptas_algorithm.py        # Gupta's composite index (1971)
│   ├── cds.py                     # CDS multi-Johnson heuristic (1970)
│   ├── lr_heuristic.py            # LR multi-candidate constructive (2001)
│   └── neh.py                     # NEH + tie-breaking variant (1983/2014)
├── metaheuristics/
│   ├── local_search.py            # Swap, insertion, or-opt, VND
│   ├── simulated_annealing.py     # SA with adaptive cooling schedule
│   ├── tabu_search.py             # TS with aspiration criterion
│   └── iterated_greedy.py         # IG destroy/reconstruct (state-of-the-art)
└── tests/
    └── test_flow_shop.py          # 67 tests covering all algorithms
```

### Benchmark Results (Taillard tai20_5, 10 instances)

| Algorithm | ARPD | Best RPD | Worst RPD | Avg Time |
|-----------|------|----------|-----------|----------|
| Palmer (1965) | 10.81% | 5.89% | 15.93% | <0.001s |
| Gupta (1971) | 12.90% | 1.55% | 20.19% | <0.001s |
| CDS (1970) | 9.57% | 4.78% | 16.10% | <0.001s |
| LR (2001) | 18.35% | 4.49% | 42.00% | 0.016s |
| NEH (1983) | 3.25% | 0.44% | 7.22% | 0.004s |
| NEH+VND | 1.89% | 0.41% | 4.72% | 0.038s |
| SA (0.5s) | ~3% | — | — | 0.5s |
| TS (0.5s) | ~2% | — | — | 0.5s |
| **IG (0.5s)** | **0.56%** | **0.00%** | **1.26%** | **0.5s** |

*ARPD = Average Relative Percentage Deviation from best known solution*

---

## Algorithm Taxonomy

```
                    PFSP Solution Methods
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
       Exact         Constructive      Improvement
          │           Heuristics        Methods
          │                │                 │
    ┌─────┼─────┐    ┌────┼────┐      ┌─────┼──────┐
    │     │     │    │    │    │      │     │      │
 Johnson B&B  MIP  Palmer CDS NEH  Local  Tabu  Simulated
  Rule              Gupta  LR      Search Search Annealing
 (F2)                                │
                                    ┌┼──────┐
                               Swap Insert Or-opt VND
                                          │
                                   Iterated Greedy
                                  (destroy + NEH rebuild)
```

---

## Key Insights

> **NEH (1983)** remains the gold standard constructive heuristic after 40+ years. Its insertion mechanism implicitly explores $O(n^2)$ solutions, and it serves as the foundation for modern methods like Iterated Greedy.

> **Iterated Greedy** dominates PFSP because its destroy-reconstruct cycle creates large jumps in the solution space that single-move neighborhoods (swap, insertion) cannot achieve. By removing $d$ jobs and reinserting them via NEH, IG effectively "teleports" between basins of attraction.

> **Adaptive cooling** is critical for SA — the temperature schedule must match the time budget. Without it, SA either explores too long (never converges) or converges too fast (misses good regions).

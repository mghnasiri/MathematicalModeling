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
├── exact/
│   ├── johnsons_rule.py           # Optimal for F2||Cmax
│   └── branch_and_bound.py        # B&B with Taillard bound
├── heuristics/
│   ├── neh.py                     # NEH constructive heuristic
│   ├── cds.py                     # CDS heuristic
│   └── palmers_slope.py           # Palmer's slope index
├── metaheuristics/
│   ├── iterated_greedy.py         # IG (Ruiz & Stützle)
│   └── genetic_algorithm.py       # GA with permutation encoding
└── tests/
    └── test_flow_shop.py
```

---

## Key Insight

> The **NEH heuristic** (1983) is remarkable — a simple constructive algorithm that remains competitive with metaheuristics after 40+ years. Its quality comes from the insertion mechanism: by trying each job in every position, it implicitly explores $O(n^2)$ solutions. Modern algorithms like Iterated Greedy still use NEH as their reconstruction phase.

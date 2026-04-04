# p-Median Problem (PMP)

## 1. Problem Definition

- **Input:**
  - $n$ demand points and $m$ candidate facility locations
  - Distances/costs $d_{ij}$ from facility $i$ to demand point $j$
  - Demand weights $w_j$ for each demand point
  - Number of facilities to open: $p$
- **Decision:** Select exactly $p$ facilities to open; assign each demand point to nearest open facility
- **Objective:** Minimize total weighted distance $\sum_j w_j \min_i d_{ij}$
- **Constraints:** Exactly $p$ facilities open; each customer to one facility
- **Classification:** NP-hard for general $p$ (Kariv & Hakimi, 1979)

---

## 2. Mathematical Formulation

$$\min \sum_{i=1}^{m} \sum_{j=1}^{n} w_j d_{ij} x_{ij} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \quad \text{(each customer assigned)} \tag{2}$$

$$x_{ij} \leq y_i \quad \forall i, j \quad \text{(assign to open facilities)} \tag{3}$$

$$\sum_{i=1}^{m} y_i = p \quad \text{(exactly } p \text{ facilities)} \tag{4}$$

$$y_i \in \{0,1\},\; x_{ij} \in \{0,1\} \tag{5}$$

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Capacitated p-Median | `variants/capacitated/` | Each facility has a capacity limit |

---

## 4. Solution Methods

### 4.1 Constructive Heuristics

- **Greedy:** Iteratively open the facility giving the largest cost reduction, until $p$ are open. $O(p \cdot m \cdot n)$.
- **Teitz-Bart Interchange:** Start with $p$ random facilities. Try swapping each open facility with each closed one; accept if cost improves. Iterate until no improving swap exists.

```
TEITZ-BART(d, w, p, m):
  S ← random subset of p facilities from {1,...,m}
  improved ← true
  while improved:
    improved ← false
    for each i ∈ S:                      // open facility
      for each j ∉ S:                    // closed facility
        S' ← S \ {i} ∪ {j}
        if cost(S') < cost(S):
          S ← S'; improved ← true
  return S
```

### 4.2 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Swap open/closed facility |
| 2 | Simulated Annealing (SA) | Trajectory | Swap with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Facility swap with tabu on recently changed |
| 4 | Iterated Greedy (IG) | Trajectory | Close facilities + greedy reopen |
| 5 | Genetic Algorithm (GA) | Population | Binary/subset encoding |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | 1-swap → 2-swap → block-swap |

---

## 5. Implementations in This Repository

```
p_median/
├── instance.py                    # PMedianInstance, validation
├── heuristics/
│   └── greedy_pmedian.py          # Greedy, Teitz-Bart interchange
├── metaheuristics/
│   ├── local_search.py            # Facility swap
│   ├── simulated_annealing.py     # SA
│   ├── tabu_search.py             # TS
│   ├── iterated_greedy.py         # IG
│   ├── genetic_algorithm.py       # GA
│   └── vns.py                     # VNS
├── variants/
│   └── capacitated/               # Capacitated p-median
└── tests/                         # 7 test files
    ├── test_p_median.py
    ├── test_pm_sa.py, test_pm_ts.py, test_pm_ig.py
    ├── test_pm_ls.py, test_pm_vns.py, test_pm_ga.py
```

**Total:** 2 heuristics (1 file), 6 metaheuristics/LS, 1 variant, 7 test files.

---

## 6. Key References

- Hakimi, S.L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. *Operations Research*, 12(3), 450-459.
- Kariv, O. & Hakimi, S.L. (1979). An algorithmic approach to network location problems. *SIAM J. Applied Math.*, 37(3), 539-560.
- Teitz, M.B. & Bart, P. (1968). Heuristic methods for estimating the generalized vertex median of a weighted graph. *Operations Research*, 16(5), 955-961.
- Reese, J. (2006). Solution methods for the p-median problem: An annotated bibliography. *Networks*, 48(3), 125-142.

# 0-1 Knapsack Problem (KP01)

## 1. Problem Definition

- **Input:**
  - A set of $n$ items $I = \{1, 2, \ldots, n\}$
  - Each item $i$ has weight $w_i > 0$ and value (profit) $v_i > 0$
  - Knapsack capacity $W > 0$
- **Decision:** Select a subset $S \subseteq I$ of items to pack
- **Objective:** Maximize total value $\sum_{i \in S} v_i$
- **Constraints:** Total weight $\sum_{i \in S} w_i \leq W$; each item is either taken or not ($x_i \in \{0,1\}$)
- **Classification:** Combinatorial optimization, binary integer program
- **Complexity:**
  - NP-hard (Karp, 1972) — by reduction from SUBSET-SUM
  - **Weakly** NP-hard — admits pseudo-polynomial DP in $O(nW)$
  - FPTAS exists: $(1-\varepsilon)$-approximation in $O(n^2 / \varepsilon)$ via value scaling
  - The greedy combined heuristic is a $1/2$-approximation

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of items | $\mathbb{Z}^+$ |
| $v_i$ | Value (profit) of item $i$ | $\mathbb{R}_{>0}$ |
| $w_i$ | Weight of item $i$ | $\mathbb{R}_{>0}$ |
| $W$ | Knapsack capacity | $\mathbb{R}_{>0}$ |
| $x_i$ | 1 if item $i$ is selected, 0 otherwise | $\{0, 1\}$ |

### 0-1 Knapsack Formulation

$$\max \sum_{i=1}^{n} v_i\, x_i \tag{1}$$

$$\text{s.t.} \quad \sum_{i=1}^{n} w_i\, x_i \leq W \tag{2}$$

$$x_i \in \{0, 1\} \quad \forall\, i \in I \tag{3}$$

### LP Relaxation

Replace (3) with $0 \leq x_i \leq 1$. The LP optimum is obtained by the **Dantzig bound**: sort items by decreasing value density $v_i/w_i$, pack greedily until capacity is reached, and fractionally pack the split item. The LP value provides an upper bound used in B&B.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Bounded Knapsack (BKP) | `variants/bounded/` | Each item has max copies $b_i$ |
| Multiple Knapsack (mKP) | `variants/multiple/` | Multiple knapsacks with different capacities |
| Multidimensional (MKP) | `variants/multidimensional/` | Multiple weight dimensions (weight, volume, etc.) |
| Subset Sum (SSP) | `variants/subset_sum/` | $v_i = w_i$; find subset summing to exactly $W$ |

### 3.1 Bounded Knapsack

Each item $i$ can be taken up to $b_i$ times. Reducible to 0-1 KP via binary representation: replace item $i$ with items of weight $2^k w_i$ for $k = 0, 1, \ldots, \lfloor \log_2 b_i \rfloor$.

### 3.2 Multidimensional Knapsack

$m$ weight dimensions: $\sum_i w_{ij} x_i \leq W_j$ for $j = 1, \ldots, m$. Strongly NP-hard even for $m = 2$. LP relaxation is tighter than 0-1 KP because multiple constraints interact.

### 3.3 Subset Sum

Special case where $v_i = w_i$. The question becomes: "Is there a subset summing to exactly $W$?" This is NP-complete but solvable in $O(nW)$ pseudo-polynomial time.

---

## 4. Benchmark Instances

### Standard Libraries

- **Pisinger instances:** Correlated, uncorrelated, weakly correlated, subset sum, circle instances. *URL:* http://www.diku.dk/~pisinger/codes.html
- **OR-Library:** Beasley's knapsack instances (up to $n = 10,000$)

### Small Illustrative Instance

```
4 items, capacity W = 10
Items: (w, v) = [(5, 10), (4, 40), (6, 30), (3, 50)]
Optimal: take items 2, 4 → value = 90, weight = 7 ≤ 10
```

### Instances in This Repository

| Instance | Items | Optimal | Type |
|----------|-------|---------|------|
| small4 | 4 | 35 | Handcrafted |
| medium8 | 8 | 300 | Moderate |
| strongly_correlated_10 | 10 | — | $v_i = w_i + 10$ |

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Dynamic Programming — $O(nW)$

**Idea:** Build a table $T[i][c]$ = max value using items $1, \ldots, i$ with capacity $c$.

$$T[i][c] = \max\bigl(T[i{-}1][c],\; T[i{-}1][c{-}w_i] + v_i\bigr) \quad \text{if } c \geq w_i$$

Backtrack from $T[n][W]$ to recover the selected items.

**Space optimization:** Only two rows needed (current and previous), reducing space from $O(nW)$ to $O(W)$.

```
ALGORITHM KnapsackDP(v[1..n], w[1..n], W)
  T ← 2D array of zeros, size (n+1) × (W+1)
  FOR i = 1 TO n:
    FOR c = 0 TO W:
      T[i][c] ← T[i-1][c]
      IF c ≥ w[i]:
        T[i][c] ← max(T[i][c], T[i-1][c-w[i]] + v[i])
  RETURN T[n][W]
```

#### Branch and Bound

**Idea:** DFS tree where each node branches on including/excluding item $i$. Upper bound at each node: LP relaxation (Dantzig bound) on remaining items. Prune when bound ≤ incumbent.

**Practical limit:** Very efficient on most instances; practical for $n$ up to thousands (depends on correlation structure).

### 5.2 Constructive Heuristics

| Method | Quality | Complexity |
|--------|---------|-----------|
| Greedy (value density) | No guarantee | $O(n \log n)$ |
| Greedy (max value) | No guarantee | $O(n \log n)$ |
| Greedy (combined) | **1/2-approximation** | $O(n \log n)$ |

The **combined greedy** takes the better of greedy-by-density and the single most valuable item that fits. This achieves at least half the optimal value.

### 5.3 Metaheuristics

This repository implements **6 metaheuristics** for 0-1 knapsack:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Bit-flip, swap neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | Bit-flip with infeasibility penalty |
| 3 | Tabu Search (TS) | Trajectory | Bit-flip with item-level tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove items + greedy reconstruct |
| 5 | Genetic Algorithm (GA) | Population | Binary encoding, uniform crossover, repair operator |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | 1-flip → 2-flip → swap |

**Repair operator** (for GA): If offspring violates capacity, remove items in order of increasing value density until feasible. Then add items in order of decreasing density until no more fit.

---

## 6. Implementation Guide

### Modeling Tips

- **DP table overflow:** For large $W$, the DP table doesn't fit in memory. Use B&B instead, or value-based DP ($O(n \cdot V_{\max})$).
- **Greedy as warm-start:** Always run greedy-combined before B&B to get a good incumbent immediately.
- **Floating-point in LP bound:** The Dantzig bound computation requires sorting by $v_i/w_i$. Use integer arithmetic where possible.

### Common Pitfalls

- **Item ordering:** DP processes items in any order (1 to $n$). Greedy and B&B need items sorted by value density.
- **Strongly correlated instances** ($v_i \approx w_i + c$) are hardest for B&B because the LP relaxation is tight, leaving many near-optimal solutions.

---

## 7. Computational Results Summary

| Method | Gap (n=20) | Gap (n=100) | Gap (n=1000) |
|--------|-----------|-------------|-------------|
| DP | 0% | 0% (if $W$ fits) | 0% (if $W$ fits) |
| B&B (LP bound) | 0% | 0% | 0% (seconds) |
| Greedy combined | ≤50% | ≤50% | ≤50% |
| GA (repair) | <1% | <1% | 1-3% |
| SA | <1% | <1% | 1-3% |

**Practical guidance:** DP is the method of choice when $W$ fits in memory. B&B handles arbitrarily large $W$. Metaheuristics are mainly useful for variants (multidimensional, multiple) where exact methods are prohibitive.

---

## 8. Implementations in This Repository

```
knapsack/
├── instance.py                    # KnapsackInstance, KnapsackSolution, validation
│
├── exact/
│   ├── dynamic_programming.py     # DP — O(nW) pseudo-polynomial
│   └── branch_and_bound.py        # B&B with LP relaxation bound
│
├── heuristics/
│   └── greedy.py                  # Value-density, max-value, combined (1/2-approx)
│
├── metaheuristics/
│   ├── local_search.py            # Bit-flip, swap neighborhoods
│   ├── simulated_annealing.py     # SA with infeasibility penalty
│   ├── tabu_search.py             # TS with item-level tabu
│   ├── iterated_greedy.py         # IG: remove + greedy reconstruct
│   ├── genetic_algorithm.py       # GA: binary encoding + repair
│   └── vns.py                     # VNS: 1-flip → 2-flip → swap
│
├── variants/
│   ├── bounded/                   # Bounded KP (max copies per item)
│   ├── multiple/                  # Multiple knapsacks
│   ├── multidimensional/          # Multi-weight dimensions
│   └── subset_sum/                # v_i = w_i, find exact sum
│
└── tests/                         # 6 test files
    ├── test_knapsack.py           # Core algorithms (DP, B&B, greedy, GA)
    ├── test_knapsack_ls.py        # Local Search
    ├── test_knapsack_sa.py        # Simulated Annealing
    ├── test_knapsack_ts.py        # Tabu Search
    ├── test_knapsack_vns.py       # VNS
    └── test_kp_ig.py              # Iterated Greedy
```

**Total:** 2 exact methods, 3 greedy variants (1 file), 6 metaheuristics/LS, 4 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, 85-103.
- Dantzig, G.B. (1957). Discrete-variable extremum problems. *Operations Research*, 5(2), 266-288.
- Horowitz, E. & Sahni, S. (1974). Computing partitions with applications to the knapsack problem. *Journal of the ACM*, 21(2), 277-292.

### Books

- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.
- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. John Wiley.

### Approximation

- Ibarra, O.H. & Kim, C.E. (1975). Fast approximation algorithms for the knapsack and sum of subset problems. *Journal of the ACM*, 22(4), 463-468.

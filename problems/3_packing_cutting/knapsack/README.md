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

### LP Relaxation and Dantzig Bound

Replace (3) with $0 \leq x_i \leq 1$. The LP optimum is obtained by the **Dantzig bound** (Dantzig, 1957):

1. Sort items by decreasing value density $v_i / w_i$, giving order $\sigma(1), \sigma(2), \ldots, \sigma(n)$.
2. Pack items greedily in this order until item $s$ (the **split item** or **break item**) exceeds the remaining capacity.
3. Fractionally include the split item: $x_{\sigma(s)} = (W - \sum_{j=1}^{s-1} w_{\sigma(j)}) / w_{\sigma(s)}$.

The resulting LP value $z^{LP}$ provides an upper bound on the integer optimum: $z^{*} \leq z^{LP}$. Since the LP relaxation has a single constraint (besides bounds), at most one variable takes a fractional value, so $z^{LP} - z^{*} < v_{\max}$. This bound is computed in $O(n \log n)$ and drives both B&B pruning and the FPTAS below.

### FPTAS via Value Scaling

The knapsack problem admits a **Fully Polynomial-Time Approximation Scheme (FPTAS)** (Ibarra & Kim, 1975):

1. Let $v_{\max} = \max_i v_i$ and set the scaling factor $K = \varepsilon \cdot v_{\max} / n$.
2. Define scaled profits $\hat{v}_i = \lfloor v_i / K \rfloor$.
3. Solve the scaled instance exactly with DP. The maximum scaled profit is at most $n / \varepsilon$, giving an $O(n^2 / \varepsilon)$ algorithm.
4. The solution is a $(1 - \varepsilon)$-approximation to the original problem.

This is the theoretically strongest polynomial-time scheme for 0-1 KP, but in practice the standard pseudo-polynomial DP or B&B is faster for moderate instance sizes.

### Core Concept

Pisinger (1997) observed that the optimal solution to any 0-1 KP instance differs from the LP relaxation solution in only a small "core" of items near the split item. Specifically, if items are sorted by value density, only items within a narrow band around the break item need to be considered by an exact solver. This **core** typically contains $O(\sqrt{n})$ items in practice, which explains why B&B with the Dantzig bound solves even large instances quickly: most items are fixed to their LP values and only the core items require branching.

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

### Instances in This Repository (Best Known Solutions)

All optimal values verified by both DP and B&B solvers in this repository.

| Instance | $n$ | $W$ | Optimal | Optimal Items | Weight | Type |
|----------|-----|-----|---------|---------------|--------|------|
| small4 | 4 | 7 | **35** | {1, 2} | 7 | Handcrafted |
| medium8 | 8 | 48 | **300** | {0, 3, 4, 7} | 48 | Moderate |
| strongly_correlated_10 | 10 | 80 | **130** | {0, 3, 4, 5, 6} | 80 | $v_i = w_i + 10$ |

#### Instance Details

**small4** ($W = 7$): Items $(w, v)$ = [(2, 10), (3, 15), (4, 20), (1, 5)]. Two optimal solutions exist: $\{1,2\}$ and $\{0,2,3\}$, both achieving value 35 at weight 7.

**medium8** ($W = 48$): Items $(w, v)$ = [(10, 60), (20, 100), (15, 70), (25, 120), (5, 50), (10, 45), (30, 80), (8, 70)]. The optimal selection packs items 0, 3, 4, 7 to capacity exactly.

**strongly_correlated_10** ($W = 80$): Weights $[10, 20, 30, 15, 25, 12, 18, 22, 8, 35]$ with $v_i = w_i + 10$. All items have similar value density ($\approx 1.3$--$2.3$), making greedy heuristics less effective. The LP gap is small, which also challenges B&B.

### Pisinger Benchmark Classes

The Pisinger benchmark generator (Pisinger, 2005) produces instances in five correlation classes, each with different difficulty profiles:

| Class | Definition | DP Difficulty | B&B Difficulty |
|-------|-----------|---------------|----------------|
| Uncorrelated | $v_i, w_i$ uniform random | Easy | Easy |
| Weakly correlated | $v_i \in [w_i - R, w_i + R]$ | Moderate | Moderate |
| Strongly correlated | $v_i = w_i + R$ | Hard (tight LP) | Hard |
| Subset sum | $v_i = w_i$ | Hard | Very hard |
| Inverse strongly corr. | $w_i = v_i + R$ | Hard | Hard |

Strongly correlated and subset-sum instances are the standard stress tests for exact solvers because their LP relaxation gap is near zero, producing many near-optimal solutions.

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Dynamic Programming — $O(nW)$

**Idea:** Build a table $T[i][c]$ = max value using items $1, \ldots, i$ with capacity $c$.

$$T[i][c] = \max\bigl(T[i{-}1][c],\; T[i{-}1][c{-}w_i] + v_i\bigr) \quad \text{if } c \geq w_i$$

Backtrack from $T[n][W]$ to recover the selected items.

**2D pseudocode** (standard formulation):

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

**Space-optimized pseudocode** (single 1D array, used in our implementation):

```
ALGORITHM KnapsackDP_SpaceOpt(v[1..n], w[1..n], W)
  dp[0..W] ← 0
  FOR j = 1 TO n:
    FOR c = W DOWN TO w[j]:          ← reverse order prevents reuse of item j
      dp[c] ← max(dp[c], dp[c - w[j]] + v[j])
  RETURN dp[W]
```

The reverse iteration over capacities is the key insight: processing $c$ from $W$ down to $w_j$ ensures that each item $j$ is used at most once (the unbounded variant iterates forward). This reduces space from $O(nW)$ to $O(W)$, though recovering the selected items requires either an auxiliary `keep[n][W]` boolean array (as in our implementation) or a second backward pass.

**Practical limit:** The table has $n \times W$ entries, so DP is feasible when $n \times W \lesssim 10^8$. For large $W$ with small $n$, B&B is preferable. For large $n$ with small max-value, a profit-indexed DP variant running in $O(n \cdot v_{\max})$ can be used instead.

#### Branch and Bound

**Idea:** DFS tree where each node branches on including/excluding item $i$. Upper bound at each node: LP relaxation (Dantzig bound) on remaining items. Prune when bound $\leq$ incumbent.

```
ALGORITHM KnapsackBB(v[1..n], w[1..n], W)
  Sort items by v[i]/w[i] descending
  incumbent ← GreedyDensity(v, w, W)        ← warm-start
  stack ← {(level=0, value=0, weight=0, selected=[])}

  WHILE stack not empty:
    (level, val, wt, sel) ← stack.pop()
    IF level = n:
      IF val > incumbent.value:
        incumbent ← (val, sel)
      CONTINUE
    item ← sorted_items[level]

    // Exclude branch
    bound_excl ← DantzigBound(level+1, val, wt)
    IF bound_excl > incumbent.value:
      stack.push(level+1, val, wt, sel)

    // Include branch (if feasible)
    IF wt + w[item] ≤ W:
      bound_incl ← DantzigBound(level+1, val+v[item], wt+w[item])
      IF bound_incl > incumbent.value:
        stack.push(level+1, val+v[item], wt+w[item], sel ∪ {item})

  RETURN incumbent
```

**Key implementation details:**
- Items are sorted by value-to-weight ratio before search, aligning the branching order with the LP relaxation for tighter early pruning.
- Greedy warm-start provides a strong initial incumbent, allowing many branches to be pruned immediately.
- The `DantzigBound` function greedily fills remaining capacity with fractional items in $O(n - \text{level})$ per call.

**Practical limit:** Very efficient on most instances; practical for $n$ up to thousands. Strongly correlated instances ($v_i \approx w_i + c$) are the hardest because the LP gap is small, producing many near-optimal nodes that resist pruning.

### 5.2 Constructive Heuristics

| Method | Quality | Complexity |
|--------|---------|-----------|
| Greedy (value density) | No guarantee | $O(n \log n)$ |
| Greedy (max value) | No guarantee | $O(n \log n)$ |
| Greedy (combined) | **1/2-approximation** | $O(n \log n)$ |

The **combined greedy** takes the better of greedy-by-density and the single most valuable feasible item. This achieves at least half the optimal value.

**1/2-approximation proof sketch** (Sahni, 1975):

Let $G$ be the greedy-by-density value and $v_{\max}$ be the value of the most valuable feasible item. The LP relaxation satisfies $z^{LP} \leq G + v_s$, where $v_s$ is the value of the split item (the first item that did not fit). Since $v_s \leq v_{\max}$, we have $z^{*} \leq z^{LP} \leq G + v_{\max}$. The combined heuristic returns $\max(G, v_{\max}) \geq (G + v_{\max}) / 2 \geq z^{*} / 2$. This bound is tight: consider $n=2$ items with $(w_1, v_1) = (1, 2)$ and $(w_2, v_2) = (W, W)$ for large $W$; greedy-by-density picks item 1 only (value 2), while the optimum is $W$.

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

#### GA Parameter Table

| Parameter | Default | Notes |
|-----------|---------|-------|
| Encoding | Binary $x \in \{0,1\}^n$ | One bit per item |
| Population size | 50 | Initialized randomly, repaired to feasibility |
| Generations | 200 | Convergence typically within 100 for $n \leq 50$ |
| Selection | Tournament ($k = 3$) | Higher $k$ increases selection pressure |
| Crossover | Uniform | Each bit from either parent with $p = 0.5$ |
| Mutation | Bit-flip ($p = 1/n$) | Independent flip per gene |
| Elitism | 1 individual | Best solution always survives |

#### Repair Operator Detail

The repair operator (Chu & Beasley, 1998) enforces feasibility in two phases:

1. **Drop phase:** If the total weight exceeds capacity, repeatedly remove the selected item with the **lowest** value-to-weight ratio $v_i / w_i$ until the solution becomes feasible.
2. **Add phase** (optional, not used in our implementation): Attempt to add unselected items in order of **decreasing** value density until no further item fits.

The drop-only variant guarantees feasibility while preserving as much high-density content as possible. The combined drop+add variant can improve solution quality but adds $O(n \log n)$ overhead per repair.

---

## 6. Implementation Guide

### Complexity Comparison

| Method | Time | Space | Optimal? | Practical Range |
|--------|------|-------|----------|-----------------|
| DP (2D table) | $O(nW)$ | $O(nW)$ | Yes | $nW \lesssim 10^8$ |
| DP (1D array) | $O(nW)$ | $O(W)$ | Yes | $nW \lesssim 10^8$ |
| B&B (Dantzig) | $O(2^n)$ worst | $O(n)$ per node | Yes | $n \lesssim 10^4$ (uncorrelated) |
| Greedy density | $O(n \log n)$ | $O(n)$ | No | Any $n$ |
| Greedy combined | $O(n \log n)$ | $O(n)$ | 1/2-approx | Any $n$ |
| GA (repair) | $O(G \cdot P \cdot n)$ | $O(P \cdot n)$ | No | Any $n$ |

Where $G$ = generations, $P$ = population size.

### Modeling Tips

- **DP table overflow:** For large $W$, the DP table doesn't fit in memory. Use B&B instead, or value-based DP ($O(n \cdot V_{\max})$).
- **Greedy as warm-start:** Always run greedy-combined before B&B to get a good incumbent immediately. Our B&B implementation sorts items by density and packs greedily as the initial incumbent.
- **Floating-point in LP bound:** The Dantzig bound computation requires sorting by $v_i/w_i$. Use integer arithmetic where possible. Our implementation converts weights to integers via rounding for the DP table.
- **Integer weights for DP:** The DP solver in `exact/dynamic_programming.py` casts weights to `int(round(w))`. Ensure instance weights are integers or near-integer for correct results.
- **Choosing between DP and B&B:** If $W$ is large but $n$ is moderate and the instance is uncorrelated, B&B will be faster. If $W$ is moderate relative to $n$, DP gives guaranteed $O(nW)$ time regardless of correlation structure.

### Common Pitfalls

- **Item ordering:** DP processes items in any order (1 to $n$). Greedy and B&B need items sorted by value density.
- **Strongly correlated instances** ($v_i \approx w_i + c$) are hardest for B&B because the LP relaxation is tight, leaving many near-optimal solutions.
- **Reverse iteration in 1D DP:** Iterating capacities from $W$ down to $w_j$ is essential for 0-1 KP. Forward iteration (as in the unbounded variant) allows an item to be selected multiple times.
- **Empty instance edge case:** All solvers handle $n = 0$ by returning an empty selection with value 0. The GA explicitly checks for this before initializing the population.
- **Repair vs. penalty in metaheuristics:** The GA uses a repair operator to enforce feasibility directly. SA and TS instead use an infeasibility penalty, which allows temporary violations to explore broader search regions before converging to feasible solutions.

---

## 7. Computational Results Summary

### Optimality Gap by Instance Size

| Method | Gap (n=20) | Gap (n=100) | Gap (n=1000) |
|--------|-----------|-------------|-------------|
| DP | 0% | 0% (if $W$ fits) | 0% (if $W$ fits) |
| B&B (LP bound) | 0% | 0% | 0% (seconds) |
| Greedy combined | ≤50% | ≤50% | ≤50% |
| GA (repair) | <1% | <1% | 1-3% |
| SA | <1% | <1% | 1-3% |

### Results on Repository Instances

| Instance | DP | B&B | Greedy Combined | GA (seed=42) |
|----------|-----|------|-----------------|-------------|
| small4 ($n=4, W=7$) | **35** | **35** | 35 | 35 |
| medium8 ($n=8, W=48$) | **300** | **300** | 300 | 300 |
| strongly_correlated_10 ($n=10, W=80$) | **130** | **130** | 130 | 130 |

Both exact methods find proven optima. The greedy heuristic and GA also reach optimality on these small instances, though this is not guaranteed for larger problems.

### Method Selection Guide

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| $nW \leq 10^8$, integer weights | DP (1D) | Guaranteed optimal, fast, simple |
| Large $W$, uncorrelated | B&B | Dantzig bound prunes aggressively |
| Large $W$, strongly correlated | B&B (expect slower) | Tight LP gap limits pruning |
| Multidimensional KP ($m \geq 2$) | GA or SA | Exact methods scale poorly |
| Quick lower bound needed | Greedy combined | $O(n \log n)$, 1/2-approx |
| Upper bound needed | Dantzig bound | $O(n \log n)$, LP relaxation |

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

1. Dantzig, G.B. (1957). Discrete-variable extremum problems. *Operations Research*, 5(2), 266-288. https://doi.org/10.1287/opre.5.2.266 -- Introduced the LP relaxation bound (Dantzig bound) via fractional greedy packing.
2. Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, 85-103. https://doi.org/10.1007/978-1-4684-2001-2_9 -- Proved 0-1 KP is NP-hard by reduction from SUBSET-SUM.
3. Horowitz, E. & Sahni, S. (1974). Computing partitions with applications to the knapsack problem. *Journal of the ACM*, 21(2), 277-292. https://doi.org/10.1145/321812.321823 -- Meet-in-the-middle $O(2^{n/2})$ exact algorithm and early B&B techniques.
4. Ibarra, O.H. & Kim, C.E. (1975). Fast approximation algorithms for the knapsack and sum of subset problems. *Journal of the ACM*, 22(4), 463-468. https://doi.org/10.1145/321906.321909 -- First FPTAS for the knapsack problem via profit scaling.
5. Sahni, S. (1975). Approximate algorithms for the 0/1 knapsack problem. *Journal of the ACM*, 22(1), 115-124. https://doi.org/10.1145/321864.321873 -- Established the 1/2-approximation guarantee for greedy heuristics.
6. Balas, E. & Zemel, E. (1980). An algorithm for large zero-one knapsack problems. *Operations Research*, 28(5), 1130-1154. https://doi.org/10.1287/opre.28.5.1130 -- Introduced the core concept for efficient exact solution of large instances.
7. Pisinger, D. (1997). A minimal algorithm for the 0-1 knapsack problem. *Operations Research*, 45(5), 758-767. https://doi.org/10.1287/opre.45.5.758 -- Exploited the core concept: only items near the LP break point matter for optimality.

### Books

8. Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. John Wiley & Sons, Chichester. -- Comprehensive treatment of exact algorithms, upper bounds, and computational experiments; includes the MT1/MT2 algorithms.
9. Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer-Verlag, Berlin. https://doi.org/10.1007/978-3-540-24777-7 -- Definitive modern reference covering all variants (bounded, unbounded, multidimensional, multi-objective), approximation schemes, and the core concept.

### Metaheuristics and Applications

10. Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the multidimensional knapsack problem. *Journal of Heuristics*, 4(1), 63-86. https://doi.org/10.1023/A:1009642405419 -- Binary-encoded GA with repair operator for multidimensional KP.
11. Pisinger, D. (2005). Where are the hard knapsack problems? *Computers & Operations Research*, 32(9), 2271-2284. https://doi.org/10.1016/j.cor.2004.03.002 -- Systematic study of instance difficulty across correlation classes; generator for benchmark instances.
12. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. -- Foundational work establishing the DP paradigm used in pseudo-polynomial knapsack algorithms.

# 0-1 Knapsack Problem (KP01)

## Problem Definition

Given $n$ items, each with weight $w_i$ and value $v_i$, and a knapsack with capacity $W$, select a subset $S \subseteq \{1, \ldots, n\}$ to maximize total value without exceeding the weight capacity.

## Mathematical Formulation

$$\max \sum_{i=1}^{n} v_i x_i$$

$$\text{s.t.} \quad \sum_{i=1}^{n} w_i x_i \leq W, \quad x_i \in \{0, 1\}$$

## Complexity

- NP-hard (Karp, 1972)
- Weakly NP-hard — admits pseudo-polynomial DP in $O(nW)$
- FPTAS exists with $(1-\varepsilon)$-approximation in $O(n / \varepsilon^2)$

## Solution Approaches

| Method | Complexity | Type | Description |
|--------|-----------|------|-------------|
| Dynamic Programming | $O(nW)$ | Exact | Bottom-up tabulation, pseudo-polynomial |
| Branch & Bound | $O(2^n)$ worst | Exact | DFS with LP relaxation upper bound |
| Greedy (value density) | $O(n \log n)$ | Heuristic | Sort by $v_i/w_i$, pack greedily |
| Greedy (combined) | $O(n \log n)$ | Heuristic | Best of density and max-value, 1/2-approx |
| Genetic Algorithm | $O(\text{pop} \cdot \text{gen} \cdot n)$ | Metaheuristic | Binary encoding, repair operator |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Bounded Knapsack (BKP)](variants/bounded/) | `variants/bounded/` | Each item has a maximum number of copies available |
| [Multiple Knapsack (mKP)](variants/multiple/) | `variants/multiple/` | Multiple knapsacks with different capacities |
| [Multidimensional Knapsack (MKP)](variants/multidimensional/) | `variants/multidimensional/` | Multiple resource dimensions (weight, volume, etc.) |
| [Subset Sum (SSP)](variants/subset_sum/) | `variants/subset_sum/` | Find subset summing to exactly a target value |

## Key References

- Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, 85-103. https://doi.org/10.1007/978-1-4684-2001-2_9
- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer. https://doi.org/10.1007/978-3-540-24777-7
- Dantzig, G.B. (1957). Discrete-variable extremum problems. *Oper. Res.*, 5(2), 266-288. https://doi.org/10.1287/opre.5.2.266
- Horowitz, E. & Sahni, S. (1974). Computing partitions with applications to the knapsack problem. *JACM*, 21(2), 277-292. https://doi.org/10.1145/321812.321823

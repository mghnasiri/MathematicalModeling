# Stochastic Knapsack Problem

## 1. Problem Definition

- **Input:** $n$ items with deterministic values $v_i$ and random weights $W_i(s)$ across $S$ scenarios, knapsack capacity $W$, risk level $\alpha$
- **Decision:** Binary selection $x_i \in \{0, 1\}$ for each item
- **Objective:** Maximize total value $\sum_i v_i x_i$
- **Constraints:** Capacity constraint must hold in expectation or with probability $\geq 1 - \alpha$
- **Classification:** NP-hard (generalizes deterministic 0-1 knapsack)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of items |
| $v_i$ | Deterministic value of item $i$ |
| $W_i(s)$ | Weight of item $i$ under scenario $s$ |
| $W$ | Knapsack capacity |
| $S$ | Number of weight scenarios |
| $\alpha$ | Maximum allowed infeasibility probability |
| $p_s$ | Probability of scenario $s$ |

### Chance-Constrained Formulation

$$\max \sum_{i=1}^{n} v_i x_i \tag{1}$$

$$P\!\left(\sum_{i=1}^{n} W_i x_i \leq W\right) \geq 1 - \alpha \tag{2}$$

$$x_i \in \{0, 1\} \tag{3}$$

### Expected-Capacity Variant

Replace (2) with $E\!\left[\sum_i W_i x_i\right] \leq W$, which reduces to a deterministic knapsack on mean weights.

### Small Illustrative Instance

```
n = 4, W = 15, α = 0.1
Values:  [10, 8, 12, 6]
Scenario 1 (p=0.5): weights = [5, 4, 7, 3]  → select {1,2,3}: Σw = 16 > 15 ✗
Scenario 2 (p=0.5): weights = [4, 3, 5, 2]  → select {1,2,3}: Σw = 12 ≤ 15 ✓

P(feasible for {1,2,3}) = 0.5 < 0.9  → infeasible at α=0.1
Select {1,3}: value=22, P(feasible) = 1.0 ✓
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy (mean weight) | Heuristic | $O(n \log n)$ | Value-density ranking on expected weights |
| Greedy (chance-constrained) | Heuristic | $O(n \cdot S)$ | Add items maintaining $P(\text{feasible}) \geq 1-\alpha$ |
| Simulated Annealing | Metaheuristic | $O(I \cdot n \cdot S)$ | Flip-bit neighborhood with infeasibility penalty |

### Greedy (Mean Weight)

Sort items by $v_i / E[W_i]$ descending. Add items greedily until expected capacity is full. Simple but ignores variance — high-variance items may cause frequent violations.

### Greedy (Chance-Constrained)

Like mean-weight greedy but checks $P(\sum W_i x_i \leq W) \geq 1 - \alpha$ after each addition by evaluating all scenarios. Rejects items that would violate the chance constraint.

```
GREEDY-CHANCE-CONSTRAINED(items, W, α, scenarios):
  sort items by v_i / E[W_i] descending
  selected ← ∅
  for each item i in sorted order:
    selected' ← selected ∪ {i}
    p_feas ← |{s : Σ_{j∈selected'} W_j(s) ≤ W}| / |scenarios|
    if p_feas ≥ 1 - α:
      selected ← selected'
  return selected
```

### Simulated Annealing

Flip-bit neighborhood (toggle one item). Infeasible solutions are penalized by $\lambda \cdot \max(0, \alpha - P(\text{feasible}))$. Temperature schedule auto-calibrated.

---

## 4. Implementations in This Repository

```
stochastic_knapsack/
├── instance.py                        # StochasticKnapsackInstance, StochasticKnapsackSolution
│                                      #   - feasibility_probability(), mean_weights
│                                      #   - random() factory
├── heuristics/
│   └── greedy_stochastic.py           # Mean-weight greedy, chance-constrained greedy
├── metaheuristics/
│   └── simulated_annealing.py         # Flip-bit SA with infeasibility penalty
└── tests/
    └── test_stochastic_knapsack.py    # 11 tests, 3 test classes
```

---

## 5. Key References

- Kleinberg, J., Rabani, Y. & Tardos, E. (1997). Allocating bandwidth for bursty connections. *STOC*, 664-673. https://doi.org/10.1145/258533.258661
- Dean, B.C., Goemans, M.X. & Vondrák, J. (2008). Approximating the stochastic knapsack problem: the benefit of adaptivity. *Math. Oper. Res.*, 33(4), 945-964. https://doi.org/10.1287/moor.1080.0330
- Li, J. & Yuan, Y. (2013). Stochastic combinatorial optimization. In *Handbook of Combinatorial Optimization* (pp. 2625-2670). Springer.

# Bi-Objective 0-1 Knapsack Problem

## 1. Problem Definition

- **Input:** $n$ items with weights $w_i$, two value vectors $v^1_i$ and $v^2_i$, knapsack capacity $W$
- **Decision:** Binary selection $x_i \in \{0, 1\}$
- **Objective:** Maximize both $\sum v^1_i x_i$ and $\sum v^2_i x_i$ simultaneously (find Pareto front)
- **Constraints:** $\sum w_i x_i \leq W$
- **Classification:** NP-hard. Finding the complete Pareto front is intractable in general.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of items |
| $w_i$ | Weight of item $i$ |
| $v^1_i, v^2_i$ | Values of item $i$ for objectives 1 and 2 |
| $W$ | Knapsack capacity |

### Multi-Objective Formulation

$$\max \left( \sum_{i=1}^{n} v^1_i x_i, \quad \sum_{i=1}^{n} v^2_i x_i \right) \tag{1}$$

$$\sum_{i=1}^{n} w_i x_i \leq W \tag{2}$$

$$x_i \in \{0, 1\} \tag{3}$$

### Epsilon-Constraint Method

Fix one objective as constraint: $\max \sum v^1_i x_i$ s.t. $\sum v^2_i x_i \geq \epsilon$, $\sum w_i x_i \leq W$. Sweep $\epsilon$ to trace the Pareto front.

### Small Illustrative Instance

```
n = 4, W = 10
Weights:   [3, 4, 5, 2]
Values 1:  [6, 8, 7, 3]
Values 2:  [4, 3, 9, 5]

Pareto-optimal solutions:
  {2, 4}: v1=10, v2=14, w=7  (high obj 2)
  {1, 2}: v1=14, v2=7,  w=7  (high obj 1)
  {1, 4}: v1=9,  v2=9,  w=5  (balanced)
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Epsilon-constraint | Exact | $O(K \cdot \text{KP})$ | Solve $K$ single-objective knapsacks with varying $\epsilon$ |

### Epsilon-Constraint

1. Find extreme points by optimizing each objective independently
2. Sweep $\epsilon$ from min to max of the secondary objective
3. For each $\epsilon$, solve a constrained single-objective knapsack
4. Collect non-dominated solutions

---

## 4. Implementations in This Repository

```
bi_objective_knapsack/
├── instance.py                        # BiObjectiveKnapsackInstance, BiObjectiveKnapsackSolution
│                                      #   - Fields: n, weights, values1, values2, capacity
│                                      #   - random() factory
├── heuristics/
│   └── epsilon_constraint.py          # Epsilon-constraint Pareto front generation
└── tests/
    └── test_bi_objective_knapsack.py  # Bi-objective knapsack test suite
```

---

## 5. Key References

- Ehrgott, M. (2005). *Multicriteria Optimization*. 2nd ed. Springer.
- Bazgan, C., Hugot, H. & Vanderpooten, D. (2009). Solving efficiently the 0-1 multi-objective knapsack problem. *Computers & Oper. Res.*, 36(1), 260-279.

# Multi-Objective Traveling Salesman Problem (MOTSP)

## 1. Problem Definition

- **Input:** $n$ cities with $k$ distance/cost matrices $D^1, \ldots, D^k$
- **Decision:** Hamiltonian cycle (tour) visiting each city exactly once
- **Objective:** Minimize all $k$ objectives simultaneously: $\left(\sum D^1_{\pi}, \ldots, \sum D^k_{\pi}\right)$ (find Pareto front)
- **Classification:** NP-hard (each single objective is a TSP)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of cities |
| $k$ | Number of objectives |
| $D^\ell_{ij}$ | Distance/cost from city $i$ to $j$ under objective $\ell$ |
| $\pi$ | Tour (permutation of cities) |

### Multi-Objective Formulation

$$\min \left( \sum_{i=1}^{n} D^1_{\pi(i),\pi(i+1)}, \ldots, \sum_{i=1}^{n} D^k_{\pi(i),\pi(i+1)} \right) \tag{1}$$

### Weighted Sum Scalarization

$$\min \sum_{\ell=1}^{k} \lambda_\ell \sum_{i=1}^{n} D^\ell_{\pi(i),\pi(i+1)} \tag{2}$$

where $\lambda \geq 0, \sum \lambda_\ell = 1$. Sweep $\lambda$ to find supported Pareto points.

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Weighted-sum NN | Heuristic | $O(L \cdot n^2)$ | Nearest neighbor with $L$ weight vectors |

### Weighted-Sum Nearest Neighbor

For each weight vector $\lambda$, combine the $k$ distance matrices into a single weighted matrix $D_\lambda = \sum_\ell \lambda_\ell D^\ell$. Run nearest-neighbor heuristic on $D_\lambda$. Collect non-dominated solutions across all weight vectors.

---

## 4. Implementations in This Repository

```
multi_objective_tsp/
├── instance.py                        # MultiObjectiveTSPInstance, MultiObjectiveTSPSolution
│                                      #   - Fields: n, n_objectives, distance_matrices
│                                      #   - random() factory
├── heuristics/
│   └── weighted_sum_nn.py             # Weighted-sum nearest neighbor
└── tests/
    └── test_multi_objective_tsp.py    # MOTSP test suite
```

---

## 5. Key References

- Jaszkiewicz, A. (2002). On the performance of multiple-objective genetic local search on the 0/1 knapsack problem — a comparative experiment. *IEEE Trans. Evol. Comput.*, 6(4), 402-412.
- Ehrgott, M. (2005). *Multicriteria Optimization*. 2nd ed. Springer.

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

#### Weighted-Sum NN Pseudocode

```
WEIGHTED_SUM_NN(n cities, k distance matrices D^1..D^k, L weight vectors):
    pareto_set = {}
    for each weight vector λ = (λ_1, ..., λ_k):
        D_λ[i][j] = Σ_{ℓ=1}^{k} λ_ℓ * D^ℓ[i][j]   for all i, j
        tour = NEAREST_NEIGHBOR(D_λ)
        cost_vector = (Σ D^1_tour, ..., Σ D^k_tour)
        if cost_vector is not dominated by any in pareto_set:
            remove dominated solutions from pareto_set
            pareto_set = pareto_set ∪ {(tour, cost_vector)}
    return pareto_set
```

**Complexity:** $O(L \cdot n^2)$ where $L$ is the number of weight vectors. Each NN call is $O(n^2)$.

**Limitation:** Weighted-sum scalarization can only find supported Pareto-optimal solutions (those on the convex hull of the Pareto front). Non-supported solutions require alternative approaches such as epsilon-constraint or evolutionary methods.

### Small Illustrative Instance

```
n = 4 cities, k = 2 objectives
D^1 =  [[0,2,5,7],    D^2 = [[0,6,3,1],
         [2,0,3,4],           [6,0,4,5],
         [5,3,0,1],           [3,4,0,7],
         [7,4,1,0]]           [1,5,7,0]]

λ = (0.5, 0.5): D_λ = average of D^1, D^2
NN tour from city 0: 0→1→2→3→0 with costs (10, 18) and (16, 12)
Pareto front discovered by sweeping λ over {(1,0), (0.5,0.5), (0,1)}.
```

### Applications

- **Logistics** (optimizing both distance and fuel cost for delivery routes)
- **Tourism** (minimizing travel time while maximizing scenic value)
- **Telecommunications** (routing minimizing both latency and installation cost)

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
- Lust, T. & Teghem, J. (2010). Two-phase Pareto local search for the biobjective traveling salesman problem. *J. Heuristics*, 16(3), 475-510.
- Angel, E., Bampis, E. & Gourves, L. (2004). A dynasearch neighborhood for the bicriteria traveling salesman problem. In *Metaheuristics: Computer Decision-Making* (pp. 153-176). Springer.

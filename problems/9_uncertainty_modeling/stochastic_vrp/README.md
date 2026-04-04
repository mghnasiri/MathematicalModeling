# Stochastic Vehicle Routing Problem (SVRP)

## 1. Problem Definition

- **Input:** $n$ customers with coordinates and stochastic demands $D_j(s)$ across $S$ scenarios, depot (node 0), vehicle capacity $Q$, $K$ vehicles, risk level $\alpha$
- **Decision:** A priori routes (depot → customers → depot) designed before demands are realized
- **Objective:** Minimize total distance + expected recourse cost from route overflows
- **Constraints:** Each customer visited exactly once; route demand feasible with probability $\geq 1 - \alpha$
- **Classification:** NP-hard (generalizes deterministic CVRP)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of customers |
| $K$ | Number of available vehicles |
| $Q$ | Vehicle capacity |
| $D_j(s)$ | Demand of customer $j$ under scenario $s$ |
| $d_{ij}$ | Euclidean distance between nodes $i$ and $j$ |
| $\alpha$ | Maximum allowed overflow probability per route |
| $S$ | Number of demand scenarios |

### Chance-Constrained Formulation

$$\min \sum_{k=1}^{K} \text{dist}(R_k) + \lambda \cdot E[\text{recourse}] \tag{1}$$

$$P\!\left(\sum_{j \in R_k} D_j \leq Q\right) \geq 1 - \alpha \quad \forall k \tag{2}$$

$$\bigcup_k R_k = \{1, \ldots, n\}, \quad R_k \cap R_l = \emptyset \tag{3}$$

### Recourse Policy

When a vehicle's load exceeds $Q$ mid-route, it returns to the depot to unload and resumes. The recourse cost approximation: $\text{penalty} = 2 \cdot \bar{d} \cdot P(\text{overflow})$ where $\bar{d}$ is the average distance on the route.

### Small Illustrative Instance

```
n = 4 customers, Q = 30, K = 2, α = 0.1
Depot at (50, 50)
Scenario 1 (p=0.5): demands = [10, 12, 8, 15]
Scenario 2 (p=0.5): demands = [18, 20, 14, 22]

Route 1 = [1, 2]: P(D1+D2 > 30) = P(10+12, 18+20) = P(22, 38) = 0.5 > α ✗
Route 1 = [1, 3]: P(D1+D3 > 30) = P(18, 32) = 0.5 > α ✗
→ Must split more conservatively under stochastic demands
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Chance-Constrained CW | Heuristic | $O(n^2 \log n \cdot S)$ | Clarke-Wright savings with $P(\text{overflow}) \leq \alpha$ check |
| Mean-Demand Savings | Heuristic | $O(n^2 \log n)$ | CW with expected demands as proxy |
| Simulated Annealing | Metaheuristic | $O(I \cdot n \cdot S)$ | Relocate/swap/2-opt with recourse penalty |

### Chance-Constrained Clarke-Wright

Standard savings algorithm but route merges are rejected if the merged route would violate $P(\sum D_j > Q) \leq \alpha$. Evaluated by checking all scenarios.

```
CC-CLARKE-WRIGHT(dist, demands, Q, α, S):
  routes ← {[j] for j = 1,...,n}        // initial: one route per customer
  savings ← [(dist[0][i]+dist[0][j]-dist[i][j], i, j) for all i<j]
  sort savings descending
  for each (s_val, i, j) in savings:
    R_i, R_j ← routes containing i, j
    if R_i ≠ R_j and i, j are route endpoints:
      R_merged ← R_i ∪ R_j
      // Check chance constraint across all scenarios
      p_overflow ← |{s : Σ_{k∈R_merged} D_k(s) > Q}| / S
      if p_overflow ≤ α:
        merge R_i and R_j
  return routes
```

### Simulated Annealing

Neighborhoods: relocate customer between routes, swap customers across routes, 2-opt within route. Objective includes distance + expected recourse. Overflow probability penalty term for infeasible moves.

---

## 4. Implementations in This Repository

```
stochastic_vrp/
├── instance.py                        # StochasticVRPInstance, StochasticVRPSolution
│                                      #   - route_overflow_probability()
│                                      #   - expected_recourse_cost()
│                                      #   - distance_matrix(), random() factory
├── heuristics/
│   └── chance_constrained_cw.py       # CC Clarke-Wright, mean-demand savings
├── metaheuristics/
│   └── simulated_annealing.py         # Relocate/swap/2-opt with recourse penalty
└── tests/
    └── test_stochastic_vrp.py         # 13 tests, 3 test classes
```

---

## 5. Key References

- Bertsimas, D.J. (1992). A vehicle routing problem with stochastic demand. *Oper. Res.*, 40(3), 574-585. https://doi.org/10.1287/opre.40.3.574
- Gendreau, M., Laporte, G. & Séguin, R. (1996). Stochastic vehicle routing. *European J. Oper. Res.*, 88(1), 3-12. https://doi.org/10.1016/0377-2217(95)00050-X
- Laporte, G., Louveaux, F. & van Hamme, L. (2002). An integer L-shaped algorithm for the CVRP with stochastic demands. *Oper. Res.*, 50(3), 415-423. https://doi.org/10.1287/opre.50.3.415.7751
- Gendreau, M., Jabali, O. & Rei, W. (2016). 50th anniversary invited article — Future directions in stochastic vehicle routing. *Transportation Science*, 50(4), 1163-1173.

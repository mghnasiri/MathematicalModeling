# Chance-Constrained Facility Location (CCFL)

## 1. Problem Definition

- **Input:** $m$ candidate facilities with fixed costs $f_i$ and capacities $C_i$; $n$ customers with assignment costs $c_{ij}$ and stochastic demands $D_j(s)$ across $S$ scenarios; risk level $\alpha$
- **Decision:** Which facilities to open; assignment of customers to open facilities
- **Objective:** Minimize total cost $\sum_i f_i y_i + \sum_{i,j} c_{ij} x_{ij}$
- **Constraints:** Each customer assigned to exactly one open facility; per-facility chance constraint $P(\sum_{j \in S_i} D_j \leq C_i) \geq 1 - \alpha$
- **Classification:** NP-hard (generalizes deterministic UFLP)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $m$ | Number of candidate facility sites |
| $n$ | Number of customers |
| $f_i$ | Fixed cost of opening facility $i$ |
| $c_{ij}$ | Cost to assign customer $j$ to facility $i$ |
| $C_i$ | Capacity of facility $i$ |
| $D_j(s)$ | Demand of customer $j$ under scenario $s$ |
| $\alpha$ | Maximum allowed violation probability |
| $y_i$ | Binary: 1 if facility $i$ is open |
| $x_{ij}$ | Binary: 1 if customer $j$ assigned to facility $i$ |

### MILP with Chance Constraints

$$\min \sum_{i=1}^{m} f_i y_i + \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \tag{2}$$

$$x_{ij} \leq y_i \quad \forall i, j \tag{3}$$

$$P\!\left(\sum_{j: x_{ij}=1} D_j \leq C_i\right) \geq 1 - \alpha \quad \forall i \tag{4}$$

$$y_i \in \{0,1\}, \quad x_{ij} \in \{0,1\} \tag{5}$$

### Small Illustrative Instance

```
m = 2 facilities, n = 3 customers, α = 0.1
Fixed costs: [100, 120], Capacities: [50, 60]
Assignment costs: [[5, 8, 6], [7, 4, 9]]
Scenarios (p=0.5 each):
  s1: demands = [15, 20, 10]  (total = 45)
  s2: demands = [25, 30, 15]  (total = 70)

Open both → assign {1,3} to fac 0, {2} to fac 1:
  Fac 0: P(15+10 ≤ 50) = P(25+15 ≤ 50) = 1.0 ✓
  Fac 1: P(20 ≤ 60) = P(30 ≤ 60) = 1.0 ✓
  Cost = 100 + 120 + 5 + 6 + 4 = 235
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy Open | Heuristic | $O(m^2 \cdot n \cdot S)$ | Iteratively open cost-reducing facilities with CC checks |
| Mean-Demand Greedy | Heuristic | $O(m \cdot n)$ | Deterministic proxy using expected demands |
| Simulated Annealing | Metaheuristic | $O(I \cdot n \cdot S)$ | Toggle/swap facilities with violation penalty |

### Greedy Open

Iteratively open the facility that most reduces total cost while maintaining chance-constraint feasibility for all assignments. Each step evaluates the full scenario matrix.

```
GREEDY-OPEN-CCFL(f, c, C, D, α, S):
  open ← ∅
  while unassigned customers exist:
    best_i ← argmin_i { f[i] + assignment_cost(i) }
      such that opening i maintains:
        P(Σ_{j assigned to i} D_j(s) ≤ C[i]) ≥ 1-α  for all s∈S
    open ← open ∪ {best_i}
    assign nearest unassigned customers to best_i (respecting CC)
  return open, assignments
```

### Simulated Annealing

Toggle facility open/close or swap customer assignments. Infeasible solutions penalized by $\lambda \cdot \sum_i \max(0, P(\text{violation}_i) - \alpha)$.

---

## 4. Implementations in This Repository

```
chance_constrained_fl/
├── instance.py                    # CCFLInstance, CCFLSolution
│                                  #   - capacity_violation_prob(), is_feasible()
│                                  #   - random() factory
├── heuristics/
│   └── greedy_ccfl.py             # Greedy open, mean-demand greedy
├── metaheuristics/
│   └── simulated_annealing.py     # Toggle/swap SA with violation penalty
└── tests/
    └── test_ccfl.py               # 11 tests, 3 test classes
```

---

## 5. Key References

- Bertsimas, D. & Sim, M. (2004). The price of robustness. *Oper. Res.*, 52(1), 35-53. https://doi.org/10.1287/opre.1030.0065
- Snyder, L.V. (2006). Facility location under uncertainty: a review. *IIE Transactions*, 38(7), 547-564. https://doi.org/10.1080/07408170500216480
- Luedtke, J. & Ahmed, S. (2008). A sample approximation approach for optimization with probabilistic constraints. *SIAM J. Optim.*, 19(2), 674-699.

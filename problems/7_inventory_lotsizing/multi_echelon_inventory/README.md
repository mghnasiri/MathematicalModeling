# Multi-Echelon Inventory Optimization

## 1. Problem Definition

- **Input:** Serial supply chain with $L$ echelons; per-echelon holding costs $h_\ell$, ordering costs $K_\ell$, lead times $\tau_\ell$; demand distribution (mean $\mu_D$, std $\sigma_D$); target service level $\text{SL}$
- **Decision:** Base-stock levels $S_\ell$ (or reorder intervals) at each echelon
- **Objective:** Minimize total expected holding cost across all echelons while meeting the target service level at the customer-facing echelon
- **Classification:** General networks are NP-hard; serial systems solvable via Clark-Scarf decomposition

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $L$ | Number of echelons (1 = customer-facing, $L$ = most upstream) |
| $h_\ell$ | Holding cost per unit per period at echelon $\ell$ |
| $K_\ell$ | Fixed ordering cost at echelon $\ell$ |
| $\tau_\ell$ | Lead time at echelon $\ell$ |
| $\mu_D, \sigma_D$ | Mean and std of demand per period |
| $\text{SL}$ | Target service level (fill rate or cycle service level) |
| $S_\ell$ | Base-stock level at echelon $\ell$ |

### Echelon Base-Stock Policy

The cumulative lead time from echelon $\ell$ to the customer:

$$L_\ell = \sum_{k=1}^{\ell} \tau_k$$

Demand during cumulative lead time: $\mu_{L_\ell} = \mu_D \cdot L_\ell$, $\sigma_{L_\ell} = \sigma_D \sqrt{L_\ell}$

Echelon base-stock level:

$$S_\ell = \mu_D \cdot L_\ell + z_{\text{SL}} \cdot \sigma_D \sqrt{L_\ell}$$

where $z_{\text{SL}} = \Phi^{-1}(\text{SL})$.

### Small Illustrative Instance

```
L = 3 echelons, lead times = [1, 2, 3] periods
μ_D = 100, σ_D = 20, SL = 0.95 → z = 1.645
Cumulative lead times: L₁=1, L₂=3, L₃=6
S₁ = 100·1 + 1.645·20·√1 = 132.9
S₂ = 100·3 + 1.645·20·√3 = 356.9
S₃ = 100·6 + 1.645·20·√6 = 680.6
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Echelon base-stock | Heuristic | $O(L)$ | Set base-stock per echelon based on cumulative lead time |
| Powers-of-two | Heuristic | $O(L \log R)$ | Reorder intervals as powers of 2; ≤ 2% above optimal (Roundy, 1985) |
| Greedy allocation | Heuristic | $O(L \cdot n)$ | Allocate safety stock greedily across echelons |

### Powers-of-Two Policy

Sets reorder intervals $T_\ell = 2^{k_\ell}$ for integer $k_\ell$, ensuring a nested schedule. Roundy (1985) proved this achieves at most 2% above the optimal cost for one-warehouse multi-retailer systems.

---

## 4. Implementations in This Repository

```
multi_echelon_inventory/
├── instance.py                    # MultiEchelonInstance, MultiEchelonSolution
│                                  #   - Fields: L, holding_costs, ordering_costs,
│                                  #     lead_times, mean_demand, std_demand, service_level
├── heuristics/
│   ├── base_stock.py              # Echelon base-stock, powers-of-two policies
│   └── greedy_allocation.py       # Greedy safety stock allocation
└── tests/
    └── test_multi_echelon.py      # Multi-echelon test suite
```

---

## 5. Key References

- Clark, A.J. & Scarf, H. (1960). Optimal policies for a multi-echelon inventory problem. *Management Science*, 6(4), 475-490. https://doi.org/10.1287/mnsc.6.4.475
- Roundy, R. (1985). 98%-effective integer-ratio lot-sizing for one-warehouse multi-retailer systems. *Management Science*, 31(11), 1416-1430. https://doi.org/10.1287/mnsc.31.11.1416
- Axsater, S. (2006). *Inventory Control*. 2nd ed. Springer.
- Graves, S.C. & Willems, S.P. (2000). Optimizing strategic safety stock placement in supply chains. *Manufacturing & Service Oper. Mgmt.*, 2(1), 68-83.

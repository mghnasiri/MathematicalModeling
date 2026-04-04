# Capacitated Lot Sizing Problem (CLSP)

## 1. Problem Definition

- **Input:** Planning horizon $T$ periods, demands $d_t$, production capacities $C_t$, fixed setup costs $K_t$, variable production costs $v_t$, holding costs $h_t$
- **Decision:** Production quantities $x_t$ and setup indicators $y_t$ for each period
- **Objective:** Minimize total setup + production + holding cost
- **Constraints:** Flow balance, capacity linking ($x_t \leq C_t y_t$), demand satisfaction
- **Classification:** NP-hard even for constant costs and capacities (Florian, Lenstra & Rinnooy Kan, 1980)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $T$ | Number of periods |
| $d_t$ | Demand in period $t$ |
| $C_t$ | Production capacity in period $t$ |
| $K_t$ | Fixed setup cost in period $t$ |
| $v_t$ | Variable production cost per unit in period $t$ |
| $h_t$ | Holding cost per unit in period $t$ |
| $x_t$ | Production quantity in period $t$ |
| $I_t$ | Ending inventory in period $t$ |
| $y_t$ | Setup indicator: 1 if production occurs in $t$ |

### MILP Formulation

$$\min \sum_{t=1}^{T} \left[ K_t y_t + v_t x_t + h_t I_t \right] \tag{1}$$

$$I_{t-1} + x_t = d_t + I_t \quad \forall t \tag{2}$$

$$x_t \leq C_t \cdot y_t \quad \forall t \tag{3}$$

$$I_0 = 0 \tag{4}$$

$$x_t, I_t \geq 0, \quad y_t \in \{0, 1\} \tag{5}$$

### Small Illustrative Instance

```
T = 4, demands = [30, 40, 20, 50], capacities = [60, 60, 60, 60]
K = [100, 100, 100, 100], v = [2, 2, 2, 2], h = [1, 1, 1, 1]

Feasible solution: produce in periods 1 and 3
  x = [70, 0, 70, 0], I = [40, 0, 50, 0]
  Cost = 100 + 140 + 40 + 100 + 140 + 50 = 570

MIP optimal may differ depending on capacity utilization.
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| MIP (HiGHS) | Exact | Exponential worst | MILP via `scipy.optimize.milp` |
| Greedy lot-for-lot | Heuristic | $O(T)$ | Produce exactly $d_t$ each period |
| Greedy forward | Heuristic | $O(T^2)$ | Greedily extend production to future periods |

### MIP Formulation

Standard MILP with $3T$ variables ($x_t, I_t, y_t$) and $3T$ constraints. Solved via SciPy HiGHS.

### Greedy Heuristics

Two approaches: (1) lot-for-lot produces exactly demand each period -- no holding cost but maximum setups; (2) greedy forward extends production to cover future demands up to capacity, reducing setup frequency.

```
LOT-SHIFTING(T, d, C, K, h):
  // Phase 1: Lot-for-lot baseline
  x[t] ← d[t] for all t; y[t] ← 1 if d[t] > 0
  // Phase 2: Shift production to eliminate setups
  for t ← T downto 1:
    if y[t] = 1 and x[t] > 0:
      // Try shifting x[t] to an earlier period with spare capacity
      for t' ← t-1 downto 1:
        shift ← min(x[t], C[t'] - x[t'])
        if K[t] > holding_cost(shift, t', t):
          x[t'] += shift; x[t] -= shift
          if x[t] = 0: y[t] ← 0; break
  return x, y
```

---

## 4. Implementations in This Repository

```
capacitated_lot_sizing/
├── instance.py                    # CapLotSizingInstance, CapLotSizingSolution
│                                  #   - Fields: T, demands, capacities, fixed_costs,
│                                  #     variable_costs, holding_costs
├── exact/
│   └── mip_clsp.py                # MILP formulation via SciPy HiGHS
├── heuristics/
│   ├── greedy_cls.py              # Greedy heuristics for CLSP
│   └── greedy_clsp.py             # Additional greedy heuristic
└── tests/
    ├── test_cap_lot_sizing.py     # CLSP test suite
    ├── test_clsp.py               # Additional CLSP tests
    └── test_mip_clsp.py           # MIP solver tests
```

---

## 5. Key References

- Florian, M., Lenstra, J.K. & Rinnooy Kan, A.H.G. (1980). Deterministic production planning: Algorithms and complexity. *Management Science*, 26(7), 669-679. https://doi.org/10.1287/mnsc.26.7.669
- Pochet, Y. & Wolsey, L.A. (2006). *Production Planning by Mixed Integer Programming*. Springer. https://doi.org/10.1007/0-387-33477-7
- Barany, I., Van Roy, T.J. & Wolsey, L.A. (1984). Strong formulations for multi-item capacitated lot sizing. *Management Science*, 30(10), 1255-1261. https://doi.org/10.1287/mnsc.30.10.1255

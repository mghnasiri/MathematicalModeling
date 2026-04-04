# Dynamic Lot Sizing (Uncapacitated)

## 1. Problem Definition

- **Input:** Planning horizon $T$ periods, time-varying demands $d_t$, ordering costs $K_t$, holding costs $h_t$
- **Decision:** Order quantities $x_t \geq 0$ for each period
- **Objective:** Minimize total ordering + holding cost $\sum_t [K_t \cdot \delta(x_t) + h_t \cdot I_t]$ where $\delta(x_t) = 1$ if $x_t > 0$
- **Constraints:** Demand must be met in each period; no backorders; $I_0 = 0$
- **Classification:** $O(T^2)$ via Wagner-Whitin DP. Zero Inventory Ordering (ZIO) property holds at optimality.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $T$ | Number of periods |
| $d_t$ | Demand in period $t$ |
| $K_t$ | Fixed ordering cost in period $t$ |
| $h_t$ | Holding cost per unit per period $t$ |
| $x_t$ | Order quantity in period $t$ |
| $I_t$ | Ending inventory in period $t$ |

### DP Formulation

$$I_t = I_{t-1} + x_t - d_t \geq 0 \quad \forall t \tag{1}$$

$$\min \sum_{t=1}^{T} \left[ K_t \cdot \mathbb{1}(x_t > 0) + h_t \cdot I_t \right] \tag{2}$$

By the ZIO property, optimal orders cover consecutive periods $[i, j]$. Define $C(i, j)$ = cost of ordering in period $i$ to cover demands $d_i, \ldots, d_j$:

$$C(i, j) = K_i + \sum_{k=i+1}^{j} h_{k-1} \cdot \sum_{\ell=k}^{j} d_\ell$$

The DP: $f(t)$ = minimum cost to cover periods $1, \ldots, t$:

$$f(t) = \min_{1 \leq i \leq t} \bigl[ f(i-1) + C(i, t) \bigr], \quad f(0) = 0$$

### Small Illustrative Instance

```
T = 4, demands = [20, 30, 10, 40]
K = [100, 100, 100, 100], h = [1, 1, 1, 1]

Option A: Order every period → 4×100 = 400 ordering, 0 holding = 400
Option B: Order in periods 1,3 → x1=50, x3=50
  Holding: 30×1 + 0 + 40×1 = 70, Ordering: 200 → Total = 270
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Wagner-Whitin DP | Exact | $O(T^2)$ | Forward DP exploiting ZIO property |
| Silver-Meal | Heuristic | $O(T^2)$ worst | Greedy: extend order horizon until avg cost/period rises |
| Part-Period Balancing | Heuristic | $O(T^2)$ worst | Order until cumulative holding $\approx$ ordering cost |

### Silver-Meal Heuristic

Starting from first uncovered period, extend the order to cover additional periods as long as average cost per period decreases. When it increases, place the order and start a new one.

```
SILVER-MEAL(T, d, K, h):
  orders ← []; t ← 1
  while t ≤ T:
    order_start ← t; cum_hold ← 0; best_avg ← ∞
    for j ← t to T:
      cum_hold ← cum_hold + h · (j - t) · d[j]
      avg_cost ← (K + cum_hold) / (j - t + 1)
      if avg_cost > best_avg: break     // avg cost rising → stop
      best_avg ← avg_cost; t_end ← j
    orders ← orders ∪ {(order_start, Σ d[order_start..t_end])}
    t ← t_end + 1
  return orders
```

### Part-Period Balancing (PPB)

Order enough to cover periods until the cumulative holding cost approximately equals the ordering cost. Based on the EOQ insight that total cost is minimized when ordering and holding costs are balanced.

---

## 4. Implementations in This Repository

```
lot_sizing/
├── instance.py                    # LotSizingInstance, LotSizingSolution
│                                  #   - Fields: T, demands, ordering_costs, holding_costs
├── exact/
│   └── wagner_whitin.py           # Wagner-Whitin DP, O(T²)
├── heuristics/
│   └── silver_meal.py             # Silver-Meal, Part-Period Balancing
└── tests/
    └── test_lot_sizing.py         # Lot sizing test suite
```

---

## 5. Key References

- Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic lot size model. *Management Science*, 5(1), 89-96. https://doi.org/10.1287/mnsc.5.1.89
- Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size quantities for the case of a deterministic time-varying demand rate. *Prod. & Inv. Mgmt.*, 14(2), 64-74.
- DeMatteis, J.J. (1968). An economic lot-sizing technique I: The part-period algorithm. *IBM Systems Journal*, 7(1), 30-38. https://doi.org/10.1147/sj.71.0030

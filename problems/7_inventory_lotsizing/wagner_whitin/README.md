# Wagner-Whitin Lot Sizing Problem

## 1. Problem Definition

- **Input:** Planning horizon $T$ periods, demands $d_t$, ordering costs $K_t$, holding costs $h_t$
- **Decision:** When and how much to order in each period
- **Objective:** Minimize total ordering + holding cost over the planning horizon
- **Constraints:** All demand met on time; no backorders; zero initial inventory
- **Classification:** $O(T^2)$ via dynamic programming. The **Zero Inventory Ordering (ZIO)** property guarantees an optimal solution where orders are placed only when inventory reaches zero.

### Wagner-Whitin Property (ZIO)

At optimality, $I_{t-1} \cdot x_t = 0$ for all $t$: either inventory entering a period is zero (so we order) or production is zero (we carry inventory). This reduces the search space from continuous to $2^T$ binary decisions, further exploitable by DP.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $T$ | Number of periods |
| $d_t$ | Demand in period $t$ ($t = 1, \ldots, T$) |
| $K_t$ | Fixed ordering cost in period $t$ |
| $h_t$ | Holding cost per unit per period in period $t$ |
| $x_t$ | Order quantity in period $t$ |
| $I_t$ | Ending inventory in period $t$ |

### DP Recurrence

$$f(j) = \min_{1 \leq i \leq j} \left\{ f(i-1) + K_i + \sum_{k=i}^{j-1} h_k \sum_{\ell=k+1}^{j} d_\ell \right\}, \quad f(0) = 0$$

$f(j)$ = minimum cost to satisfy demands in periods $1, \ldots, j$.

### Small Illustrative Instance

```
T = 5, d = [10, 20, 15, 25, 10]
K = 50 (constant), h = 1 (constant)

Optimal: Order in periods 1 and 4
  Period 1: x₁ = 10+20+15 = 45 (covers t=1,2,3)
  Period 4: x₄ = 25+10 = 35 (covers t=4,5)
  Holding: 20×1 + 15×2 + 10×1 = 60
  Total: 50 + 60 + 50 + 0 = 160
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Wagner-Whitin DP | Exact | $O(T^2)$ | Forward DP using ZIO property |

### Wagner-Whitin DP

Evaluates all $O(T^2)$ subproblems: for each period $j$, find the best period $i$ to place the last order covering periods $i$ through $j$. The ZIO property ensures only "order from scratch" decisions are considered.

```
WAGNER-WHITIN-DP(T, d, K, h):
  f[0] ← 0
  pred[0] ← -1
  for j ← 1 to T:
    f[j] ← ∞
    for i ← 1 to j:
      cost ← f[i-1] + K[i] + Σ_{k=i}^{j-1} h[k] · Σ_{ℓ=k+1}^{j} d[ℓ]
      if cost < f[j]:
        f[j] ← cost
        pred[j] ← i
  return f[T], backtrack(pred)
```

---

## 4. Implementations in This Repository

```
wagner_whitin/
├── instance.py                    # WagnerWhitinInstance, WagnerWhitinSolution
│                                  #   - Fields: T, demands, ordering_costs, holding_costs
├── exact/
│   └── wagner_whitin_dp.py        # Wagner-Whitin DP, O(T²)
└── tests/
    └── test_wagner_whitin.py      # Wagner-Whitin test suite
```

---

## 5. Key References

- Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic lot size model. *Management Science*, 5(1), 89-96. https://doi.org/10.1287/mnsc.5.1.89
- Aggarwal, A. & Park, J.K. (1993). Improved algorithms for economic lot size problems. *Oper. Res.*, 41(3), 549-571. https://doi.org/10.1287/opre.41.3.549
- Pochet, Y. & Wolsey, L.A. (2006). *Production Planning by Mixed Integer Programming*. Springer.

---

## 6. Notes

- The Aggarwal-Park (1993) algorithm reduces Wagner-Whitin to $O(T)$ using the SMAWK technique for concave cost structures.
- The Wagner-Whitin model assumes deterministic demand, zero lead time, and no capacity constraints.
- Federgruen, A. & Tzur, M. (1991). A simple forward algorithm to solve general dynamic lot sizing models with $n$ periods in $O(n \log n)$ or $O(n)$ time. *Management Science*, 37(8), 909-925.

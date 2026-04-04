# Vehicle/Container Loading Problem

## 1. Problem Definition

- **Input:** $n$ items with weights $w_i$ and volumes $v_i$; vehicles with weight capacity $W$ and volume capacity $V$
- **Decision:** Assign each item to a vehicle
- **Objective:** Minimize the number of vehicles used
- **Constraints:** Total weight and total volume in each vehicle must not exceed $W$ and $V$ respectively
- **Classification:** NP-hard (generalizes 1D bin packing with dual capacity constraints)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of items |
| $w_i$ | Weight of item $i$ |
| $v_i$ | Volume of item $i$ |
| $W$ | Vehicle weight capacity |
| $V$ | Vehicle volume capacity |
| $K$ | Upper bound on number of vehicles |

### MILP Formulation

$$\min \sum_{k=1}^{K} y_k \tag{1}$$

$$\sum_{k=1}^{K} x_{ik} = 1 \quad \forall i \tag{2}$$

$$\sum_{i=1}^{n} w_i x_{ik} \leq W \cdot y_k \quad \forall k \tag{3}$$

$$\sum_{i=1}^{n} v_i x_{ik} \leq V \cdot y_k \quad \forall k \tag{4}$$

$$x_{ik} \in \{0,1\}, \quad y_k \in \{0,1\} \tag{5}$$

### Small Illustrative Instance

```
n = 5 items
Weights:  [30, 40, 20, 50, 10], W = 80
Volumes:  [3,  2,  5,  4,  6],  V = 10

Vehicle 1: items {1, 4} → w=70, v=7  ✓
Vehicle 2: items {2, 5} → w=50, v=8  ✓
Vehicle 3: items {3}    → w=20, v=5  ✓
Total: 3 vehicles
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| First-Fit Decreasing | Heuristic | $O(n^2)$ | Sort items, assign to first feasible vehicle |

### First-Fit Decreasing (FFD)

Sort items by primary dimension (weight) descending, then assign each item to the first vehicle that has sufficient remaining weight and volume capacity. If no vehicle fits, open a new one. Extends the classic FFD for bin packing to handle dual constraints.

```
FFD-DUAL-CAPACITY(items, W, V):
  sort items by weight descending
  vehicles ← []
  for each item (w_i, v_i):
    placed ← false
    for each vehicle k in vehicles:
      if rem_weight[k] ≥ w_i and rem_vol[k] ≥ v_i:
        assign item to vehicle k
        rem_weight[k] -= w_i; rem_vol[k] -= v_i
        placed ← true; break
    if not placed:
      open new vehicle with (W - w_i, V - v_i)
  return vehicles
```

---

## 4. Implementations in This Repository

```
vehicle_loading/
├── instance.py                    # VehicleLoadingInstance, VehicleLoadingSolution
│                                  #   - Fields: n_items, weights, volumes,
│                                  #     weight_capacity, volume_capacity
│                                  #   - random() factory
├── heuristics/
│   └── greedy_ffd.py              # FFD heuristic for dual-capacity bin packing
└── tests/
    └── test_vehicle_loading.py    # Vehicle loading test suite
```

---

## 5. Key References

- Christensen, H.I., Khan, A., Pokutta, S. & Tetali, P. (2017). Approximation and online algorithms for multidimensional bin packing: A survey. *Computer Science Review*, 24, 63-79.
- Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation algorithms for bin packing: A survey. In *Approximation Algorithms for NP-hard Problems* (pp. 46-93). PWS Publishing.

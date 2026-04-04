# Dial-a-Ride Problem (DARP)

## 1. Problem Definition

- **Input:** $m$ vehicles with capacity, $n$ ride requests each with pickup/delivery locations and time windows, maximum ride time constraint
- **Decision:** Assign requests to vehicles and sequence pickup-delivery pairs
- **Objective:** Minimize total travel distance
- **Constraints:** Pickup before delivery; vehicle capacity; time windows; max ride time
- **Classification:** NP-hard (generalizes VRP with Pickup and Delivery)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Insertion DARP | Heuristic | $O(n^2 \cdot m)$ | Iteratively insert requests into cheapest feasible position |

---

## 3. Implementations in This Repository

```
dial_a_ride/
├── instance.py                    # DARPInstance, DARPSolution
├── heuristics/
│   └── insertion_darp.py          # Insertion heuristic for DARP
└── tests/
    └── test_darp.py               # DARP test suite
```

---

## 4. Key References

- Cordeau, J.-F. & Laporte, G. (2007). The dial-a-ride problem: models and algorithms. *Annals Oper. Res.*, 153(1), 29-46. https://doi.org/10.1007/s10479-007-0170-8
- Jaw, J.J., Odoni, A.R., Psaraftis, H.N. & Wilson, N.H.M. (1986). A heuristic algorithm for the multi-vehicle advance request dial-a-ride problem. *Transp. Res. B*, 20(3), 243-257.

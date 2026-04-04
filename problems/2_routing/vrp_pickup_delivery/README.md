# VRP with Pickup and Delivery (VRPPD)

## 1. Problem Definition

- **Input:** $m$ vehicles with capacity, $n$ requests each with pickup and delivery locations, distance matrix
- **Decision:** Assign requests to vehicles and sequence visits
- **Objective:** Minimize total travel distance
- **Constraints:** Pickup visited before delivery; vehicle load never exceeds capacity; all requests served
- **Classification:** NP-hard (generalizes CVRP)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Insertion VRPPD | Heuristic | $O(n^2 \cdot m)$ | Insert pickup-delivery pairs in cheapest feasible position |

---

## 3. Implementations in This Repository

```
vrp_pickup_delivery/
├── instance.py                    # VRPPDInstance, VRPPDSolution
├── heuristics/
│   └── insertion_vrppd.py         # Insertion heuristic for VRPPD
└── tests/
    └── test_vrppd.py              # VRPPD test suite
```

---

## 4. Key References

- Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery problem. *Transp. Sci.*, 29(1), 17-29. https://doi.org/10.1287/trsc.29.1.17
- Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transp. Sci.*, 40(4), 455-472.

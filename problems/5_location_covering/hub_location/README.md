# Hub Location Problem (p-Hub Median)

## 1. Problem Definition

- **Input:** $n$ nodes, flow matrix $W_{ij}$, distance matrix $D_{ij}$, number of hubs $p$, inter-hub discount factor $\alpha$
- **Decision:** Select $p$ hub nodes; assign each non-hub to a hub (single allocation)
- **Objective:** Minimize total transportation cost (flows routed: origin hub → destination hub with discount $\alpha$)
- **Classification:** NP-hard (O'Kelly, 1987)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy hub | Heuristic | $O(n^2 \cdot p)$ | Greedily open hubs minimizing total cost |

---

## 3. Implementations in This Repository

```
hub_location/
├── instance.py                    # HubLocationInstance, HubLocationSolution
├── heuristics/
│   └── greedy_hub.py              # Greedy hub selection
└── tests/
    └── test_hub_location.py       # Hub location test suite
```

---

## 4. Key References

- O'Kelly, M.E. (1987). A quadratic integer program for the location of interacting hub facilities. *European J. Oper. Res.*, 32(3), 393-404.

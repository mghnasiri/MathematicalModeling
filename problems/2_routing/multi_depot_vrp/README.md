# Multi-Depot Vehicle Routing Problem (MDVRP)

## 1. Problem Definition

- **Input:** $m$ depots each with a vehicle fleet, $n$ customers with demands, vehicle capacity $Q$, distance matrix
- **Decision:** Assign customers to depots and construct routes from each depot
- **Objective:** Minimize total travel distance
- **Constraints:** Each customer visited exactly once; vehicle capacity respected; routes start/end at their depot
- **Classification:** NP-hard (generalizes CVRP)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Nearest depot | Heuristic | $O(n \cdot m)$ | Assign each customer to nearest depot, then build routes |

---

## 3. Implementations in This Repository

```
multi_depot_vrp/
├── instance.py                        # MDVRPInstance, MDVRPSolution
├── heuristics/
│   └── nearest_depot.py              # Nearest depot assignment + routing
└── tests/
    └── test_multi_depot_vrp.py        # MDVRP test suite
```

---

## 4. Key References

- Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search heuristic for periodic and multi-depot VRP. *Networks*, 30(2), 105-119.
- Montoya-Torres, J.R. et al. (2015). A literature review on the VRP with multiple depots. *Computers & Industrial Engineering*, 79, 115-129.

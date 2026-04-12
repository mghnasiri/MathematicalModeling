# Multi-Depot Vehicle Routing Problem (MDVRP)

## 1. Problem Definition

- **Input:** $m$ depots each with a vehicle fleet, $n$ customers with demands, vehicle capacity $Q$, distance matrix
- **Decision:** Assign customers to depots and construct routes from each depot
- **Objective:** Minimize total travel distance
- **Constraints:** Each customer visited exactly once; vehicle capacity respected; routes start/end at their depot
- **Classification:** NP-hard (generalizes CVRP)

### Problem Variants

- **Heterogeneous fleet MDVRP:** Vehicles at each depot may have different capacities.
- **MDVRP with time windows:** Customer visits must occur within specified time intervals.
- **Open MDVRP:** Vehicles do not need to return to their depot after the last delivery.

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Nearest depot | Heuristic | $O(n \cdot m)$ | Assign each customer to nearest depot, then build routes |
| Cluster-first route-second | Heuristic | $O(n \cdot m + n^2)$ | Partition customers into depot clusters, solve CVRP per cluster |
| Tabu search | Metaheuristic | problem-dependent | Inter-depot and intra-route moves with short-term memory |

### Nearest-Depot Assignment Pseudocode

```
NEAREST_DEPOT_MDVRP(depots, customers, demands, capacity):
    // Phase 1: Assign customers to nearest depot
    clusters = {d: [] for d in depots}
    for each customer c:
        d* = argmin_{d in depots} distance(c, d)
        clusters[d*].append(c)

    // Phase 2: Build routes per depot using savings heuristic
    all_routes = []
    for each depot d:
        // Initialize: one route per customer
        routes = [[d, c, d] for c in clusters[d]]
        // Compute savings
        savings = []
        for each pair (i, j) in clusters[d]:
            s_ij = dist(d, i) + dist(d, j) - dist(i, j)
            savings.append((s_ij, i, j))
        sort savings descending
        // Merge routes by savings
        for each (s_ij, i, j) in savings:
            route_i = route containing i (as last customer)
            route_j = route containing j (as first customer)
            if route_i != route_j AND load(route_i) + load(route_j) <= capacity:
                merge route_i and route_j
        all_routes.extend(routes for depot d)
    return all_routes
```

---

## 3. Illustrative Instance

2 depots, 5 customers, vehicle capacity Q = 10:

| Node | Type | Location | Demand |
|------|------|----------|--------|
| D1 | Depot | (0, 0) | - |
| D2 | Depot | (10, 0) | - |
| C1 | Customer | (1, 2) | 3 |
| C2 | Customer | (3, 1) | 4 |
| C3 | Customer | (8, 1) | 5 |
| C4 | Customer | (9, 3) | 3 |
| C5 | Customer | (2, 3) | 2 |

Assignment: C1, C2, C5 -> D1 (nearest). C3, C4 -> D2 (nearest). D1 routes: [D1, C5, C1, C2, D1] (load=9). D2 routes: [D2, C3, C4, D2] (load=8). Total: 2 routes.

---

## 4. Applications

- **Grocery delivery:** Multiple distribution centers serve overlapping delivery zones; assigning customers to the nearest warehouse reduces transit time.
- **Courier services:** Regional hubs dispatch fleets of vans; customers on depot boundaries may be served by either hub.
- **Emergency services:** Fire stations or ambulance depots positioned across a city; each depot covers a service region.
- **Beverage distribution:** Breweries or bottling plants at different locations supply retailers through dedicated vehicle fleets.

---

## 5. Implementations in This Repository

```
multi_depot_vrp/
├── instance.py                        # MDVRPInstance, MDVRPSolution
├── heuristics/
│   └── nearest_depot.py              # Nearest depot assignment + routing
└── tests/
    └── test_multi_depot_vrp.py        # MDVRP test suite
```

---

## 6. Key References

- Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search heuristic for periodic and multi-depot VRP. *Networks*, 30(2), 105-119.
- Montoya-Torres, J.R. et al. (2015). A literature review on the VRP with multiple depots. *Computers & Industrial Engineering*, 79, 115-129.
- Renaud, J., Laporte, G. & Boctor, F.F. (1996). A tabu search heuristic for the multi-depot vehicle routing problem. *Computers & Oper. Res.*, 23(3), 229-235.

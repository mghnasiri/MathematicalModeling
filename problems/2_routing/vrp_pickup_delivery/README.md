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
| ALNS | Metaheuristic | problem-dependent | Adaptive Large Neighborhood Search with destroy/repair operators |
| Branch-and-cut | Exact | exponential | Integer programming with precedence and pairing cuts |

### Paired Insertion Heuristic Pseudocode

```
INSERTION_VRPPD(requests, vehicles, capacity):
    routes = {v: [depot, depot] for v in vehicles}
    unassigned = sort requests by distance(pickup, delivery) descending

    for each request r = (pickup_r, delivery_r, load_r) in unassigned:
        best_cost = infinity
        best_vehicle = None
        best_pos_p = None
        best_pos_d = None

        for each vehicle v:
            for each position i in route[v]:       // pickup position
                for each position j in {i+1 .. len(route[v])}:  // delivery after pickup
                    new_route = insert pickup_r at i, delivery_r at j
                    // Check precedence: pickup before delivery
                    // Check capacity: cumulative load never exceeds Q
                    loads_ok = True
                    running_load = 0
                    for each stop s in new_route:
                        if s is a pickup: running_load += load(s)
                        if s is a delivery: running_load -= load(s)
                        if running_load > capacity: loads_ok = False; break
                    if loads_ok:
                        delta = cost(new_route) - cost(route[v])
                        if delta < best_cost:
                            best_cost = delta
                            best_vehicle, best_pos_p, best_pos_d = v, i, j

        if best_vehicle is not None:
            apply insertion to route[best_vehicle]
    return routes
```

---

## 3. Illustrative Instance

1 vehicle, capacity Q = 5, 3 pickup-delivery requests:

| Request | Pickup Loc | Delivery Loc | Load |
|---------|-----------|-------------|------|
| R1 | (1, 0) | (4, 0) | 2 |
| R2 | (0, 2) | (3, 2) | 3 |
| R3 | (2, 1) | (5, 1) | 1 |

Depot at (0, 0). Insertion order (by distance): R2 (dist=3), R1 (dist=3), R3 (dist=3). After inserting all three: Route = [depot, P2, P1, D1, P3, D2, D3, depot]. Max load at any point: after P2+P1 = 5 (feasible). Total distance minimized by insertion position search.

---

## 4. Implementations in This Repository

```
vrp_pickup_delivery/
├── instance.py                    # VRPPDInstance, VRPPDSolution
├── heuristics/
│   └── insertion_vrppd.py         # Insertion heuristic for VRPPD
└── tests/
    └── test_vrppd.py              # VRPPD test suite
```

---

## 5. Key References

- Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery problem. *Transp. Sci.*, 29(1), 17-29. https://doi.org/10.1287/trsc.29.1.17
- Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transp. Sci.*, 40(4), 455-472.
- Parragh, S.N., Doerner, K.F. & Hartl, R.F. (2008). A survey on pickup and delivery problems. Part I: Transportation between a depot and many customers. *J. Betriebswirtschaft*, 58(1), 21-51.

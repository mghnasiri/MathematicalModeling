# Dial-a-Ride Problem (DARP)

## 1. Problem Definition

- **Input:** $m$ vehicles with capacity, $n$ ride requests each with pickup/delivery locations and time windows, maximum ride time constraint
- **Decision:** Assign requests to vehicles and sequence pickup-delivery pairs
- **Objective:** Minimize total travel distance
- **Constraints:** Pickup before delivery; vehicle capacity; time windows; max ride time
- **Classification:** NP-hard (generalizes VRP with Pickup and Delivery)

### Key Constraints

- **Pairing:** Each pickup $i^+$ must be visited before its corresponding delivery $i^-$ on the same route.
- **Maximum ride time:** The elapsed time from pickup to delivery for any request must not exceed $L$.
- **Time windows:** Both pickup and delivery locations have earliest/latest service times.

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Insertion DARP | Heuristic | $O(n^2 \cdot m)$ | Iteratively insert requests into cheapest feasible position |
| Cluster-first | Heuristic | $O(n \cdot m)$ | Group requests geographically, then route each cluster |
| Tabu search | Metaheuristic | problem-dependent | Request relocation with aspiration and diversification |

### Insertion Heuristic Pseudocode

```
INSERTION_DARP(requests, vehicles, max_ride_time):
    routes = {v: [depot, depot] for v in vehicles}
    unassigned = sort requests by earliest pickup time

    for each request r = (pickup_r, delivery_r) in unassigned:
        best_cost = infinity
        best_vehicle = None
        best_pos_p = None
        best_pos_d = None

        for each vehicle v:
            for each position i in route[v] (for pickup):
                for each position j >= i+1 in route[v] (for delivery):
                    // Try inserting pickup at i, delivery at j
                    new_route = insert pickup_r at i, delivery_r at j
                    if is_feasible(new_route, capacity[v], time_windows, max_ride_time):
                        delta = cost(new_route) - cost(route[v])
                        if delta < best_cost:
                            best_cost = delta
                            best_vehicle = v
                            best_pos_p = i
                            best_pos_d = j

        if best_vehicle is not None:
            insert pickup_r at best_pos_p in route[best_vehicle]
            insert delivery_r at best_pos_d in route[best_vehicle]
        else:
            mark r as unserved

    return routes
```

---

## 3. Illustrative Instance

2 vehicles (capacity 2), 3 ride requests:

| Request | Pickup | Delivery | Time Window (pickup) | Max Ride |
|---------|--------|----------|---------------------|----------|
| R1 | (0,0) | (3,4) | [0, 10] | 15 |
| R2 | (1,1) | (4,3) | [5, 15] | 12 |
| R3 | (2,0) | (5,1) | [0, 8] | 10 |

Depot at (0,0). Insertion: R1 and R3 assigned to vehicle 1 (both early pickups, nearby). R2 assigned to vehicle 2. Vehicle 1 route: depot -> R1-pickup -> R3-pickup -> R3-delivery -> R1-delivery -> depot.

---

## 4. Applications

- **Paratransit services:** Door-to-door transportation for elderly and disabled passengers with advance booking and maximum ride time constraints.
- **Airport shuttle services:** Shared-ride vans collecting passengers from hotels with specific flight departure times.
- **On-demand ride pooling:** Services like shared taxis or ride-hailing pools where multiple passengers share a vehicle.
- **Medical transportation:** Non-emergency patient transport with appointment time windows and wheelchair capacity constraints.

---

## 5. Implementations in This Repository

```
dial_a_ride/
├── instance.py                    # DARPInstance, DARPSolution
├── heuristics/
│   └── insertion_darp.py          # Insertion heuristic for DARP
└── tests/
    └── test_darp.py               # DARP test suite
```

---

## 6. Key References

- Cordeau, J.-F. & Laporte, G. (2007). The dial-a-ride problem: models and algorithms. *Annals Oper. Res.*, 153(1), 29-46. https://doi.org/10.1007/s10479-007-0170-8
- Jaw, J.J., Odoni, A.R., Psaraftis, H.N. & Wilson, N.H.M. (1986). A heuristic algorithm for the multi-vehicle advance request dial-a-ride problem. *Transp. Res. B*, 20(3), 243-257.
- Parragh, S.N., Doerner, K.F. & Hartl, R.F. (2008). A survey on pickup and delivery problems. Part II: Transportation between pickup and delivery locations. *J. Betriebswirtschaft*, 58(2), 81-117.

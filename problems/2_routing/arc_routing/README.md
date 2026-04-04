# Capacitated Arc Routing Problem (CARP)

## 1. Problem Definition

- **Input:** Undirected graph with required edges (each with demand), depot node, vehicles with capacity $Q$
- **Decision:** Routes starting/ending at depot that traverse all required edges
- **Objective:** Minimize total traversal cost
- **Constraints:** Vehicle capacity respected per route; all required edges served
- **Classification:** NP-hard (Golden & Wong, 1981)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Path scanning | Heuristic | $O(E^2)$ | Extend routes by nearest feasible required edge |
| Augment-merge | Heuristic | $O(E^2)$ | Build single-edge routes, merge by savings criterion |
| Ulusoy's splitting | Heuristic | $O(E \cdot V)$ | Giant tour on edges, optimal split into capacity-feasible routes |

### Path Scanning Pseudocode

```
PATH_SCANNING(graph, depot, required_edges, capacity):
    unserved = set(required_edges)
    routes = []
    while unserved is not empty:
        route = [depot]
        load = 0
        current = depot
        while True:
            candidates = {e in unserved : load + demand(e) <= capacity}
            if candidates is empty:
                break
            e* = argmin_{e in candidates} shortest_path(current, closer_endpoint(e))
                 // tie-break: farthest from depot, or largest demand
            traverse shortest path from current to e*
            route.append(traversal of e*)
            load += demand(e*)
            unserved.remove(e*)
            current = far endpoint of e*
        route.append(path back to depot)
        routes.append(route)
    return routes
```

---

## 3. Illustrative Instance

Graph with 4 nodes and 5 edges (3 required), depot = node 0, capacity Q = 10:

| Edge | Demand | Cost | Required |
|------|--------|------|----------|
| (0,1) | 0 | 2 | No |
| (1,2) | 4 | 3 | Yes |
| (2,3) | 5 | 4 | Yes |
| (0,3) | 3 | 6 | Yes |
| (1,3) | 0 | 2 | No |

Path scanning: Start at depot 0. Nearest required edge from 0: (0,3) at cost 0. Serve (0,3), load=3, current=3. Next nearest: (2,3) at cost 0, serve it, load=8, current=2. Next: (1,2) at cost 0, load=12 > Q=10 -- skip. Return to depot. Route 2: serve (1,2), load=4. Total: 2 routes.

---

## 4. Applications

- **Street sweeping and snow plowing:** Municipal vehicles must traverse every street segment in a district while respecting vehicle capacity (salt/plow time).
- **Postal delivery:** Mail carriers walk or drive every block on their route; minimizing deadheading (traveling non-required edges) saves time and fuel.
- **Utility inspection:** Gas/electric companies inspect every pipeline or power line segment; capacitated by crew shift length.
- **Waste collection:** Garbage trucks service every residential street and must return to the depot when full.

---

## 5. Implementations in This Repository

```
arc_routing/
├── instance.py                    # CARPInstance, CARPSolution
├── heuristics/
│   └── path_scanning.py           # Path scanning heuristic
└── tests/
    └── test_carp.py               # CARP test suite
```

---

## 6. Key References

- Golden, B.L. & Wong, R.T. (1981). Capacitated arc routing problems. *Networks*, 11(3), 305-315.
- Eiselt, H.A., Gendreau, M. & Laporte, G. (1995). Arc routing problems, part I: The Chinese postman problem. *Oper. Res.*, 43(2), 231-242.
- Lacomme, P., Prins, C. & Ramdane-Cherif, W. (2004). Competitive memetic algorithms for arc routing problems. *Annals Oper. Res.*, 131(1-4), 159-185.

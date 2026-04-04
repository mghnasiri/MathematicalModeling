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

---

## 3. Implementations in This Repository

```
arc_routing/
├── instance.py                    # CARPInstance, CARPSolution
├── heuristics/
│   └── path_scanning.py           # Path scanning heuristic
└── tests/
    └── test_carp.py               # CARP test suite
```

---

## 4. Key References

- Golden, B.L. & Wong, R.T. (1981). Capacitated arc routing problems. *Networks*, 11(3), 305-315.

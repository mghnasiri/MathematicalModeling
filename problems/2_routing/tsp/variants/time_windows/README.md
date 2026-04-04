# TSP with Time Windows (TSP Variant)

## What Changes

The standard TSP only minimizes travel distance. The **TSP with Time Windows (TSPTW)** adds a **time dimension**: each city i has a time window [e_i, l_i] during which service must begin. If the salesman arrives before e_i, they wait; arriving after l_i makes the tour infeasible. Service takes s_i time units at each city.

- **Feasibility becomes non-trivial**: many permutations violate time windows, so even finding a feasible tour can be challenging.
- **Waiting time** occurs when arriving early, creating a tension between short distances and good timing.
- **Sequence matters temporally**: the visit order must respect both distance efficiency and time window compatibility.

**Real-world motivation**: delivery scheduling with customer time slots, technician routing with appointment windows, dial-a-ride services, aircraft runway scheduling with approach windows, exam room scheduling.

## Mathematical Formulation

Extends base TSP with temporal constraints:

```
min  sum_{k=0}^{n-1} d(pi(k), pi((k+1) mod n))
s.t. pi is a permutation of {0, ..., n-1}           (Hamiltonian cycle)
     e_i <= t_i <= l_i                for all i      (time windows)
     t_{pi(k+1)} >= t_{pi(k)} + s_{pi(k)} + d(pi(k), pi(k+1))   (travel + service)
     t_i = max(arrival_i, e_i)                       (wait if early)
```

City 0 is the depot with a planning horizon time window [0, H].

## Complexity

- **NP-hard** (generalizes TSP: with infinitely wide windows, TSPTW reduces to TSP).
- Even deciding whether a feasible tour exists is NP-complete for arbitrary time windows.
- Tighter time windows reduce the feasible space but make construction harder.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Nearest Feasible Neighbor | Yes | Visit nearest city whose window can still be met; O(n^2). |
| Earliest Deadline Insertion | Yes | Insert cities by deadline at cheapest feasible position; O(n^3). |
| 2-opt | Limited | Must recheck time window feasibility after each swap. |
| Or-opt (relocate) | Yes | Move a city, recompute arrival times, check feasibility. |
| Simulated Annealing | Yes | Or-opt moves with weighted penalty for time window violations. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `TSPTWInstance` (distance_matrix, time_windows, service_times), `TSPTWSolution`, `tour_schedule()`, `tour_feasible()`, `validate_solution()`, `small_tsptw_5()` benchmark |
| `heuristics.py` | `nearest_feasible()` (TW-aware NN), `earliest_deadline_insertion()` (deadline-sorted insertion) |
| `metaheuristics.py` | `simulated_annealing()` with or-opt moves and weighted infeasibility penalty for crossing TW boundaries |
| `tests/test_tsptw.py` | Test suite covering time window validation, feasibility checking, heuristic and SA quality |

## Key References

- Dumas, Y., Desrosiers, J., Gelinas, E. & Solomon, M.M. (1995). An optimal algorithm for the TSPTW. *Operations Research*, 43(2), 367-371.
- Gendreau, M., Hertz, A., Laporte, G. & Stan, M. (1998). A generalized insertion heuristic for the TSPTW. *Operations Research*, 46(3), 330-335.
- Ohlmann, J.W. & Thomas, B.W. (2007). A compressed-annealing heuristic for the TSPTW. *INFORMS Journal on Computing*, 19(1), 80-90.
- Solomon, M.M. (1987). Algorithms for the VRP and scheduling problems with time window constraints. *Operations Research*, 35(2), 254-265.

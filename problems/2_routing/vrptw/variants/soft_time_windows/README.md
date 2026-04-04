# VRPTW with Soft Time Windows (VRPTW Variant)

## What Changes

In the **soft time windows** variant, time window constraints [e_i, l_i] are
relaxed from hard constraints to soft constraints with penalty costs. A vehicle
may arrive after the latest time l_i, but incurs a per-unit-time penalty
proportional to the lateness. Early arrivals still wait (no penalty for
earliness). The objective becomes a weighted sum of travel distance and total
time window violation penalty. This models real-world delivery scenarios where
late delivery is costly but not catastrophic -- grocery delivery with customer
preferences, parcel delivery with service level agreements, and home healthcare
with appointment flexibility.

The key structural difference from hard VRPTW is that feasibility is no longer
binary per customer. Routes that would be infeasible under hard windows become
feasible but penalized, expanding the search space and enabling cost-quality
tradeoffs.

## Mathematical Formulation

The base VRPTW formulation replaces hard window constraints with penalties:

**Arrival time:**
```
a_i = max(e_i, arrival_time_i)    (wait if early)
```

**Lateness penalty:**
```
L_i = max(0, a_i - l_i)    for all customers i
```

**Objective (replaces pure distance):**
```
min  sum of route distances + alpha * sum_i L_i
```

where alpha > 0 is the penalty weight per unit of lateness. When alpha
approaches infinity, the problem reduces to hard VRPTW.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Soft TW (general) | NP-hard | Taillard et al. (1997) |
| Hard VRPTW | NP-hard | Solomon (1987) |
| CVRP (no windows) | NP-hard | -- |

NP-hard since it generalizes both CVRP and hard VRPTW. The soft windows
add flexibility but do not reduce computational complexity.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Solomon I1 (base VRPTW heuristic) | Partially | Respects hard windows; needs penalty adaptation |
| Nearest Neighbor TW (variant) | Yes | Greedy nearest with penalty-aware insertion |
| Sequential Insertion (variant) | Yes | Insert customers considering penalty cost |
| SA (variant metaheuristic) | Yes | Relocate/swap/2-opt with penalty evaluation |

## Implementations

Python files in this directory:
- `instance.py` -- SoftTWInstance, penalty weight, lateness computation
- `heuristics.py` -- Nearest neighbor TW, sequential insertion
- `metaheuristics.py` -- SA with relocate/swap/2-opt and penalty evaluation
- `tests/test_softtw.py` -- 16 tests

## Applications

- Grocery delivery (customer time preferences, not strict)
- Parcel delivery with SLAs (penalty for late delivery tiers)
- Home healthcare (appointment windows with patient flexibility)
- Field service scheduling (preferred vs mandatory time slots)

## Key References

- Taillard, E., Badeau, P., Gendreau, M., Guertin, F. & Potvin, J.Y. (1997). "A tabu search heuristic for the vehicle routing problem with soft time windows." Transportation Science 31(2), 170-186.
- Solomon, M.M. (1987). "Algorithms for the vehicle routing and scheduling problems with time window constraints." Operations Research 35(2), 254-265.
- Figliozzi, M.A. (2010). "An iterative route construction and improvement algorithm for the vehicle routing problem with soft time windows." Transportation Research Part C 18(5), 668-679. [TODO: verify]

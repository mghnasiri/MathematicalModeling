# Capacitated Facility Location Problem (FL Variant)

## What Changes

In the **capacitated** variant (CFLP), each facility i has a maximum capacity
u_i that limits the total customer demand it can serve. The base UFLP allows
any open facility to serve unlimited demand; CFLP adds the constraint that
the sum of demands assigned to facility i cannot exceed u_i. This models
practical limitations like warehouse throughput capacity, server bandwidth
limits, hospital bed counts, and school enrollment caps.

The key structural difference is that customer assignment becomes a
transportation sub-problem rather than simple nearest-facility assignment.
A customer may not be assigned to its cheapest open facility if that facility
is at capacity, leading to split assignments (fractional x_ij) in the LP
relaxation.

## Mathematical Formulation

The base UFLP formulation gains capacity constraints:

```
min  sum_i  f_i * y_i  +  sum_i sum_j  c_ij * x_ij
s.t. sum_i  x_ij = 1                    for all j  (full assignment)
     sum_j  d_j * x_ij <= u_i * y_i     for all i  (capacity limit)
     y_i in {0,1},  x_ij >= 0
```

The capacity constraint couples facility opening decisions with assignment
decisions, making the LP relaxation tighter but the problem harder. Note
that x_ij may be fractional (split assignments) in optimal solutions when
capacities are tight.

## Complexity

| Variant | Complexity | Approx Ratio | Reference |
|---------|-----------|-------------|-----------|
| CFLP (general) | NP-hard | -- | Cornuejols et al. (1991) |
| UFLP (uncapacitated base) | NP-hard | 1.488 | Li (2013) |
| Uniform capacities | NP-hard | -- | -- |

NP-hard and strictly harder than UFLP. The LP relaxation gap is larger
for CFLP, and the greedy analysis is more complex because capacity
feasibility must be maintained.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy Add (base UFLP) | Partially | Ignores capacity -- may violate limits |
| Greedy Add capacity-aware (variant) | Yes | Check capacity before each assignment |
| SA (variant metaheuristic) | Yes | Toggle/swap/reassign with capacity checks |

## Implementations

Python files in this directory:
- `instance.py` -- CFLPInstance, facility capacities, demand validation
- `heuristics.py` -- Capacity-aware greedy add
- `metaheuristics.py` -- SA with toggle/swap/reassign and capacity checks
- `tests/test_cflp.py` -- 30 tests

## Applications

- **Warehouse location**: distribution centers with limited throughput
- **Server placement**: data centers with bandwidth/CPU limits
- **Hospital siting**: emergency facilities with bed capacity
- **School districting**: schools with enrollment limits

## Key References

- Cornuejols, G., Sridharan, R. & Thizy, J.M. (1991). "A comparison of heuristics and relaxations for the capacitated plant location problem." EJOR 50(3), 280-297.
- Klose, A. & Drexl, A. (2005). "Facility location models for distribution system design." EJOR 162(1), 4-29.
- Shmoys, D.B., Tardos, E. & Aardal, K. (1997). "Approximation algorithms for facility location problems." Proc. 29th ACM STOC, 265-274.

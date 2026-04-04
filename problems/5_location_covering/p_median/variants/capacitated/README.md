# Capacitated p-Median Problem (p-Median Variant)

## What Changes

In the **capacitated** variant (CPMP), each facility has a maximum capacity u_i
that limits the total demand it can serve, in addition to the constraint that
exactly p facilities must be opened. The base p-median problem assigns each
customer to its nearest open facility without capacity restrictions; CPMP
requires that facility workloads respect capacity limits. This models scenarios
where service points have finite throughput -- distribution centers with limited
storage, schools with enrollment caps, and healthcare clinics with patient
capacity.

The key structural difference is that the assignment sub-problem becomes a
capacitated transportation problem rather than trivial nearest-assignment.
Customers may need to be assigned to a farther facility when nearer ones are
at capacity.

## Mathematical Formulation

The base p-median formulation gains capacity constraints:

```
min  sum_i sum_j  d_j * c_ij * x_ij
s.t. sum_i  x_ij = 1                    for all j  (full assignment)
     sum_j  d_j * x_ij <= u_i * y_i     for all i  (capacity)
     sum_i  y_i = p                      (exactly p open)
     y_i in {0,1},  x_ij >= 0
```

where d_j is customer j's demand, c_ij is the distance from customer j to
facility i, and u_i is facility i's capacity.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| CPMP (general) | NP-hard | Mulvey & Beck (1984) |
| p-Median (uncapacitated base) | NP-hard | Kariv & Hakimi (1979) |
| p = 1 (capacitated 1-median) | Polynomial | -- |

NP-hard since it generalizes the uncapacitated p-median problem. The capacity
constraints make the Teitz-Bart interchange heuristic more complex, as
swapping facilities requires re-solving the assignment sub-problem.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy Add (base p-median) | Partially | Ignores capacity constraints |
| Greedy Add (capacity-aware variant) | Yes | Verify capacity feasibility on each addition |
| Teitz-Bart (capacity-aware variant) | Yes | Interchange with capacity-feasible reassignment |
| SA (variant metaheuristic) | Yes | Facility swap + customer reassignment |

## Implementations

Python files in this directory:
- `instance.py` -- CPMedianInstance, facility capacities, demand data
- `heuristics.py` -- Capacity-aware greedy add, capacity-aware Teitz-Bart
- `metaheuristics.py` -- SA with facility swap and customer reassignment
- `tests/test_cpmp.py` -- 17 tests

## Applications

- Distribution center planning (limited warehouse throughput)
- School districting (enrollment capacity per school)
- Healthcare clinic siting (patient volume limits)
- Emergency service deployment (station capacity constraints)

## Key References

- Mulvey, J.M. & Beck, M.P. (1984). "Solving capacitated clustering problems." EJOR 18(3), 339-348.
- Lorena, L.A.N. & Senne, E.L.F. (2004). "A column generation approach to capacitated p-median problems." C&OR 31(6), 863-876.
- Kariv, O. & Hakimi, S.L. (1979). "An algorithmic approach to network location problems." SIAM J. Applied Mathematics 37(3), 539-560.

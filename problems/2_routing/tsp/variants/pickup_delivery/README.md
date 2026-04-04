# Pickup and Delivery Problem (TSP Variant)

## What Changes

The standard TSP visits all cities in any order. The **Pickup and Delivery Problem (PDP)** adds **precedence constraints**: each request consists of a pickup location and a delivery location, and the pickup must be visited before its corresponding delivery. A single vehicle starts and ends at a depot.

- **Precedence constraints** restrict the feasible permutation space, making many TSP tour improvements infeasible.
- **Paired structure**: locations come in pickup-delivery pairs (2n locations + 1 depot for n pairs).
- **Neighborhood design** must preserve or repair precedence feasibility after each move.
- **Solution validation** requires checking that every pickup appears before its matching delivery in the tour.

**Real-world motivation**: courier services (pickup packages, deliver to destinations), ride-sharing (pick up and drop off passengers), supply chain collect-and-deliver logistics, dial-a-ride transportation for disabled or elderly passengers.

## Mathematical Formulation

Extends base TSP with precedence constraints:

```
min  sum_{k=0}^{2n} d(pi(k), pi((k+1) mod (2n+1)))
s.t. pi is a permutation of {0, 1, ..., 2n}         (Hamiltonian cycle)
     pi(0) = 0                                       (start at depot)
     pos(pickup_i) < pos(delivery_i)  for all i      (precedence)
```

Locations: 0 = depot, 1..n = pickups, n+1..2n = deliveries. Pair i has pickup at location i and delivery at location i+n. Distances are Euclidean, computed from 2D coordinates.

## Complexity

- **NP-hard** (generalizes TSP: with no precedence constraints, PDP reduces to TSP).
- The precedence constraints reduce the feasible solution space but do not reduce the worst-case complexity class.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Nearest Feasible Neighbor | Yes | Greedy NN respecting precedence; O(n^2). |
| Cheapest Pair Insertion | Yes | Insert pickup-delivery pairs at cheapest feasible positions; O(n^3). |
| 2-opt | Limited | Must check precedence feasibility after each reversal. |
| Or-opt (relocate) | Yes | Move a location, reject if precedence violated. |
| Pair relocate | Yes | Move both pickup and delivery of a pair together. |
| Simulated Annealing | Yes | Relocate + swap + pair-relocate moves with precedence checks. |
| LNS (large neighborhood) | Possible | Remove and reinsert multiple pairs; effective for large instances. Not implemented. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `PDPInstance` (num_pairs, distance_matrix, coords), `PDPSolution`, `precedence_feasible()`, `validate_solution()`, `small_pdp_3()` benchmark |
| `heuristics.py` | `nearest_feasible()` (precedence-aware NN), `cheapest_pair_insertion()` (pair-wise insertion) |
| `metaheuristics.py` | `simulated_annealing()` with relocate/swap/pair-relocate moves, precedence feasibility rejection |
| `tests/test_pdp.py` | Test suite covering precedence validation, heuristic correctness, SA improvement |

## Relationship to Base TSP

When there are no precedence constraints (or when all pickups are visited before all deliveries trivially), the PDP reduces to a standard TSP over 2n+1 locations. The precedence constraints are the sole structural addition, and they restrict the feasible tour space to a strict subset of all Hamiltonian cycles.

## Key References

- Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery problem. *Transportation Science*, 29(1), 17-29.
- Ruland, K.S. & Rodin, E.Y. (1997). The pickup and delivery problem: Faces and branch-and-cut algorithm. *Computers & Mathematics with Applications*, 33(12), 1-13.
- Renaud, J., Boctor, F.F. & Laporte, G. (2000). An improved petal heuristic for the vehicle routing problem. *JORS*, 51(8), 923-928.

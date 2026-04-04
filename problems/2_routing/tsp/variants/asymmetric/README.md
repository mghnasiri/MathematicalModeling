# Asymmetric Traveling Salesman Problem (TSP Variant)

## What Changes

The standard (symmetric) TSP assumes d(i,j) = d(j,i) for all city pairs. The **Asymmetric TSP (ATSP)** drops this assumption: the distance matrix is directed, so the cost from city i to j may differ from j to i. This fundamentally changes the problem structure:

- **2-opt is ineffective**: reversing a tour segment changes all arc directions, often increasing cost rather than improving it. Symmetric local search moves do not transfer directly.
- **Approximation is harder**: symmetric TSP admits a 1.5-approx (Christofides), but ATSP only has O(log n / log log n) approximation (Asadpour et al., 2017).

**Real-world motivation**: one-way streets in urban routing, shipping with different upstream/downstream costs, airline routing with wind effects, asymmetric tariffs in logistics networks, robot path planning in environments with directional constraints.

## Mathematical Formulation

Only the distance symmetry assumption changes from base TSP:

```
min  sum_{k=0}^{n-1} d(pi(k), pi((k+1) mod n))
s.t. pi is a permutation of {0, ..., n-1}           (Hamiltonian cycle)
     d(i,j) != d(j,i) in general                    (asymmetric distances)
```

The distance matrix D is a general n x n matrix with D[i][i] = 0 but no symmetry constraint. Note that any symmetric TSP instance is a valid ATSP instance (with D = D^T), so ATSP strictly generalizes symmetric TSP.

## Complexity

- **NP-hard**, same class as symmetric TSP.
- **Harder to approximate**: best known is O(log n / log log n) (Asadpour et al., 2017), compared to 1.5-approx for metric symmetric TSP.
- The ATSP can be reduced to a symmetric TSP of size 2n (Jonker & Volgenant transformation), but this doubles instance size.
- Standard benchmark library: TSPLIB contains ATSP instances (e.g., ftv33, ftv47, rbg323).

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Nearest Neighbor | Yes | Directed NN, O(n^2). Same greedy logic but follows arcs. |
| Multi-start NN | Yes | Try all n starting cities, keep best. |
| 2-opt | No | Segment reversal changes arc directions unpredictably. |
| Or-opt (relocate) | Yes | Moving a single city preserves arc direction intent. |
| Swap | Yes | Exchange two city positions; direction-aware. |
| Simulated Annealing | Yes | Or-opt + swap neighborhood. Warm-started with multi-start NN. |
| Assignment relaxation | Possible | Solve linear assignment for lower bound; patch subtours. Not implemented. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `ATSPInstance` and `ATSPSolution` dataclasses, directed `tour_cost()`, `small_atsp_5()` benchmark, `validate_solution()` |
| `heuristics.py` | `nearest_neighbor_atsp()` (directed NN), `multi_start_nn()` (all starts) |
| `metaheuristics.py` | `simulated_annealing()` with or-opt + swap moves, geometric cooling, time limit support |
| `tests/test_atsp.py` | Test suite covering correctness, solution validation, and SA improvement |

## Key References

- Kanellakis, P.C. & Papadimitriou, C.H. (1980). Local search for the asymmetric traveling salesman problem. *Operations Research*, 28(5), 1086-1099.
- Asadpour, A., Goemans, M.X., Madry, A., Oveis Gharan, S. & Saberi, A. (2017). An O(log n / log log n)-approximation algorithm for the ATSP. *Operations Research*, 65(4), 1043-1061.
- Cirasella, J., Johnson, D.S., McGeoch, L.A. & Zhang, W. (2001). The asymmetric traveling salesman problem: algorithms, instance generators, and tests. *ALENEX*, 32-59. [TODO: verify DOI]

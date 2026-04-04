# Prize-Collecting TSP (TSP Variant)

## What Changes

The standard TSP requires visiting **all** cities. The **Prize-Collecting TSP (PCTSP)** makes city visits **optional**: each city has a prize, and the goal is to select a profitable subset of cities and find a short tour through them. The trade-off is between travel cost and collected prizes.

- **City selection** becomes a decision variable: not all cities need to be visited.
- **Minimum prize threshold**: a constraint may require collecting at least a target total prize.
- **Objective changes** from pure distance minimization to: minimize (travel cost - collected prizes).
- The solution space is larger than TSP since both the city subset and visit order must be chosen.

**Real-world motivation**: sales territory planning (visit high-value customers, skip low-value ones), orienteering (collect points within a time budget), selective routing in logistics, tourist trip planning.

## Mathematical Formulation

Extends base TSP with city selection and prizes:

```
min  sum_{k} d(pi(k), pi((k+1) mod |S|)) - sum_{i in S} p_i
s.t. S subseteq {0, ..., n-1}                    (city subset selection)
     sum_{i in S} p_i >= P_min                    (minimum prize threshold)
     pi is a Hamiltonian cycle over S             (tour through selected cities)
```

Where p_i is the prize at city i and P_min is the minimum prize to collect.

## Complexity

- **NP-hard** (generalizes TSP: when all prizes are required, PCTSP reduces to TSP).
- A **2-approximation** is known via primal-dual methods (Goemans & Williamson, 1995).
- The orienteering variant (maximize prize within distance budget) is also NP-hard.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy Prize Insertion | Yes | Add cities by prize-to-insertion-cost ratio until min_prize met. |
| Nearest Neighbor PCTSP | Yes | NN visiting nearest cities until prize threshold reached. |
| Add/remove moves | Yes | Add an unvisited city or remove a low-value city from the tour. |
| 2-opt | Yes | Applied to the current tour subset only. |
| Simulated Annealing | Yes | Add/remove/swap/2-opt moves on variable-length tours. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `PCTSPInstance` (coords, prizes, min_prize), `PCTSPSolution` (tour, travel_cost, total_prize, objective), `validate_solution()`, `small_pctsp_6()` benchmark |
| `heuristics.py` | `greedy_prize()` (ratio-based insertion), `nearest_neighbor_pctsp()` (NN until threshold) |
| `metaheuristics.py` | `simulated_annealing()` with add/remove/swap/2-opt moves on variable-length tours |
| `tests/test_pctsp.py` | Test suite covering prize constraints, objective computation, heuristic quality |

## Key References

- Balas, E. (1989). The prize collecting traveling salesman problem. *Networks*, 19(6), 621-636.
- Goemans, M.X. & Williamson, D.P. (1995). A general approximation technique for constrained forest problems. *SIAM Journal on Computing*, 24(2), 296-317.
- Dell'Amico, M., Maffioli, F. & Varbrand, P. (1995). On prize-collecting tours and the asymmetric travelling salesman problem. *International Transactions in OR*, 2(3), 297-308. [TODO: verify DOI]

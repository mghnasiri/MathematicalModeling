# Vehicle Routing Problem with Time Windows (VRPTW)

## Problem Definition

Extends the CVRP with **time window constraints** $[e_i, l_i]$ for each customer $i$. A vehicle must arrive at customer $i$ no later than $l_i$. If it arrives before $e_i$, it waits until $e_i$ (no early service). Service takes $s_i$ time units. The depot has a time window $[e_0, l_0]$ defining the planning horizon.

## Mathematical Formulation

**Parameters:**
- $n$: number of customers
- $Q$: vehicle capacity
- $q_i$: demand of customer $i$
- $d_{ij}$: distance/travel time from node $i$ to node $j$
- $[e_i, l_i]$: time window for node $i$
- $s_i$: service time at node $i$

**Constraints (in addition to CVRP):**
- $a_i \leq l_i$ for all customers $i$ (arrive before window closes)
- $b_i = \max(a_i, e_i)$ (service begins at arrival or window opening)
- $a_j \geq b_i + s_i + t_{ij}$ for consecutive visits $i \to j$

## Complexity

NP-hard — generalizes CVRP (set all time windows to $[0, \infty]$).

## Solution Approaches

### Constructive Heuristics

| Method | Complexity | Description |
|--------|-----------|-------------|
| Solomon I1 | $O(n^2 K)$ | Iterative insertion with composite criterion (distance + time urgency) |
| Nearest Neighbor TW | $O(n^2)$ | Greedy nearest feasible customer, start new route when blocked |

### Metaheuristics

| Method | Neighborhoods | Description |
|--------|--------------|-------------|
| Simulated Annealing | Relocate, swap | Feasibility-checked inter-route moves |
| Genetic Algorithm | Giant-tour OX | TW-aware split decoder |

## Implementations in This Repository

```
vrptw/
├── instance.py              # VRPTWInstance, VRPTWSolution, validation
├── heuristics/
│   └── solomon_insertion.py # Solomon I1 + nearest neighbor TW
├── metaheuristics/
│   ├── simulated_annealing.py # Relocate/swap with TW feasibility
│   └── genetic_algorithm.py # Giant-tour encoding, TW-aware split
└── tests/
    └── test_vrptw.py        # 31 tests, 8 test classes
```

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Soft Time Windows](variants/soft_time_windows/) | `variants/soft_time_windows/` | Time window violations allowed with penalty costs |

## Key References

- Solomon, M.M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Oper. Res.*, 35(2), 254-265. https://doi.org/10.1287/opre.35.2.254
- Desrochers, M., Desrosiers, J. & Solomon, M. (1992). A new optimization algorithm for the VRP with time windows. *Oper. Res.*, 40(2), 342-354. https://doi.org/10.1287/opre.40.2.342
- Chiang, W.-C. & Russell, R.A. (1996). SA metaheuristics for the VRPTW. *Ann. Oper. Res.*, 63(1), 3-27. https://doi.org/10.1007/BF02601637
- Potvin, J.-Y. & Bengio, S. (1996). The VRP with time windows part II: Genetic search. *INFORMS J. Comput.*, 8(2), 165-172. https://doi.org/10.1287/ijoc.8.2.165

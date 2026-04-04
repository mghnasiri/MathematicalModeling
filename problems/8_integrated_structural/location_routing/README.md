# Location-Routing Problem (LRP)

## Family 8 -- Integrated Structural

## 1. Problem Definition

The **Location-Routing Problem (LRP)** jointly optimizes facility (depot) location decisions and vehicle routing decisions. Given a set of candidate depot locations with fixed opening costs and capacities, a set of customers with demands, and a fleet of capacitated vehicles, the goal is to:

1. Select which depots to open.
2. Assign customers to open depots.
3. Design vehicle routes from each depot to serve its assigned customers.

The objective is to minimize the **total cost**, comprising depot opening costs and total routing distance. Solving location and routing sequentially (first locate, then route) does not guarantee global optimality, making the integrated approach essential.

**Complexity:** NP-hard (generalizes both the Uncapacitated Facility Location Problem and the Capacitated Vehicle Routing Problem).

### Variants

| Variant | Description |
|---------|-------------|
| **CLRP** | Capacitated LRP -- depots have finite capacity (implemented here) |
| **ULRP** | Uncapacitated LRP -- depots have unlimited capacity |
| **LRP-TW** | LRP with time windows on customer service |
| **LRP-PD** | LRP with pickup and delivery |
| **Multi-echelon LRP** | Hierarchical depot structure (e.g., warehouses + satellites) |

## 2. Mathematical Formulation (MILP)

### Sets and Parameters

| Symbol | Description |
|--------|-------------|
| J = {1, ..., m} | Candidate depot locations |
| I = {1, ..., n} | Customer locations |
| f_j | Fixed opening cost of depot j |
| U_j | Capacity of depot j (maximum total demand) |
| d_i | Demand of customer i |
| Q | Vehicle capacity |
| c_{ij} | Distance/cost between nodes i and j |

### Decision Variables

| Variable | Description |
|----------|-------------|
| y_j in {0, 1} | 1 if depot j is opened |
| x_{ijk} in {0, 1} | 1 if vehicle travels arc (i, k) from depot j |
| z_{ij} in {0, 1} | 1 if customer i is assigned to depot j |

### Objective

```
min  sum_{j in J} f_j * y_j  +  sum_{j in J} sum_{(i,k) in A} c_{ik} * x_{ijk}
```

### Constraints

```
(1)  sum_{j in J} z_{ij} = 1                       for all i in I     (each customer assigned once)
(2)  z_{ij} <= y_j                                  for all i, j       (assign only to open depots)
(3)  sum_{i in I} d_i * z_{ij} <= U_j               for all j in J     (depot capacity)
(4)  sum_{i in S} d_i * z_{ij} <= Q                 for all routes S   (vehicle capacity)
(5)  flow conservation and subtour elimination       (standard VRP constraints)
(6)  y_j in {0,1}, z_{ij} in {0,1}, x_{ijk} in {0,1}
```

## 3. Solution Methods

### Sequential Greedy Heuristic (implemented)

A three-phase decomposition approach:

1. **Phase 1 -- Location:** Open depots greedily by efficiency score (fixed cost normalized by capacity and proximity to customers). Assign each customer to the nearest open depot with sufficient capacity.
2. **Phase 2 -- Routing:** For each open depot, build vehicle routes using nearest-neighbor insertion, respecting vehicle capacity Q.
3. **Phase 3 -- Improvement:** Reassign border customers between depots when it significantly reduces routing cost without violating capacity constraints.

Complexity: O(m*n + n^2).

### Simulated Annealing (implemented)

Iterative improvement metaheuristic with four neighborhood operators:

| Operator | Description |
|----------|-------------|
| **Toggle depot** | Open a closed depot or close an open one (redistributing customers) |
| **Reassign customer** | Move a customer to a different open depot |
| **Swap customers** | Exchange two customers between different depots |
| **2-opt intra-route** | Reverse a segment within a single route |

Warm-started with the greedy heuristic. Uses geometric cooling schedule with Boltzmann acceptance.

### Other Methods (not yet implemented)

| Method | Reference |
|--------|-----------|
| GRASP + ILP | Prins et al. (2006) |
| Adaptive Large Neighborhood Search | Hemmelmayr et al. (2012) |
| Branch-and-Price | Baldacci et al. (2011) |
| Two-phase Lagrangean relaxation | Prins et al. (2006) |

## 4. Implementations

```
location_routing/
    instance.py                      LRPInstance, LRPSolution, validation
    heuristics/
        greedy_lrp.py                Sequential greedy (location + NN routing)
    metaheuristics/
        simulated_annealing.py       SA with toggle/reassign/swap/2-opt moves
    tests/
        test_lrp.py                  23 tests, 5 test classes
    README.md                        This file
```

### Instance Format

- **Depots** indexed 0..m-1 in the distance matrix
- **Customers** indexed m..m+n-1 in the distance matrix
- Customer indices in routes and assignments are 0-based (0..n-1)
- `LRPInstance.customer_node(i)` maps customer i to distance matrix index m+i
- `LRPInstance.depot_node(j)` maps depot j to distance matrix index j

### Benchmark Instances

| Name | Depots | Customers | Q | Description |
|------|--------|-----------|---|-------------|
| `small_3_8` | 3 | 8 | 30 | Hand-crafted with spatial clusters |
| `medium_5_15` | 5 | 15 | 50 | Seed-generated random instance |

### Running

```bash
# Run all LRP tests (23 tests)
python -m pytest problems/8_integrated_structural/location_routing/tests/ -v

# Run specific test class
python -m pytest problems/8_integrated_structural/location_routing/tests/test_lrp.py::TestGreedyLRP -v
```

## 5. Key References

1. **Laporte, G.** (1988). Location-routing problems. In: Golden, B.L. & Assad, A.A. (eds) *Vehicle Routing: Methods and Studies*, North-Holland, 163-198.

2. **Nagy, G. & Salhi, S.** (2007). Location-routing: Issues, models and methods. *European Journal of Operational Research*, 177(2), 649-672. [doi:10.1016/j.ejor.2006.04.004](https://doi.org/10.1016/j.ejor.2006.04.004)

3. **Prodhon, C. & Prins, C.** (2014). A survey of recent research on location-routing problems. *European Journal of Operational Research*, 238(1), 1-17. [doi:10.1016/j.ejor.2014.01.005](https://doi.org/10.1016/j.ejor.2014.01.005)

4. **Drexl, M. & Schneider, M.** (2015). A survey of variants and extensions of the location-routing problem. *European Journal of Operational Research*, 241(2), 283-308. [doi:10.1016/j.ejor.2014.08.030](https://doi.org/10.1016/j.ejor.2014.08.030)

5. **Prins, C., Prodhon, C. & Wolfler Calvo, R.** (2006). Solving the capacitated location-routing problem by a cooperative Lagrangean relaxation-granular tabu search heuristic. *Transportation Science*, 40(1), 18-32. [doi:10.1287/trsc.1050.0126](https://doi.org/10.1287/trsc.1050.0126)

6. **Yu, V.F., Lin, S.W., Lee, W. & Ting, C.J.** (2010). A simulated annealing heuristic for the capacitated location routing problem. *Computers & Industrial Engineering*, 58(2), 288-299. [doi:10.1016/j.cie.2009.10.007](https://doi.org/10.1016/j.cie.2009.10.007)

7. **Salhi, S. & Rand, G.K.** (1989). The effect of ignoring routes when locating depots. *European Journal of Operational Research*, 39(2), 150-156. [doi:10.1016/0377-2217(89)90188-4](https://doi.org/10.1016/0377-2217(89)90188-4)

## See also

- [`../../5_location_covering/facility_location/`](../../5_location_covering/facility_location/) -- Facility location implementations (UFLP)
- [`../../2_routing/cvrp/`](../../2_routing/cvrp/) -- Capacitated Vehicle Routing Problem
- [`../../2_routing/vrptw/`](../../2_routing/vrptw/) -- VRP with Time Windows

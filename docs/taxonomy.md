# Problem Taxonomy — Full Classification

## Family 1 · Scheduling
**Path**: `problems/1_scheduling/`

### 1. Single Machine Scheduling
**Path**: `problems/1_scheduling/single_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total completion | 1 \|\| ΣCj | P | SPT |
| Weighted completion | 1 \|\| ΣwjCj | P | WSPT |
| Maximum lateness | 1 \|\| Lmax | P | EDD |
| Number of tardy jobs | 1 \|\| ΣUj | P | Moore's |
| Total tardiness | 1 \|\| ΣTj | NP-hard | DP |
| Weighted tardiness | 1 \|\| ΣwjTj | NP-hard (strongly) | B&B, SA |

### 2. Parallel Machine Scheduling
**Path**: `problems/1_scheduling/parallel_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Identical, makespan | Pm \|\| Cmax | NP-hard | LPT (4/3) |
| Uniform machines | Qm \|\| Cmax | NP-hard | |
| Unrelated machines | Rm \|\| Cmax | NP-hard (strongly) | LP rounding |

### 3. Flow Shop Scheduling
**Path**: `problems/1_scheduling/flow_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 2-machine makespan | F2 \|\| Cmax | P | Johnson's rule |
| m-machine makespan | Fm \| prmu \| Cmax | NP-hard (m >= 3) | NEH, IG |
| Blocking flow shop | Fm \| block \| Cmax | NP-hard (m >= 3) | |
| No-wait flow shop | Fm \| no-wait \| Cmax | NP-hard (m >= 3) | TSP reduction |

### 4. Job Shop Scheduling
**Path**: `problems/1_scheduling/job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Makespan | Jm \|\| Cmax | NP-hard (m >= 2) | Tabu search, CP |

### 5. Flexible Job Shop Scheduling
**Path**: `problems/1_scheduling/flexible_job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total flexibility | T-FJSP | NP-hard | Integrated TS, GA |
| Partial flexibility | P-FJSP | NP-hard | Hierarchical TS |

### 6. Resource-Constrained Project Scheduling
**Path**: `problems/1_scheduling/rcpsp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Basic | RCPSP | NP-hard (strongly) | GA + SGS |

---

## Family 2 · Routing
**Path**: `problems/2_routing/`

### 7. Traveling Salesman Problem
**Path**: `problems/2_routing/tsp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Symmetric TSP | TSP | NP-hard | Held-Karp DP, B&B |
| Asymmetric TSP | ATSP | NP-hard | B&B |

### 8. Capacitated Vehicle Routing
**Path**: `problems/2_routing/cvrp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| CVRP | CVRP | NP-hard | Clarke-Wright, Sweep |

### 9. VRP with Time Windows
**Path**: `problems/2_routing/vrptw/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| VRPTW | VRPTW | NP-hard | Solomon I1, SA, GA |

---

## Family 3 · Packing & Cutting
**Path**: `problems/3_packing_cutting/`

### 10. 0-1 Knapsack
**Path**: `problems/3_packing_cutting/knapsack/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 0-1 Knapsack | KP01 | NP-hard (weakly) | DP O(nW), B&B |

### 11. 1D Bin Packing
**Path**: `problems/3_packing_cutting/bin_packing/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Bin Packing | BPP1D | NP-hard (strongly) | FFD (11/9 approx) |

### 12. 1D Cutting Stock
**Path**: `problems/3_packing_cutting/cutting_stock/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Cutting Stock | CSP1D | NP-hard | Column generation |

---

## Family 4 · Assignment & Matching
**Path**: `problems/4_assignment_matching/`

### 13. Linear Assignment
**Path**: `problems/4_assignment_matching/assignment/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Square LAP | LAP | P | Hungarian O(n^3) |

### 14. Quadratic Assignment
**Path**: `problems/4_assignment_matching/quadratic_assignment/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| QAP | QAP | NP-hard | SA, TS |

---

## Family 5 · Location & Covering
**Path**: `problems/5_location_covering/`

### 15. Facility Location
**Path**: `problems/5_location_covering/facility_location/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Uncapacitated | UFLP | NP-hard | Greedy add/drop, SA |
| Capacitated | CFLP | NP-hard | — |

### 16. p-Median
**Path**: `problems/5_location_covering/p_median/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| p-Median | PMP | NP-hard | Greedy, Teitz-Bart |

### 17. Set Covering
**Path**: `problems/5_location_covering/set_covering/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Set Covering | SCP | NP-hard | Greedy (ln n approx) |

### 18. Set Packing
**Path**: `problems/5_location_covering/set_packing/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Set Packing | SPP | NP-hard | Greedy |

---

## Family 6 · Network Flow & Design
**Path**: `problems/6_network_flow_design/`

### 19. Shortest Path
**Path**: `problems/6_network_flow_design/shortest_path/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Non-negative SSSP | SPP | P | Dijkstra O((V+E) log V) |
| General SSSP | SPP | P | Bellman-Ford O(VE) |

### 20. Maximum Flow
**Path**: `problems/6_network_flow_design/max_flow/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Max Flow / Min Cut | Max-Flow | P | Edmonds-Karp O(VE^2) |

### 21. Minimum Spanning Tree
**Path**: `problems/6_network_flow_design/min_spanning_tree/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| MST | MST | P | Kruskal O(E log E), Prim O(E log V) |

---

## Family 7 · Inventory & Lot Sizing
**Path**: `problems/7_inventory_lotsizing/`

### 22. EOQ Models
**Path**: `problems/7_inventory_lotsizing/eoq/`

### 23. Lot Sizing
**Path**: `problems/7_inventory_lotsizing/lot_sizing/`

### 24. Wagner-Whitin
**Path**: `problems/7_inventory_lotsizing/wagner_whitin/`

### 25. Capacitated Lot Sizing
**Path**: `problems/7_inventory_lotsizing/capacitated_lot_sizing/`

### 26. Multi-Echelon Inventory
**Path**: `problems/7_inventory_lotsizing/multi_echelon_inventory/`

### 27. Safety Stock
**Path**: `problems/7_inventory_lotsizing/safety_stock/`

---

## Family 8 · Integrated Structural
**Path**: `problems/8_integrated_structural/`

### 28. Location-Routing Problem (LRP)
Combined facility location and vehicle routing.

### 29. Inventory-Routing Problem (IRP)
Combined inventory replenishment and delivery routing.

### 30. Assembly Line Balancing (SALBP)
Combined task assignment and scheduling with cycle time constraints.

---

## Family 9 · Uncertainty Modeling
**Path**: `problems/9_uncertainty_modeling/`

Paradigms for optimization under uncertainty, applied across all problem families.

- Stochastic Programming (Two-Stage SP, SAA)
- Robust Optimization (min-max, min-max regret)
- Chance-Constrained Programming
- Distributionally Robust Optimization (DRO)

### Implemented problems
- Newsvendor
- Two-Stage SP
- Robust Shortest Path
- Stochastic Knapsack
- Chance-Constrained Facility Location
- Robust Portfolio
- Stochastic VRP
- Robust Scheduling
- DRO

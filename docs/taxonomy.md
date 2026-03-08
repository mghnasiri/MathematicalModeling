# Problem Taxonomy — Full Classification

## Scheduling Problems (Phase 1)

### 1. Single Machine Scheduling
**Path**: `problems/scheduling/single_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total completion | $1 \mid\mid \sum C_j$ | P | SPT |
| Weighted completion | $1 \mid\mid \sum w_j C_j$ | P | WSPT |
| Maximum lateness | $1 \mid\mid L_{\max}$ | P | EDD |
| Number of tardy jobs | $1 \mid\mid \sum U_j$ | P | Moore's |
| Total tardiness | $1 \mid\mid \sum T_j$ | NP-hard | DP |
| Weighted tardiness | $1 \mid\mid \sum w_j T_j$ | NP-hard (strongly) | B&B, SA |
| With release dates | $1 \mid r_j \mid \sum C_j$ | NP-hard | B&B |
| With setups (= TSP) | $1 \mid s_{jk} \mid C_{\max}$ | NP-hard | TSP methods |

### 2. Parallel Machine Scheduling
**Path**: `problems/scheduling/parallel_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Identical, makespan | $P_m \mid\mid C_{\max}$ | NP-hard | LPT (4/3) |
| Preemptive makespan | $P_m \mid pmtn \mid C_{\max}$ | P | McNaughton |
| Total completion | $P_m \mid\mid \sum C_j$ | P | SPT |
| Uniform machines | $Q_m \mid\mid C_{\max}$ | NP-hard | |
| Unrelated machines | $R_m \mid\mid C_{\max}$ | NP-hard (strongly) | LP rounding |

### 3. Flow Shop Scheduling
**Path**: `problems/scheduling/flow_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 2-machine makespan | $F_2 \mid\mid C_{\max}$ | P | Johnson's rule |
| m-machine makespan | $F_m \mid\mid C_{\max}$ | NP-hard ($m \geq 3$) | NEH, IG |
| Permutation flow shop | $F_m \mid prmu \mid C_{\max}$ | NP-hard ($m \geq 3$) | NEH, IG |
| Blocking flow shop | $F_m \mid block \mid C_{\max}$ | NP-hard ($m \geq 3$) | |
| No-wait flow shop | $F_m \mid no\text{-}wait \mid C_{\max}$ | NP-hard ($m \geq 3$) | TSP reduction |

### 4. Job Shop Scheduling
**Path**: `problems/scheduling/job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Makespan | $J_m \mid\mid C_{\max}$ | NP-hard ($m \geq 2$) | Tabu search, CP |
| 2-job case | $J_m \mid n=2 \mid C_{\max}$ | P | Jackson's rule |
| With recirculation | $J_m \mid rcrc \mid C_{\max}$ | NP-hard | |

### 5. Flexible Job Shop Scheduling
**Path**: `problems/scheduling/flexible_job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total flexibility | T-FJSP | NP-hard | Integrated TS, GA |
| Partial flexibility | P-FJSP | NP-hard | Hierarchical TS |
| Multi-objective | MO-FJSP | NP-hard | NSGA-II |

### 6. Resource-Constrained Project Scheduling
**Path**: `problems/scheduling/rcpsp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Basic | RCPSP | NP-hard (strongly) | GA + SGS |
| Multi-mode | MRCPSP | NP-hard | |
| Gen. precedence | RCPSP/max | NP-hard | |
| Multi-skill | MS-RCPSP | NP-hard | |

---

## Routing Problems (Phase 2)

### 7. Traveling Salesman Problem
**Path**: `problems/routing/tsp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Symmetric TSP | TSP | NP-hard | Held-Karp DP, B&B |
| Asymmetric TSP | ATSP | NP-hard | B&B |

### 8. Capacitated Vehicle Routing
**Path**: `problems/routing/cvrp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| CVRP | CVRP | NP-hard | Clarke-Wright, Sweep |

### 9. VRP with Time Windows
**Path**: `problems/routing/vrptw/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| VRPTW | VRPTW | NP-hard | Solomon I1, SA, GA |

---

## Packing & Cutting Problems (Phase 3)

### 10. 0-1 Knapsack
**Path**: `problems/packing/knapsack/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 0-1 Knapsack | KP01 | NP-hard (weakly) | DP O(nW), B&B |
| Fractional Knapsack | — | P | Greedy (ratio sort) |
| Bounded Knapsack | BKP | NP-hard | DP |

### 11. 1D Bin Packing
**Path**: `problems/packing/bin_packing/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Bin Packing | BPP1D | NP-hard (strongly) | FFD (11/9 approx) |
| 2D Bin Packing | BPP2D | NP-hard | — |

### 12. 1D Cutting Stock
**Path**: `problems/packing/cutting_stock/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Cutting Stock | CSP1D | NP-hard | Column generation |

---

## Location & Network Problems (Phase 4)

### 13. Facility Location
**Path**: `problems/location_network/facility_location/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Uncapacitated | UFLP | NP-hard | Greedy add/drop, SA |
| Capacitated | CFLP | NP-hard | — |

### 14. p-Median
**Path**: `problems/location_network/p_median/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| p-Median | PMP | NP-hard | Greedy, Teitz-Bart |
| p-Center | PCP | NP-hard | — |

### 15. Shortest Path
**Path**: `problems/location_network/shortest_path/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Non-negative SSSP | SPP | P | Dijkstra O((V+E) log V) |
| General SSSP | SPP | P | Bellman-Ford O(VE) |
| All-pairs | APSP | P | Floyd-Warshall O(V^3) |

---

## Future Phases (Planned)

### Phase 5: Stochastic & Robust
- Two-Stage SP, Robust Optimization, Chance-Constrained

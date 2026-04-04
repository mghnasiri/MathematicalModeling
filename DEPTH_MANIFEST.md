# DEPTH MANIFEST — Second Pass

> Generated: 2026-04-04 | Depth & Variant Enrichment Pass
>
> This document inventories all work needed in the second pass: deep README
> enrichment, variant expansion, code audit, and Family 8 implementation.

---

## Section A: 15 Priority Problem Families

| # | Family | README Lines | .py Files | Test Files | Gap Assessment |
|---|--------|-------------|-----------|------------|----------------|
| 1 | `1_scheduling/flow_shop/` | 462 | 41 | 7 | Already deepest README. Needs: all standard formulations (time-indexed, CP-SAT), complete BKS table from Taillard, full pseudocode for all 11 heuristics + 17 metaheuristics, parameter tables, hybrid/advanced methods, Lagrangian relaxation |
| 2 | `1_scheduling/job_shop/` | 248 | 15 | 5 | Needs: disjunctive graph formulation detail, all 5 metaheuristic pseudocodes, benchmark instances (ft06/ft10/ft20/la01-la40), N1/N5/N7 neighborhood definitions, Shifting Bottleneck pseudocode |
| 3 | `1_scheduling/single_machine/` | 254 | 18 | 6 | Needs: all formulations (MILP for tardiness, DP recurrence), complete dispatching rule proofs, ATC formula derivation, B&B detail, all 6 metaheuristic pseudocodes |
| 4 | `1_scheduling/parallel_machine/` | 187 | 17 | 6 | Needs: MIP formulation detail, MULTIFIT pseudocode, LPT approximation proof sketch, all 6 metaheuristic pseudocodes + parameter tables |
| 5 | `1_scheduling/nurse_scheduling/` | 35 | 3 | 1 | **Severely short** (35 lines). Needs: full build from scratch — NSP formulation, shift patterns, greedy roster pseudocode, BKS from NSPLib |
| 6 | `2_routing/tsp/` | 331 | 18 | 5 | Needs: MCF/SCF formulations, Christofides description, 1-tree pseudocode, TSPLIB BKS table, all 7 metaheuristic pseudocodes + parameter tables |
| 7 | `2_routing/cvrp/` | 302 | 16 | 6 | Needs: set-partitioning formulation, capacity cuts, split procedure detail, CVRPLIB BKS table, all 7 metaheuristic pseudocodes + ALNS description |
| 8 | `2_routing/vrptw/` | 174 | 15 | 6 | Needs: TW propagation formulation, Solomon I1 pseudocode, Solomon BKS table (C/R/RC classes), all 7 metaheuristic pseudocodes |
| 9 | `3_packing_cutting/knapsack/` | 237 | 16 | 6 | Needs: FPTAS description, core problem concept, Pisinger BKS, DP space optimization, all 6 metaheuristic pseudocodes |
| 10 | `3_packing_cutting/bin_packing/` | 147 | 14 | 6 | Needs: set-cover ILP, FFD pseudocode with 11/9 proof sketch, L1/L2 lower bounds, all 6 metaheuristic pseudocodes |
| 11 | `3_packing_cutting/cutting_stock/` | 130 | 16 | 7 | Needs: Gilmore-Gomory column generation detail (master + pricing), pattern generation pseudocode, all 6 metaheuristic pseudocodes |
| 12 | `5_location_covering/facility_location/` | 96 | 14 | 6 | Needs: LP relaxation quality, Jain-Vazirani primal-dual, 1.488-approx discussion, all 6 metaheuristic pseudocodes |
| 13 | `6_network_flow_design/shortest_path/` | 105 | 4 | 1 | Needs: LP formulation (TU), A* description, Johnson's algorithm, BFS for unweighted, Fibonacci heap discussion |
| 14 | `9_uncertainty_modeling/two_stage_sp/` | 123 | 4 | 1 | Needs: L-shaped decomposition, SAA convergence theory, EVPI/VSS formulas, illustrative instance walkthrough |
| 15 | `9_uncertainty_modeling/newsvendor/` | 79 | 4 | 1 | Needs: censored demand, multi-product with budget, service level approach, demand distribution fitting |

### Summary

- **Total README lines across 15 families:** 3,259 (current)
- **Target:** 6,000–12,000 lines (400–800 per family)
- **Total .py files to audit:** 215 (excluding variants)
- **Total test files:** 74

---

## Section B: All 48 Variant Folders

### Flow Shop Variants (9)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/no_wait/` | 74 | Yes (3 .py) | Already decent; expand formulation, BKS |
| `variants/blocking/` | 64 | Yes (3 .py) | Expand formulation detail, add BKS |
| `variants/setup_times/` | 64 | Yes (3 .py) | Expand formulation detail, add BKS |
| `variants/open_shop/` | 21 | No | Needs: formulation, complexity, methods table |
| `variants/stochastic/` | 19 | No | Needs: stochastic model, recourse description |
| `variants/hybrid/` | 18 | No | Needs: parallel machine stage model |
| `variants/tardiness/` | 18 | No | Needs: tardiness objective, weighted variant |
| `variants/lot_streaming/` | 17 | No | Needs: sublot formulation, continuous vs. discrete |
| `variants/distributed/` | 17 | No | Needs: factory assignment + sequencing model |

### CVRP Variants (10)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/open_vrp/` | 31 | Yes (3 .py) | Expand: modified savings, route not return |
| `variants/electric/` | 24 | Yes (3 .py) | Expand: battery constraints, charging stations |
| `variants/backhaul/` | 23 | Yes (3 .py) | Expand: linehaul-first constraint |
| `variants/multi_trip/` | 23 | Yes (3 .py) | Expand: trip chaining, route duration |
| `variants/backhauls/` | 22 | Yes (3 .py) | Expand: mixed delivery/pickup |
| `variants/split_delivery/` | 17 | Yes (3 .py) | Needs: formulation allowing split visits |
| `variants/multi_compartment/` | 17 | Yes (3 .py) | Needs: compartment capacity constraints |
| `variants/multi_depot/` | 16 | Yes (3 .py) | Needs: depot assignment + routing |
| `variants/periodic/` | 16 | Yes (3 .py) | Needs: visit frequency + day assignment |
| `variants/cumulative/` | 16 | Yes (3 .py) | Needs: cumulative objective (sum of arrival times) |

### TSP Variants (4)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/time_windows/` | 32 | Yes (3 .py) | Expand: TW feasibility check, insertion |
| `variants/pickup_delivery/` | 30 | Yes (3 .py) | Expand: precedence + capacity constraints |
| `variants/asymmetric/` | 17 | Yes (3 .py) | Needs: ATSP formulation, Kanellakis-Papadimitriou |
| `variants/prize_collecting/` | 17 | Yes (3 .py) | Needs: penalty model, Balas formulation |

### Job Shop Variants (3)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/no_wait/` | 30 | Yes (3 .py) | Expand: no-wait constraint, delay matrix |
| `variants/flexible_tardiness/` | 24 | Yes (3 .py) | Expand: machine flexibility + tardiness |
| `variants/weighted_tardiness/` | 23 | Yes (3 .py) | Expand: weighted tardiness objective |

### Knapsack Variants (4)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/multidimensional/` | 30 | Yes (3 .py) | Expand: m-constraint formulation, surrogate relaxation |
| `variants/multiple/` | 30 | Yes (3 .py) | Expand: bin assignment + item selection |
| `variants/bounded/` | 17 | Yes (3 .py) | Needs: bounded quantity formulation |
| `variants/subset_sum/` | 17 | Yes (3 .py) | Needs: decision variant, pseudo-poly DP |

### Bin Packing Variants (3)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/two_dimensional/` | 32 | Yes (3 .py) | Expand: guillotine cuts, shelf algorithms |
| `variants/variable_size/` | 24 | Yes (3 .py) | Expand: multiple bin types, cost minimization |
| `variants/online/` | 17 | Yes (3 .py) | Needs: competitive ratio, NF/FF/BF analysis |

### Single Machine Variants (2)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/batch/` | 17 | Yes (3 .py) | Needs: batch formulation, family setup |
| `variants/preemptive/` | 16 | Yes (3 .py) | Needs: preemption model, SRPT optimality |

### Parallel Machine Variants (2)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/sdst/` | 31 | Yes (3 .py) | Expand: setup time matrix, modified LPT |
| `variants/unrelated_tardiness/` | 19 | Yes (3 .py) | Needs: Rm||sum wjTj formulation |

### RCPSP Variants (1)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/multi_mode/` | 16 | Yes (3 .py) | Needs: mode selection + resource trade-offs |

### VRPTW Variants (1)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/soft_time_windows/` | 16 | Yes (3 .py) | Needs: penalty functions, early/late costs |

### Cutting Stock Variants (1)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/two_dimensional/` | 25 | Yes (3 .py) | Needs: 2D cutting formulation, guillotine constraint |

### Facility Location Variants (1)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/capacitated/` | 30 | Yes (3 .py) | Expand: CFLP formulation, capacity constraints |

### p-Median Variants (1)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/capacitated/` | 17 | Yes (3 .py) | Needs: capacity constraints, CPMP formulation |

### Assignment Variants (3)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/generalized/` | 23 | Yes (3 .py) | Expand: GAP formulation, agents with capacities |
| `variants/quadratic/` | 17 | Yes (3 .py) | Needs: Koopmans-Beckmann, Gilmore-Lawler bound |
| `variants/max_weight_matching/` | 16 | Yes (3 .py) | Needs: weighted matching, Edmonds algorithm |

### Network Variants (3)

| Variant | Lines | Has Code? | Gap |
|---------|-------|-----------|-----|
| `variants/all_pairs/` (shortest_path) | 19 | No | Needs: Floyd-Warshall, Johnson's algorithm |
| `variants/steiner_tree/` (MST) | 16 | No | Needs: Steiner tree formulation, NP-hardness |
| `variants/min_cost_flow/` (max_flow) | 15 | No | Needs: cost + capacity model, network simplex |

### Variant Summary

- **Total variant folders:** 48
- **Variant folders with code:** 42
- **Variant folders without code:** 6 (open_shop, stochastic, hybrid, tardiness, lot_streaming, distributed in flow_shop; all_pairs, steiner_tree, min_cost_flow in network)
- **Current average line count:** 22.4 lines
- **Target average:** 60–100 lines

---

## Section C: Python Files to Audit (15 Priority Families)

### `1_scheduling/flow_shop/` (41 files)

**instance.py** (1): `instance.py`

**exact/** (3): `johnsons_rule.py`, `mip_formulation.py`, `branch_and_bound.py`

**heuristics/** (11): `palmers_slope.py`, `guptas_algorithm.py`, `dannenbring.py`, `cds.py`, `neh.py`, `lr_heuristic.py`, `nehkk.py`, `rajendran_ziegler.py`, `ra_heuristic.py`, `bonney_gundry.py`, `beam_search.py`

**metaheuristics/** (17): `iterated_greedy.py`, `simulated_annealing.py`, `genetic_algorithm.py`, `tabu_search.py`, `ant_colony.py`, `local_search.py`, `vns.py`, `memetic_algorithm.py`, `particle_swarm.py`, `differential_evolution.py`, `scatter_search.py`, `eda.py`, `harmony_search.py`, `bbo.py`, `bee_colony.py`, `tlbo.py`, `whale_optimization.py`

**tests/** (7): `test_flow_shop.py`, `test_new_algorithms.py`, `test_ts_aco_sdst.py`, `test_vns_ma_eda.py`, `test_pso_de_ss_ra.py`, `test_rz_hs_bbo.py`, `test_nehkk_abc_beam.py`, `test_bg_tlbo_woa.py`

**Other** (1): `benchmark_runner.py`

### `1_scheduling/job_shop/` (15 files)

**instance.py**, **heuristics/** (2): `dispatching_rules.py`, `shifting_bottleneck.py`

**metaheuristics/** (5): `simulated_annealing.py`, `tabu_search.py`, `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `vns.py`

**tests/** (5 + conftest): `test_job_shop.py`, `test_job_shop_ga.py`, `test_job_shop_ig.py`, `test_job_shop_vns.py`, `test_jsp_ls.py`, `conftest.py`

### `1_scheduling/single_machine/` (18 files)

**instance.py**, **exact/** (2): `dynamic_programming.py`, `branch_and_bound.py`

**heuristics/** (3): `dispatching_rules.py`, `moores_algorithm.py`, `apparent_tardiness_cost.py`

**metaheuristics/** (6): `simulated_annealing.py`, `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `tabu_search.py`, `vns.py`

**tests/** (6): `test_single_machine.py`, `test_sm_ga.py`, `test_sm_ig.py`, `test_sm_ls.py`, `test_sm_tabu_search.py`, `test_sm_vns.py`

### `1_scheduling/parallel_machine/` (17 files)

**instance.py**, **exact/** (1): `mip_makespan.py`

**heuristics/** (3): `lpt.py`, `multifit.py`, `list_scheduling.py`

**metaheuristics/** (6): `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `simulated_annealing.py`, `tabu_search.py`, `vns.py`

**tests/** (6): `test_parallel_machine.py`, `test_pm_ig.py`, `test_pm_ls.py`, `test_pm_sa.py`, `test_pm_ts.py`, `test_pm_vns.py`

### `1_scheduling/nurse_scheduling/` (3 files)

**instance.py**, **heuristics/** (1): `greedy_roster.py`

**tests/** (1): `test_nurse_scheduling.py`

### `2_routing/tsp/` (18 files)

**instance.py**, **exact/** (2): `held_karp.py`, `branch_and_bound.py`

**heuristics/** (3): `nearest_neighbor.py`, `cheapest_insertion.py`, `greedy.py`

**metaheuristics/** (7): `local_search.py`, `simulated_annealing.py`, `genetic_algorithm.py`, `tabu_search.py`, `ant_colony.py`, `iterated_greedy.py`, `vns.py`

**tests/** (5): `test_tsp.py`, `test_tsp_aco.py`, `test_tsp_ig.py`, `test_tsp_ts.py`, `test_tsp_vns.py`

### `2_routing/cvrp/` (16 files)

**instance.py**, **heuristics/** (2): `clarke_wright.py`, `sweep.py`

**metaheuristics/** (7): `simulated_annealing.py`, `genetic_algorithm.py`, `local_search.py`, `tabu_search.py`, `ant_colony.py`, `iterated_greedy.py`, `vns.py`

**tests/** (6): `test_cvrp.py`, `test_cvrp_aco.py`, `test_cvrp_ig.py`, `test_cvrp_ls.py`, `test_cvrp_ts.py`, `test_cvrp_vns.py`

### `2_routing/vrptw/` (15 files)

**instance.py**, **heuristics/** (1): `solomon_insertion.py`

**metaheuristics/** (7): `simulated_annealing.py`, `genetic_algorithm.py`, `local_search.py`, `tabu_search.py`, `ant_colony.py`, `iterated_greedy.py`, `vns.py`

**tests/** (6): `test_vrptw.py`, `test_vrptw_aco.py`, `test_vrptw_ig.py`, `test_vrptw_ls.py`, `test_vrptw_ts.py`, `test_vrptw_vns.py`

### `3_packing_cutting/knapsack/` (16 files)

**instance.py**, **exact/** (2): `dynamic_programming.py`, `branch_and_bound.py`

**heuristics/** (1): `greedy.py`

**metaheuristics/** (6): `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `simulated_annealing.py`, `tabu_search.py`, `vns.py`

**tests/** (6): `test_knapsack.py`, `test_knapsack_ls.py`, `test_knapsack_sa.py`, `test_knapsack_ts.py`, `test_knapsack_vns.py`, `test_kp_ig.py`

### `3_packing_cutting/bin_packing/` (14 files)

**instance.py**, **heuristics/** (1): `first_fit.py`

**metaheuristics/** (6): `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `simulated_annealing.py`, `tabu_search.py`, `vns.py`

**tests/** (6): `test_bin_packing.py`, `test_bpp_ig.py`, `test_bpp_ls.py`, `test_bpp_sa.py`, `test_bpp_ts.py`, `test_bpp_vns.py`

### `3_packing_cutting/cutting_stock/` (16 files)

**instance.py**, **heuristics/** (1): `greedy_csp.py`

**metaheuristics/** (7 incl. __init__): `__init__.py`, `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `simulated_annealing.py`, `tabu_search.py`, `vns.py`

**tests/** (7): `test_cutting_stock.py`, `test_csp_ga.py`, `test_csp_ig.py`, `test_csp_ls.py`, `test_csp_sa.py`, `test_csp_ts.py`, `test_csp_vns.py`

### `5_location_covering/facility_location/` (14 files)

**instance.py**, **heuristics/** (1): `greedy_facility.py`

**metaheuristics/** (6): `simulated_annealing.py`, `genetic_algorithm.py`, `iterated_greedy.py`, `local_search.py`, `tabu_search.py`, `vns.py`

**tests/** (6): `test_facility_location.py`, `test_fl_ga.py`, `test_fl_ig.py`, `test_fl_ls.py`, `test_fl_ts.py`, `test_fl_vns.py`

### `6_network_flow_design/shortest_path/` (4 files)

**instance.py**, **exact/** (2): `dijkstra.py`, `bellman_ford.py`

**tests/** (1): `test_shortest_path.py`

### `9_uncertainty_modeling/two_stage_sp/` (4 files)

**instance.py**, **heuristics/** (1): `deterministic_equivalent.py`

**metaheuristics/** (1): `sample_average.py`

**tests/** (1): `test_two_stage_sp.py`

### `9_uncertainty_modeling/newsvendor/` (4 files)

**instance.py**, **exact/** (1): `critical_fractile.py`

**heuristics/** (1): `multi_product.py`

**tests/** (1): `test_newsvendor.py`

### Audit Summary

| Family | .py Files | Test Files | Total |
|--------|-----------|------------|-------|
| flow_shop | 34 | 7 | 41 |
| job_shop | 9 | 6 | 15 |
| single_machine | 12 | 6 | 18 |
| parallel_machine | 11 | 6 | 17 |
| nurse_scheduling | 2 | 1 | 3 |
| tsp | 13 | 5 | 18 |
| cvrp | 10 | 6 | 16 |
| vrptw | 9 | 6 | 15 |
| knapsack | 10 | 6 | 16 |
| bin_packing | 8 | 6 | 14 |
| cutting_stock | 9 | 7 | 16 |
| facility_location | 8 | 6 | 14 |
| shortest_path | 3 | 1 | 4 |
| two_stage_sp | 3 | 1 | 4 |
| newsvendor | 3 | 1 | 4 |
| **TOTAL** | **144** | **71** | **215** |

# CLAUDE.md

## Project Overview

Operations Research problem repository featuring mathematical formulations, algorithm implementations, and benchmarks. Educational/research-focused with academic rigor (references, complexity analysis, scheduling notation). Currently implementing **Phase 1** (Scheduling), **Phase 2** (Routing), **Phase 3** (Packing & Cutting), **Phase 4** (Location & Network), and **Phase 5** (Stochastic & Robust Optimization).

**Author**: Mohammad Ghafourian Nasiri
**License**: MIT
**Python**: 3.10+ required (uses `from __future__ import annotations`, `|` union syntax)

## Repository Structure

```
MathematicalModeling/
├── CLAUDE.md
├── README.md              # Project overview + taxonomy
├── CONTRIBUTING.md         # Guidelines for adding problems/algorithms
├── requirements.txt        # numpy, scipy, matplotlib, pytest, pandas
├── docs/
│   └── taxonomy.md         # Full problem classification (9 families)
├── shared/                 # Reusable infrastructure
│   ├── parsers/
│   │   └── taillard_parser.py   # Taillard benchmark downloader/parser
│   ├── exact/              # (placeholder for shared exact method components)
│   ├── metaheuristics/     # (placeholder for generic metaheuristic frameworks)
│   └── visualization/      # (placeholder for visualization tools)
└── problems/
    └── scheduling/
        ├── flow_shop/      # FULLY IMPLEMENTED
        │   ├── instance.py             # FlowShopInstance, FlowShopSolution dataclasses
        │   ├── benchmark_runner.py     # CLI for evaluating algorithms on Taillard instances
        │   ├── exact/
        │   │   ├── johnsons_rule.py    # Optimal for F2||Cmax, O(n log n)
        │   │   ├── mip_formulation.py  # SciPy HiGHS + OR-Tools CP-SAT solvers
        │   │   └── branch_and_bound.py # Taillard lower bound, NEH warm-start
        │   ├── heuristics/
        │   │   ├── palmers_slope.py    # Slope index, O(n*m + n log n)
        │   │   ├── guptas_algorithm.py # Priority-based, O(n*m + n log n)
        │   │   ├── dannenbring.py      # Rapid Access weighted Johnson's, O(n*m + n log n)
        │   │   ├── cds.py             # Campbell-Dudek-Smith, O(m * n log n)
        │   │   ├── neh.py             # Nawaz-Enscore-Ham, O(n^2 * m), best constructive
        │   │   └── lr_heuristic.py    # Multi-candidate greedy, O(n^3 * m)
        │   ├── metaheuristics/
        │   │   ├── iterated_greedy.py      # Ruiz & Stuetzle (2007), state-of-the-art
        │   │   ├── simulated_annealing.py  # Osman & Potts (1989), classic SA
        │   │   ├── genetic_algorithm.py    # Reeves (1995), OX crossover + insertion mutation
        │   │   ├── tabu_search.py          # Nowicki & Smutnicki (1996), fast TS
        │   │   ├── ant_colony.py           # Stützle (1998), ACO with MMAS pheromone bounds
        │   │   └── local_search.py         # Swap, insertion, or-opt, VND neighborhoods
        │   ├── variants/
        │   │   ├── no_wait/           # Fm | prmu, no-wait | Cmax
        │   │   │   ├── instance.py    # NoWaitFlowShopInstance, delay matrix computation
        │   │   │   ├── heuristics.py  # NN, NEH-NW, Gangadharan-Rajendran
        │   │   │   ├── metaheuristics.py  # Iterated Greedy for no-wait
        │   │   │   └── README.md
        │   │   ├── blocking/          # Fm | prmu, blocking | Cmax
        │   │   │   ├── instance.py    # BlockingFlowShopInstance, departure times
        │   │   │   ├── heuristics.py  # NEH-B, Profile Fitting
        │   │   │   ├── metaheuristics.py  # Iterated Greedy for blocking
        │   │   │   └── README.md
        │   │   └── setup_times/       # Fm | prmu, Ssd | Cmax
        │   │       ├── instance.py    # SDSTFlowShopInstance, setup time matrices
        │   │       ├── heuristics.py  # NEH-SDST, GRASP-SDST
        │   │       ├── metaheuristics.py  # Iterated Greedy for SDST
        │   │       └── README.md
        │   └── tests/
        │       ├── test_flow_shop.py       # 57 tests, original PFSP algorithms
        │       ├── test_new_algorithms.py  # 38 tests, new algorithms + variants
        │       └── test_ts_aco_sdst.py     # 35 tests, TS, ACO, SDST variant
        ├── parallel_machine/ # FULLY IMPLEMENTED (7 Python files, 43-test suite)
        │   ├── instance.py              # ParallelMachineInstance (Pm, Qm, Rm)
        │   ├── exact/
        │   │   └── mip_makespan.py      # MIP formulation via SciPy HiGHS
        │   ├── heuristics/
        │   │   ├── lpt.py               # LPT (4/3 approx) + SPT for ΣCj
        │   │   ├── multifit.py          # MULTIFIT (1.22 approx), FFD + binary search
        │   │   └── list_scheduling.py   # Greedy list scheduling (2-1/m approx)
        │   ├── metaheuristics/
        │   │   └── genetic_algorithm.py # GA with integer-vector encoding
        │   └── tests/
        │       └── test_parallel_machine.py  # 43 tests, 8 test classes
        ├── single_machine/   # FULLY IMPLEMENTED (7 Python files, 55-test suite)
        │   ├── instance.py              # SingleMachineInstance, objective functions
        │   ├── exact/
        │   │   ├── dynamic_programming.py  # Bitmask DP for 1||ΣTj, O(2^n * n)
        │   │   └── branch_and_bound.py     # B&B for 1||ΣwjTj, ATC warm-start
        │   ├── heuristics/
        │   │   ├── dispatching_rules.py    # SPT, WSPT, EDD, LPT — O(n log n)
        │   │   ├── moores_algorithm.py     # 1||ΣUj — O(n log n)
        │   │   └── apparent_tardiness_cost.py  # ATC for 1||ΣwjTj — O(n²)
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py  # SA for ΣwjTj and ΣTj
        │   └── tests/
        │       └── test_single_machine.py  # 55 tests, 12 test classes
        ├── job_shop/         # FULLY IMPLEMENTED (5 Python files, 41-test suite)
        │   ├── instance.py              # JobShopInstance, ft06/ft10 benchmarks
        │   ├── heuristics/
        │   │   ├── dispatching_rules.py # SPT, LPT, MWR, LWR, FIFO (Giffler-Thompson)
        │   │   └── shifting_bottleneck.py # Adams-Balas-Zawack (1988)
        │   ├── metaheuristics/
        │   │   ├── simulated_annealing.py # Critical-path neighborhood SA
        │   │   └── tabu_search.py         # N1 neighborhood with aspiration
        │   └── tests/
        │       └── test_job_shop.py       # 41 tests, 7 test classes
        ├── flexible_job_shop/ # FULLY IMPLEMENTED (4 Python files, 37-test suite)
        │   ├── instance.py              # FlexibleJobShopInstance (total/partial)
        │   ├── heuristics/
        │   │   ├── dispatching_rules.py # SPT/LPT/MWR/LWR + ECT/SPT machine selection
        │   │   └── hierarchical.py      # Route-then-sequence decomposition
        │   ├── metaheuristics/
        │   │   └── genetic_algorithm.py # Pezzella-style integrated encoding
        │   └── tests/
        │       └── test_fjsp.py         # 37 tests, 6 test classes
        └── rcpsp/            # FULLY IMPLEMENTED (4 Python files, 35-test suite)
            ├── instance.py              # RCPSPInstance, precedence DAG, resources
            ├── heuristics/
            │   ├── serial_sgs.py        # Serial SGS (LFT/EST/MTS/GRPW rules)
            │   └── parallel_sgs.py      # Parallel SGS (non-delay schedules)
            ├── metaheuristics/
            │   └── genetic_algorithm.py # Activity-list GA (Hartmann 1998)
            └── tests/
                └── test_rcpsp.py        # 35 tests, 7 test classes
    └── routing/
        ├── tsp/              # FULLY IMPLEMENTED (8 Python files, 55-test suite)
        │   ├── instance.py              # TSPInstance, TSPSolution, benchmark instances
        │   ├── exact/
        │   │   ├── held_karp.py         # Held-Karp DP, O(2^n * n^2), optimal for n <= 23
        │   │   └── branch_and_bound.py  # B&B with 1-tree lower bound, NN warm-start
        │   ├── heuristics/
        │   │   ├── nearest_neighbor.py  # NN + multi-start, O(n^2)
        │   │   ├── cheapest_insertion.py # Cheapest/farthest/nearest insertion, O(n^3)
        │   │   └── greedy.py            # Greedy nearest-edge, O(n^2 log n)
        │   ├── metaheuristics/
        │   │   ├── local_search.py      # 2-opt, Or-opt, VND neighborhoods
        │   │   ├── simulated_annealing.py # SA with 2-opt moves
        │   │   └── genetic_algorithm.py # OX crossover, swap mutation
        │   └── tests/
        │       └── test_tsp.py          # 55 tests, 10 test classes
        ├── cvrp/             # FULLY IMPLEMENTED (5 Python files, 41-test suite)
        │   ├── instance.py              # CVRPInstance, CVRPSolution, validation
        │   ├── heuristics/
        │   │   ├── clarke_wright.py     # Clarke-Wright savings, O(n^2 log n)
        │   │   └── sweep.py            # Angular sweep + multi-start, O(n log n)
        │   ├── metaheuristics/
        │   │   ├── simulated_annealing.py # Relocate/swap/2-opt* neighborhoods
        │   │   └── genetic_algorithm.py # Giant-tour encoding, OX crossover
        │   └── tests/
        │       └── test_cvrp.py         # 41 tests, 8 test classes
        └── vrptw/            # FULLY IMPLEMENTED (4 Python files, 31-test suite)
            ├── instance.py              # VRPTWInstance, VRPTWSolution, validation
            ├── heuristics/
            │   └── solomon_insertion.py # Solomon I1 + nearest neighbor TW
            ├── metaheuristics/
            │   ├── simulated_annealing.py # Relocate/swap with TW feasibility
            │   └── genetic_algorithm.py # Giant-tour encoding, TW-aware split
            └── tests/
                └── test_vrptw.py        # 31 tests, 8 test classes
    └── packing/
        ├── knapsack/         # FULLY IMPLEMENTED (5 Python files, 37-test suite)
        │   ├── instance.py              # KnapsackInstance, KnapsackSolution, validation
        │   ├── exact/
        │   │   ├── dynamic_programming.py # Bitmask DP, O(n*W) pseudo-polynomial
        │   │   └── branch_and_bound.py    # B&B with LP relaxation bound
        │   ├── heuristics/
        │   │   └── greedy.py              # Value-density, max-value, combined (1/2-approx)
        │   ├── metaheuristics/
        │   │   └── genetic_algorithm.py   # Binary encoding with repair operator
        │   └── tests/
        │       └── test_knapsack.py       # 37 tests, 8 test classes
        ├── bin_packing/      # FULLY IMPLEMENTED (3 Python files, 29-test suite)
        │   ├── instance.py              # BinPackingInstance, L1/L2 lower bounds
        │   ├── heuristics/
        │   │   └── first_fit.py         # FF, FFD (11/9 approx), BFD
        │   ├── metaheuristics/
        │   │   └── genetic_algorithm.py # Permutation encoding, FF decoder
        │   └── tests/
        │       └── test_bin_packing.py  # 29 tests, 7 test classes
        └── cutting_stock/    # FULLY IMPLEMENTED (2 Python files, 21-test suite)
            ├── instance.py              # CuttingStockInstance, pattern validation
            ├── heuristics/
            │   └── greedy_csp.py        # Greedy largest-first, FFD-based
            └── tests/
                └── test_cutting_stock.py # 21 tests, 6 test classes
    └── location_network/
        ├── facility_location/ # FULLY IMPLEMENTED (3 Python files, 16-test suite)
        │   ├── instance.py              # FacilityLocationInstance, validation
        │   ├── heuristics/
        │   │   └── greedy_facility.py   # Greedy add, greedy drop
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py # Toggle/swap with Boltzmann acceptance
        │   └── tests/
        │       └── test_facility_location.py # 16 tests, 6 test classes
        ├── p_median/         # FULLY IMPLEMENTED (2 Python files, 13-test suite)
        │   ├── instance.py              # PMedianInstance, validation
        │   ├── heuristics/
        │   │   └── greedy_pmedian.py    # Greedy, Teitz-Bart interchange
        │   └── tests/
        │       └── test_p_median.py     # 13 tests, 5 test classes
        ├── shortest_path/    # FULLY IMPLEMENTED (3 Python files, 21-test suite)
        │   ├── instance.py              # ShortestPathInstance, edge/matrix creation
        │   ├── exact/
        │   │   ├── dijkstra.py          # Dijkstra O((V+E)logV), non-negative weights
        │   │   └── bellman_ford.py      # Bellman-Ford O(VE), negative cycle detection
        │   └── tests/
        │       └── test_shortest_path.py # 21 tests, 7 test classes
        ├── max_flow/         # FULLY IMPLEMENTED (2 Python files, 16-test suite)
        │   ├── instance.py              # MaxFlowInstance, capacity matrix, validation
        │   ├── exact/
        │   │   └── edmonds_karp.py      # Edmonds-Karp O(VE^2), min-cut extraction
        │   └── tests/
        │       └── test_max_flow.py     # 16 tests, 5 test classes
        ├── min_spanning_tree/ # FULLY IMPLEMENTED (2 Python files, 16-test suite)
        │   ├── instance.py              # MSTInstance, undirected graph, validation
        │   ├── exact/
        │   │   └── mst_algorithms.py    # Kruskal O(E log E), Prim O(E log V)
        │   └── tests/
        │       └── test_mst.py          # 16 tests, 5 test classes
        └── assignment/       # FULLY IMPLEMENTED (3 Python files, 17-test suite)
            ├── instance.py              # AssignmentInstance, cost matrix
            ├── exact/
            │   └── hungarian.py         # Hungarian (Kuhn-Munkres) O(n^3)
            ├── heuristics/
            │   └── greedy_assignment.py # Greedy min-cost assignment O(n^2)
            └── tests/
                └── test_assignment.py   # 17 tests, 6 test classes
    └── stochastic_robust/
        ├── newsvendor/           # FULLY IMPLEMENTED (3 Python files, 13-test suite)
        │   ├── instance.py              # NewsvendorInstance, critical fractile
        │   ├── exact/
        │   │   └── critical_fractile.py # Critical fractile + grid search
        │   ├── heuristics/
        │   │   └── multi_product.py     # Marginal allocation, independent+scale
        │   └── tests/
        │       └── test_newsvendor.py   # 13 tests, 3 test classes
        ├── two_stage_sp/         # FULLY IMPLEMENTED (3 Python files, 10-test suite)
        │   ├── instance.py              # TwoStageSPInstance, newsvendor/capacity planning
        │   ├── heuristics/
        │   │   └── deterministic_equivalent.py  # Extensive form LP, EV solution
        │   ├── metaheuristics/
        │   │   └── sample_average.py    # SAA with replications
        │   └── tests/
        │       └── test_two_stage_sp.py # 10 tests, 4 test classes
        ├── robust_shortest_path/ # FULLY IMPLEMENTED (3 Python files, 13-test suite)
        │   ├── instance.py              # RobustSPInstance, scenario weights
        │   ├── exact/
        │   │   └── minmax_cost.py       # Label-setting + enumeration
        │   ├── heuristics/
        │   │   └── minmax_regret.py     # Regret enumeration, midpoint
        │   └── tests/
        │       └── test_robust_sp.py    # 13 tests, 4 test classes
        ├── stochastic_knapsack/  # FULLY IMPLEMENTED (3 Python files, 11-test suite)
        │   ├── instance.py              # StochasticKnapsackInstance, random weights
        │   ├── heuristics/
        │   │   └── greedy_stochastic.py # Mean-weight + chance-constrained greedy
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py # Flip-bit SA with infeasibility penalty
        │   └── tests/
        │       └── test_stochastic_knapsack.py # 11 tests, 3 test classes
        ├── chance_constrained_fl/ # FULLY IMPLEMENTED (3 Python files, 11-test suite)
        │   ├── instance.py              # CCFLInstance, stochastic demands
        │   ├── heuristics/
        │   │   └── greedy_ccfl.py       # Greedy open, mean-demand greedy
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py # Toggle/swap with violation penalty
        │   └── tests/
        │       └── test_ccfl.py         # 11 tests, 3 test classes
        ├── robust_portfolio/     # FULLY IMPLEMENTED (3 Python files, 14-test suite)
        │   ├── instance.py              # RobustPortfolioInstance, Markowitz
        │   ├── exact/
        │   │   └── quadratic_solver.py  # Mean-variance QP + robust SOCP
        │   ├── heuristics/
        │   │   └── equal_weight.py      # Equal-weight, min-variance, max-return
        │   └── tests/
        │       └── test_portfolio.py    # 14 tests, 3 test classes
        ├── stochastic_vrp/       # FULLY IMPLEMENTED (3 Python files, 13-test suite)
        │   ├── instance.py              # StochasticVRPInstance, demand uncertainty
        │   ├── heuristics/
        │   │   └── chance_constrained_cw.py # CC Clarke-Wright savings
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py # Relocate/swap/2-opt with recourse
        │   └── tests/
        │       └── test_stochastic_vrp.py # 13 tests, 3 test classes
        ├── robust_scheduling/    # FULLY IMPLEMENTED (3 Python files, 13-test suite)
        │   ├── instance.py              # RobustSchedulingInstance, uncertain p_j
        │   ├── heuristics/
        │   │   └── minmax_regret_heuristics.py # Midpoint/scenario/worst-case WSPT
        │   ├── metaheuristics/
        │   │   └── simulated_annealing.py # SA for min-max regret ΣwjCj
        │   └── tests/
        │       └── test_robust_scheduling.py # 13 tests, 3 test classes
        └── dro/                  # FULLY IMPLEMENTED (3 Python files, 12-test suite)
            ├── instance.py              # DROInstance, ambiguity sets
            ├── exact/
            │   └── wasserstein_dro.py   # Wasserstein LP + nominal LP
            ├── heuristics/
            │   └── moment_dro.py        # Moment-based DRO with inner LP
            └── tests/
                └── test_dro.py          # 12 tests, 4 test classes
```

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all scheduling tests (341 tests)
python -m pytest problems/scheduling/ -v

# Run all flow shop tests (130 tests)
python -m pytest problems/scheduling/flow_shop/tests/ -v

# Run parallel machine tests (43 tests)
python -m pytest problems/scheduling/parallel_machine/tests/ -v

# Run single machine tests (55 tests)
python -m pytest problems/scheduling/single_machine/tests/ -v

# Run job shop tests (41 tests)
python -m pytest problems/scheduling/job_shop/tests/ -v

# Run flexible job shop tests (37 tests)
python -m pytest problems/scheduling/flexible_job_shop/tests/ -v

# Run RCPSP tests (35 tests)
python -m pytest problems/scheduling/rcpsp/tests/ -v

# Run specific test class
python -m pytest problems/scheduling/flow_shop/tests/test_flow_shop.py::TestNEH -v

# Run benchmarks (example: 20 jobs, 5 machines)
python problems/scheduling/flow_shop/benchmark_runner.py --class 20_5 --all

# Run all routing tests (127 tests)
python -m pytest problems/routing/ -v

# Run TSP tests (55 tests)
python -m pytest problems/routing/tsp/tests/ -v

# Run CVRP tests (41 tests)
python -m pytest problems/routing/cvrp/tests/ -v

# Run VRPTW tests (31 tests)
python -m pytest problems/routing/vrptw/tests/ -v

# Run all packing tests (87 tests)
python -m pytest problems/packing/ -v

# Run knapsack tests (37 tests)
python -m pytest problems/packing/knapsack/tests/ -v

# Run bin packing tests (29 tests)
python -m pytest problems/packing/bin_packing/tests/ -v

# Run cutting stock tests (21 tests)
python -m pytest problems/packing/cutting_stock/tests/ -v

# Run all location/network tests (99 tests)
python -m pytest problems/location_network/ -v

# Run facility location tests (16 tests)
python -m pytest problems/location_network/facility_location/tests/ -v

# Run p-median tests (13 tests)
python -m pytest problems/location_network/p_median/tests/ -v

# Run shortest path tests (21 tests)
python -m pytest problems/location_network/shortest_path/tests/ -v

# Run max flow tests (16 tests)
python -m pytest problems/location_network/max_flow/tests/ -v

# Run MST tests (16 tests)
python -m pytest problems/location_network/min_spanning_tree/tests/ -v

# Run assignment tests (17 tests)
python -m pytest problems/location_network/assignment/tests/ -v

# Run all stochastic/robust tests (110 tests)
python -m pytest problems/stochastic_robust/ -v

# Run newsvendor tests (13 tests)
python -m pytest problems/stochastic_robust/newsvendor/tests/ -v

# Run two-stage SP tests (10 tests)
python -m pytest problems/stochastic_robust/two_stage_sp/tests/ -v

# Run robust shortest path tests (13 tests)
python -m pytest problems/stochastic_robust/robust_shortest_path/tests/ -v

# Run stochastic knapsack tests (11 tests)
python -m pytest problems/stochastic_robust/stochastic_knapsack/tests/ -v

# Run chance-constrained FL tests (11 tests)
python -m pytest problems/stochastic_robust/chance_constrained_fl/tests/ -v

# Run robust portfolio tests (14 tests)
python -m pytest problems/stochastic_robust/robust_portfolio/tests/ -v

# Run stochastic VRP tests (13 tests)
python -m pytest problems/stochastic_robust/stochastic_vrp/tests/ -v

# Run robust scheduling tests (13 tests)
python -m pytest problems/stochastic_robust/robust_scheduling/tests/ -v

# Run DRO tests (12 tests)
python -m pytest problems/stochastic_robust/dro/tests/ -v
```

### Dependencies

**Core**: numpy (>=1.24), scipy (>=1.10), pandas (>=2.0), matplotlib (>=3.7), pytest (>=7.3)
**Optional**: ortools (>=9.6), gurobipy (>=10.0), pyomo (>=6.5)

## Flow Shop Problem Family

### Standard Permutation Flow Shop (Fm | prmu | Cmax)

All n jobs must be processed on m machines in the same order. The objective
is to find the job permutation that minimizes the makespan (completion time
of the last job on the last machine). NP-hard for m >= 3.

**Exact methods:**
- Johnson's Rule — optimal for F2||Cmax in O(n log n)
- Branch & Bound — Taillard (1993) lower bounds, NEH warm-start, practical for n <= ~20
- MIP — position-based formulation via SciPy HiGHS or OR-Tools CP-SAT

**Constructive heuristics** (fast, single-pass):
- Palmer's Slope Index (1965) — weakest, O(n*m + n log n)
- Gupta's Algorithm (1971) — bottleneck-aware priority, O(n*m + n log n)
- Dannenbring's Rapid Access (1977) — weighted Johnson's reduction, O(n*m + n log n)
- CDS (Campbell-Dudek-Smith, 1970) — m-1 virtual 2-machine sub-problems, O(m * n log n)
- NEH (Nawaz-Enscore-Ham, 1983) — best constructive heuristic, O(n^2 * m)
- LR (Liu & Reeves, 2001) — multi-candidate composite index, O(n^3 * m)

**Metaheuristics** (iterative improvement):
- Local Search — swap, insertion, or-opt, VND neighborhoods
- Simulated Annealing (Osman & Potts, 1989) — Boltzmann acceptance, insertion neighborhood
- Genetic Algorithm (Reeves, 1995) — OX crossover, insertion mutation, steady-state
- Tabu Search (Nowicki & Smutnicki, 1996) — short-term memory, aspiration criterion
- Ant Colony Optimization (Stützle, 1998) — pheromone trails, MMAS bounds, elitist update
- Iterated Greedy (Ruiz & Stuetzle, 2007) — state-of-the-art destroy-and-repair

### No-Wait Flow Shop (Fm | prmu, no-wait | Cmax)

Jobs cannot wait between consecutive machines — processing must be contiguous.
Reduces to an asymmetric TSP on the inter-job delay matrix. NP-hard for m >= 3.

**Applications:** Steel manufacturing, chemical processing, food processing.

**Algorithms:** Nearest Neighbor, NEH-NW, Gangadharan-Rajendran heuristic,
Iterated Greedy (IG-NW).

### Blocking Flow Shop (Fm | prmu, blocking | Cmax)

No intermediate buffers between machines — a completed job blocks its machine
until the next machine is available. Uses departure time recursion instead of
standard completion times. NP-hard for m >= 3.

**Applications:** Manufacturing with limited buffers, robotic cells, paint shops.

**Algorithms:** NEH-B, Profile Fitting, Iterated Greedy (IG-B).

### Sequence-Dependent Setup Times Flow Shop (Fm | prmu, Ssd | Cmax)

Setup times depend on both the current and preceding job on each machine.
Models real-world changeover times that vary by product sequence. NP-hard
for m >= 2.

**Applications:** Printing (color changeovers), chemical processing, automotive
manufacturing, semiconductor fabrication, food processing.

**Algorithms:** NEH-SDST (setup-aware workload sorting), GRASP-SDST (randomized
greedy with local search), Iterated Greedy (IG-SDST).

## Parallel Machine Problem Family

### Problem Definition (Pm | β | γ)

n jobs must be assigned to m parallel machines. Three machine environments:
- **Identical (Pm)**: all machines have equal speed
- **Uniform (Qm)**: machine i has speed s_i
- **Unrelated (Rm)**: processing time p_ij depends on both job and machine

Primary objective: Cmax (makespan). NP-hard even for P2||Cmax (reduces to PARTITION).

**Exact methods:**
- MIP — assignment-based formulation via SciPy HiGHS

**Constructive heuristics:**
- LPT (Longest Processing Time) — 4/3 - 1/(3m) approximation for Cmax
- SPT (Shortest Processing Time) — optimal for Pm||ΣCj with round-robin
- MULTIFIT (Coffman, Garey & Johnson, 1978) — FFD + binary search, 1.22 approximation
- List Scheduling (Graham, 1966) — 2 - 1/m approximation

**Metaheuristics:**
- Genetic Algorithm — integer-vector encoding, uniform crossover, load-balancing LS

## Single Machine Problem Family

### Problem Definition (1 | β | γ)

n jobs processed on one machine. The schedule is fully determined by the
processing order. Multiple objectives supported, ranging from polynomial-time
solvable to strongly NP-hard.

### Tractable Objectives (polynomial-time optimal rules)

| Objective | Rule | Complexity | Reference |
|-----------|------|------------|-----------|
| 1 \|\| ΣCj | SPT (Shortest Processing Time) | O(n log n) | Conway et al. (1967) |
| 1 \|\| ΣwjCj | WSPT (Smith's Rule: sort by pj/wj) | O(n log n) | Smith (1956) |
| 1 \|\| Lmax | EDD (Earliest Due Date) | O(n log n) | Jackson (1955) |
| 1 \|\| ΣUj | Moore's Algorithm | O(n log n) | Moore (1968) |

### NP-Hard Objectives

| Objective | Methods | Reference |
|-----------|---------|-----------|
| 1 \|\| ΣTj | DP (bitmask, exact for n ≤ 20), SA | Lawler (1977), Du & Leung (1990) |
| 1 \|\| ΣwjTj | B&B (ATC warm-start), ATC heuristic, SA | Potts & Van Wassenhove (1985) |

**Constructive heuristics:**
- SPT, WSPT, EDD, LPT — optimal dispatching rules for tractable objectives
- Moore's Algorithm — greedy EDD-based with longest-job removal for ΣUj
- ATC (Apparent Tardiness Cost) — composite dispatching rule for ΣwjTj, O(n²)

**Exact methods:**
- Dynamic Programming — bitmask DP for 1 || ΣTj, O(2^n × n), practical for n ≤ 20
- Branch and Bound — DFS with EDD lower bounds for 1 || ΣwjTj, ATC warm-start

**Metaheuristics:**
- Simulated Annealing — swap/insertion neighborhood for ΣwjTj and ΣTj

## Job Shop Problem Family

### Problem Definition (Jm | | Cmax)

n jobs, each consisting of a sequence of operations on specific machines.
Different jobs may visit machines in different orders (job-specific routing).
The disjunctive graph (Roy & Sussmann, 1964) is the central data structure.

Complexity: NP-hard even for J2||Cmax. The ft10 (10x10) instance remained
unsolved for 26 years (1963-1989).

**Constructive heuristics:**
- Dispatching Rules (SPT, LPT, MWR, LWR, FIFO) — Giffler & Thompson (1960) active schedule generation
- Shifting Bottleneck — Adams, Balas & Zawack (1988), iterative single-machine sub-problems

**Metaheuristics:**
- Simulated Annealing — Van Laarhoven et al. (1992), critical-path neighborhood
- Tabu Search — Nowicki & Smutnicki (1996), N1 neighborhood with aspiration criterion

**Benchmark instances:** ft06 (6x6, optimal=55), ft10 (10x10, optimal=930)

## Flexible Job Shop Problem Family

### Problem Definition (FJm | | Cmax)

Extends JSP: each operation can be processed on any machine from a set of
eligible machines. Introduces routing (machine assignment) + sequencing.
Total FJSP: all machines eligible; Partial FJSP: subsets of eligible machines.

**Constructive heuristics:**
- Dispatching Rules — SPT/LPT/MWR/LWR priority + ECT/SPT machine assignment
- Hierarchical — Route-then-sequence decomposition (Brandimarte, 1993)

**Metaheuristics:**
- Genetic Algorithm — Pezzella et al. (2008), integrated routing+sequencing encoding

## RCPSP Problem Family

### Problem Definition (PS | prec | Cmax)

n activities with precedence constraints and renewable resource requirements.
Schedule all activities to minimize project duration. Activities 0 (source)
and n+1 (sink) are dummies.

Strongly NP-hard even with 2 resource types.

**Schedule Generation Schemes (SGS):**
- Serial SGS — schedule one activity at a time, generates active schedules
- Parallel SGS — at each time step schedule all feasible, generates non-delay schedules

**Priority rules:** LFT (Latest Finish Time), EST (Earliest Start Time),
MTS (Most Total Successors), GRPW (Greatest Rank Positional Weight)

**Metaheuristics:**
- Genetic Algorithm — Hartmann (1998), activity-list encoding with Serial SGS decoder

## TSP Problem Family

### Problem Definition (TSP / ATSP)

Given n cities and pairwise distances, find the shortest Hamiltonian cycle
(tour) visiting each city exactly once and returning to the start.

Complexity: NP-hard (Karp, 1972). Symmetric TSP for undirected graphs,
Asymmetric TSP (ATSP) for directed graphs.

**Exact methods:**
- Held-Karp DP (1962) — O(2^n × n^2), optimal for n ≤ 23
- Branch and Bound — 1-tree lower bound, NN warm-start, practical for n ≤ ~25

**Constructive heuristics:**
- Nearest Neighbor — greedy O(n^2), multi-start variant
- Insertion heuristics — cheapest, farthest, nearest; O(n^3), 2-approx for metric TSP
- Greedy (nearest edge) — O(n^2 log n)

**Metaheuristics:**
- Local Search — 2-opt (segment reversal), Or-opt (segment relocation), VND
- Simulated Annealing — 2-opt neighborhood, auto-calibrated temperature
- Genetic Algorithm — OX crossover, swap mutation, optional 2-opt LS

**Benchmark instances:** small4 (4 cities), small5 (5 cities), gr17 (17 cities, optimal=2016)

## CVRP Problem Family

### Problem Definition (CVRP)

n customers with demands, a depot (node 0), and identical vehicles with
capacity Q. Find routes (depot → customers → depot) minimizing total
distance, visiting each customer exactly once, respecting capacity.

Complexity: NP-hard (generalizes both TSP and Bin Packing).

**Constructive heuristics:**
- Clarke-Wright Savings (1964) — merge route pairs by largest savings, O(n^2 log n)
- Sweep Algorithm (Gillett & Miller, 1974) — angular sweep from depot, O(n log n)

**Metaheuristics:**
- Simulated Annealing — relocate/swap/2-opt* inter-route neighborhoods
- Genetic Algorithm — giant-tour encoding (Prins, 2004), OX crossover, split decoder

## Facility Location Problem Family

### Problem Definition (UFLP)

Given m potential facility sites with opening costs f_i and n customers
with assignment costs c_ij, select facilities to open and assign each
customer to minimize total fixed + assignment cost.

Complexity: NP-hard. Best known approximation: 1.488 (Li, 2013).

**Heuristics:**
- Greedy Add — iteratively open most cost-reducing facility
- Greedy Drop — start all open, drop least impactful facility

**Metaheuristics:**
- Simulated Annealing — toggle/swap moves with Boltzmann acceptance

## p-Median Problem Family

### Problem Definition (PMP)

Open exactly p facilities from m candidates to minimize total weighted
distance from n customers to their nearest open facility.

Complexity: NP-hard for general p (Kariv & Hakimi, 1979).

**Heuristics:**
- Greedy — iteratively add most cost-reducing facility until p open
- Teitz-Bart Interchange — swap open/closed facilities until no improvement

## Shortest Path Problem Family

### Problem Definition (SPP)

Find minimum-weight path from source s to target t in a directed graph.

**Exact methods:**
- Dijkstra's Algorithm — O((V+E) log V) with binary heap, non-negative weights
- Bellman-Ford — O(VE), handles negative weights, detects negative cycles

## Maximum Flow Problem Family

### Problem Definition (Max-Flow)

Given a directed graph G = (V, E) with edge capacities c(u,v), a source s,
and a sink t, find the maximum flow from s to t such that flow on each edge
does not exceed capacity, and flow conservation holds at all non-terminal nodes.

The Max-Flow Min-Cut Theorem: maximum flow equals minimum s-t cut capacity.

**Exact methods:**
- Edmonds-Karp — O(V * E^2), BFS augmenting paths with min-cut extraction

## Minimum Spanning Tree Problem Family

### Problem Definition (MST)

Given an undirected, weighted, connected graph G = (V, E), find the spanning
tree of minimum total edge weight (n-1 edges connecting all n vertices).

**Exact methods:**
- Kruskal's Algorithm — O(E log E) with union-find
- Prim's Algorithm — O(E log V) with binary heap

## Linear Assignment Problem Family

### Problem Definition (LAP)

Given an n×n cost matrix C, find a one-to-one assignment of agents to tasks
minimizing total cost: min Σ c_{i,σ(i)} where σ is a permutation.

Complexity: Polynomial — O(n^3) via Hungarian method.

**Exact methods:**
- Hungarian (Kuhn-Munkres, 1955) — O(n^3), shortest augmenting path variant

**Heuristics:**
- Greedy — assign cheapest available task, O(n^2)

## 0-1 Knapsack Problem Family

### Problem Definition (KP01)

Given n items with weights w_i and values v_i, and a knapsack with
capacity W, select a subset to maximize total value subject to
sum(w_i) <= W.

Complexity: NP-hard (weakly) — admits pseudo-polynomial DP in O(nW).

**Exact methods:**
- Dynamic Programming — O(n * W) pseudo-polynomial, practical for moderate W
- Branch and Bound — LP relaxation upper bound, greedy warm-start

**Heuristics:**
- Greedy value density — sort by v_i/w_i, O(n log n)
- Greedy combined — best of density and max-value, 1/2-approximation

**Metaheuristics:**
- Genetic Algorithm — binary encoding, uniform crossover, repair operator

**Benchmark instances:** small4 (4 items, opt=35), medium8 (8 items, opt=300),
strongly_correlated_10 (10 items, v_i = w_i + 10)

## 1D Bin Packing Problem Family

### Problem Definition (BPP1D)

Given n items with sizes s_i and bins of capacity C, pack all items
into the minimum number of bins.

Complexity: NP-hard (strongly).

**Heuristics:**
- First Fit (FF) — O(n^2), place in first bin that fits
- First Fit Decreasing (FFD) — O(n log n), 11/9 * OPT + 6/9 approximation
- Best Fit Decreasing (BFD) — O(n log n), place in tightest fitting bin

**Metaheuristics:**
- Genetic Algorithm — permutation encoding, FF decoder

**Benchmark instances:** easy6, tight8, uniform10

## 1D Cutting Stock Problem Family

### Problem Definition (CSP1D)

Given stock material of length L and m item types with lengths l_i
and demands d_i, cut stock rolls to satisfy all demands using minimum
rolls. Generalization of Bin Packing where identical items are interchangeable.

Complexity: NP-hard (reduces from Bin Packing).

**Heuristics:**
- Greedy largest-first — fill each roll with largest remaining items
- FFD-based — expand demands to individual items, apply FFD, aggregate patterns

**Benchmark instances:** simple3 (3 types, L=100), classic4 (4 types, L=10)

## VRPTW Problem Family

### Problem Definition (VRPTW)

Extends CVRP with time window constraints [e_i, l_i] for each customer.
Vehicles must arrive before l_i; if arriving before e_i, they wait.
Each customer requires s_i service time. The depot has a planning horizon.

Complexity: NP-hard (generalizes CVRP).

**Constructive heuristics:**
- Solomon I1 Insertion (1987) — iterative insertion with composite criterion (distance + urgency), O(n^2 K)
- Nearest Neighbor TW — greedy nearest feasible customer, O(n^2)

**Metaheuristics:**
- Simulated Annealing — relocate/swap with time window feasibility checks
- Genetic Algorithm — giant-tour encoding with TW-aware split decoder

**Benchmark instances:** solomon_c101_mini (8 customers, clustered), tight_tw5 (5 customers, narrow windows)

## Newsvendor Problem Family

### Problem Definition (NV | stochastic demand | min E[cost])

A retailer orders Q units of a perishable product before uncertain demand D.
Overage cost c_o = c - v; underage cost c_u = p - c. Optimal order satisfies
P(D <= Q*) = c_u / (c_u + c_o) (critical fractile).

**Exact methods:**
- Critical Fractile — O(S log S), sort scenarios, scan CDF
- Grid Search — O(S * G), brute-force over demand range

**Heuristics:**
- Marginal Allocation — greedy for multi-product with budget constraint
- Independent + Scale — solve independently, scale to fit budget

## Two-Stage Stochastic Programming Family

### Problem Definition (2SSP)

First-stage decisions x before uncertainty; second-stage recourse y(s) after.
min c^T x + E[q(s)^T y(s)] subject to constraints.

**Methods:**
- Deterministic Equivalent — expand all scenarios into one LP (HiGHS)
- Expected Value (EV) — solve on mean scenario, lower bound
- Sample Average Approximation (SAA) — solve on random subsets, replicate

## Robust Shortest Path Problem Family

### Problem Definition

Find path minimizing worst-case cost or regret across S weight scenarios.

**Exact methods:**
- Label-Setting — multi-objective Dijkstra with dominance pruning (min-max cost)
- Scenario Enumeration — Dijkstra per scenario, cross-evaluate

**Heuristics:**
- Regret Enumeration — evaluate candidate paths against per-scenario optima
- Midpoint — shortest path on mean-weight graph

## Stochastic Knapsack Problem Family

### Problem Definition

Select items with deterministic values but random weights to maximize value
subject to capacity constraint holding with probability >= 1-alpha.

**Heuristics:**
- Greedy (mean weight) — value-density on expected weights
- Greedy (chance-constrained) — add items maintaining P(feasible) >= 1-alpha

**Metaheuristics:**
- Simulated Annealing — flip-bit with infeasibility penalty

## Chance-Constrained Facility Location

### Problem Definition

Open facilities and assign customers minimizing cost, subject to
P(demand <= capacity) >= 1-alpha per facility under stochastic demands.

**Heuristics:**
- Greedy Open — iteratively open cost-reducing facilities with CC checks
- Mean-Demand Greedy — deterministic proxy using expected demands

**Metaheuristics:**
- Simulated Annealing — toggle/swap with violation penalty

## Robust Portfolio Optimization

### Problem Definition

Markowitz mean-variance with ellipsoidal uncertainty on expected returns:
max mu^T w - delta * ||Sigma^{1/2} w||_2 - lambda * w^T Sigma w.

**Exact methods:**
- QP Solver (mean-variance) — SLSQP on Markowitz objective
- QP Solver (robust) — SLSQP with uncertainty penalty (SOCP)

**Heuristics:**
- Equal Weight (1/n) — naive diversification
- Min Variance — Sigma^{-1} 1 closed-form
- Max Return — concentrate on best expected return

## Stochastic VRP Problem Family

### Problem Definition

CVRP with stochastic customer demands. Routes designed a priori;
overflow triggers recourse (return to depot).

**Heuristics:**
- Chance-Constrained Clarke-Wright — savings with P(overflow) <= alpha
- Mean-Demand Savings — CW with expected demands

**Metaheuristics:**
- Simulated Annealing — relocate/swap/2-opt with recourse penalty

## Robust Scheduling Problem Family

### Problem Definition (1 | uncertain p_j | min max-regret ΣwjCj)

Single machine scheduling with uncertain processing times.
Minimize worst-case regret of total weighted completion time.

**Heuristics:**
- Midpoint WSPT — WSPT on mean processing times
- Scenario Enumeration — WSPT per scenario, cross-evaluate regret
- Worst-Case WSPT — WSPT on maximum processing times

**Metaheuristics:**
- Simulated Annealing — swap/insertion to minimize max regret

## Distributionally Robust Optimization (DRO)

### Problem Definition

min_x max_{P in A} E_P[f(x, xi)] — optimize under worst-case distribution.

**Exact methods:**
- Wasserstein DRO — LP reformulation with L1-norm regularization
- Nominal LP — baseline without robustness

**Heuristics:**
- Moment-Based DRO — inner LP for worst-case distribution, grid search

## Code Conventions

### Architecture Pattern

Every problem follows this structure:

1. **`instance.py`** — Problem and Solution dataclasses at problem root
2. **`exact/`** — Optimal solvers (B&B, MIP, DP, polynomial algorithms)
3. **`heuristics/`** — Constructive heuristics (fast, approximate)
4. **`metaheuristics/`** — Improvement methods (local search, population-based)
5. **`tests/`** — pytest suite
6. **`benchmark_runner.py`** — CLI to evaluate on standard benchmarks
7. **`variants/`** — Problem variants with modified constraints

### File Template

Each algorithm file must contain:
- **Module docstring** with algorithm name, problem notation (alpha | beta | gamma), complexity, and references with DOI links
- **Dataclass-based** instances and solutions (using `@dataclass`)
- **Main solve function** as the entry point (e.g., `neh()`, `simulated_annealing()`)
- **`if __name__ == "__main__"` block** with example usage and comparisons
- **Type hints** on all function signatures

### Naming & Style

- **PEP 8**: snake_case for functions/variables, PascalCase for classes, CONSTANT_CASE for constants
- **Numpy arrays** for numerical data (processing times, completion matrices)
- **Docstrings**: Google-style with Args, Returns, Raises sections
- **No global state**: all state passed explicitly via function parameters
- **Determinism**: random algorithms must accept a `seed` parameter
- **Warm-starts**: exact methods should warm-start with best constructive heuristic (typically NEH)

### Import Conventions

- Relative imports within same problem directory: `from instance import FlowShopInstance`
- Absolute imports for shared modules: `from shared.parsers.taillard_parser import load_taillard_instance`
- `sys.path.insert()` used for nested directory access (established pattern in this repo)
- **Variant modules** use `importlib.util` for explicit file-path imports to avoid name collisions with parent `instance.py`
- Optional imports wrapped in try/except (e.g., `ortools`, `gurobipy`)

### Data Structure Conventions

```python
@dataclass
class ProblemInstance:
    n: int                          # primary dimension
    param: np.ndarray               # numpy for numerical data

    @classmethod
    def from_file(cls, filepath: str) -> 'ProblemInstance': ...

    @classmethod
    def random(cls, **kwargs) -> 'ProblemInstance': ...

@dataclass
class Solution:
    objective: int | float
    representation: list[int]       # e.g., permutation

    def __repr__(self) -> str: ...
```

### Testing Conventions

- **pytest** framework with fixtures and parametrization
- Run tests with `python -m pytest` to ensure correct module resolution
- Test categories per algorithm: correctness on small handcrafted instances, comparison against known optima, edge cases (single job/machine), benchmark validation on Taillard instances
- Test class naming: `TestAlgorithmName` (e.g., `TestNEH`, `TestJohnsonsRule`)
- Variant tests use `importlib.util` based module loading to avoid import collisions
- Large instance tests should be marked or have reasonable timeouts

### Documentation Per Problem

Each problem folder should contain:
- **README.md**: Problem definition, mathematical formulation, complexity
- **BENCHMARKS.md**: Standard benchmark instances with URLs
- **LITERATURE.md**: Key papers with DOI links and one-sentence summaries

## Key Domain Concepts

- **Scheduling notation**: alpha | beta | gamma (machine environment | constraints | objective)
- **PFSP**: Permutation Flow Shop Problem — all jobs follow same machine order
- **NWFSP**: No-Wait Flow Shop — jobs proceed without waiting between machines
- **BFSP**: Blocking Flow Shop — no intermediate buffers between machines
- **Makespan (Cmax)**: Completion time of last job — primary objective
- **RPD**: Relative Percentage Deviation from best known solution — primary benchmark metric
- **Taillard benchmarks**: 120 standard PFSP instances across 12 size classes (20-500 jobs, 5-20 machines)
- **Delay matrix**: For NWFSP, asymmetric matrix D[j][k] giving minimum start-to-start gap between jobs
- **Departure time**: For BFSP, time when a job leaves a machine (may be after processing completes due to blocking)
- **Tardiness**: Tj = max(0, Cj - dj) — lateness clipped at zero
- **Weighted tardiness**: ΣwjTj — strongly NP-hard single machine objective
- **SPT/WSPT/EDD**: Optimal dispatching rules for tractable single machine objectives
- **ATC**: Apparent Tardiness Cost — composite dispatching rule combining WSPT ratio with due date urgency
- **Disjunctive graph**: Standard JSP model — conjunctive arcs for job precedence, disjunctive arcs for machine conflicts
- **Critical path**: Longest path in the disjunctive graph, determines the makespan
- **Critical block**: Consecutive operations on same machine on the critical path — key for effective neighborhoods
- **SGS**: Schedule Generation Scheme — decodes a priority list into a feasible RCPSP schedule
- **Activity list**: Precedence-feasible permutation of activities — standard encoding for RCPSP metaheuristics
- **TSP**: Traveling Salesman Problem — find shortest Hamiltonian cycle visiting all cities exactly once
- **ATSP**: Asymmetric TSP — directed distances, d(i,j) ≠ d(j,i)
- **Hamiltonian cycle**: A cycle visiting every vertex exactly once — the tour structure in TSP
- **CVRP**: Capacitated Vehicle Routing Problem — TSP generalization with multiple vehicles and capacity constraints
- **Savings**: Clarke-Wright measure s(i,j) = d(0,i) + d(0,j) - d(i,j) — benefit of merging two routes
- **Giant tour**: Single permutation of all customers, split into capacity-feasible routes for CVRP
- **2-opt**: Local search that reverses a tour segment — fundamental TSP improvement
- **1-tree**: Minimum spanning tree plus one extra edge — basis for TSP lower bounds
- **VRPTW**: Vehicle Routing Problem with Time Windows — CVRP with arrival time constraints [e_i, l_i]
- **Time window**: Interval [earliest, latest] during which service can begin at a customer
- **Solomon benchmarks**: Standard VRPTW instances (C/R/RC classes) with 25-100 customers
- **Newsvendor**: Single-period inventory under demand uncertainty — critical fractile Q* = F^{-1}(c_u/(c_u+c_o))
- **Critical fractile**: Optimal service level ratio c_u/(c_u+c_o) for the newsvendor problem
- **Overage/underage cost**: c_o = c - v (excess), c_u = p - c (shortage) — key newsvendor parameters
- **Two-stage stochastic programming**: First-stage decisions before uncertainty, second-stage recourse after
- **Recourse**: Corrective actions taken after uncertainty is revealed in stochastic programs
- **Deterministic equivalent**: Extensive form LP expanding all scenarios — size grows with S
- **Value of stochastic solution (VSS)**: Benefit of stochastic over deterministic approach
- **Sample Average Approximation (SAA)**: Solve stochastic programs on random scenario subsets
- **Min-max regret**: Minimize worst-case deviation from scenario-optimal solution
- **Wasserstein DRO**: Distributionally robust optimization with Wasserstein distance ambiguity set
- **Moment-based ambiguity**: DRO where distributions must match mean/covariance within tolerance
- **Chance constraint**: Probabilistic constraint P(g(x,xi) <= 0) >= 1-alpha
- **Robust portfolio**: Markowitz mean-variance with ellipsoidal uncertainty on expected returns
- **Stochastic VRP**: Vehicle routing with random demands — recourse for route overflow

## Adding New Problems

Follow the pattern established by `problems/scheduling/flow_shop/`:

1. Create the problem directory under appropriate family
2. Implement `instance.py` with dataclass-based Instance and Solution
3. Add algorithms in `exact/`, `heuristics/`, `metaheuristics/` subdirectories
4. Write comprehensive tests in `tests/`
5. Add benchmark runner if standard benchmarks exist
6. Include README.md, BENCHMARKS.md, LITERATURE.md documentation
7. For variants of an existing problem, use the `variants/` subdirectory pattern
8. See `CONTRIBUTING.md` for the full template and guidelines

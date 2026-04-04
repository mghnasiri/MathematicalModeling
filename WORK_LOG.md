# WORK LOG

> Running log of all enrichment work, timestamped per folder.

---

## 2026-04-04 — Phase 0: Reconnaissance

**Completed:** Full directory inventory of `/problems/` and `/applications/`.

**Key findings:**
- 13 problem family directories (incl. 2 legacy), 74 subfamilies, 48+ variants
- 623 Python files in problems/, 34 in applications/
- 42 problem subfamilies have NO README at all
- Family 7 (Inventory) has zero READMEs across all 7 subfamilies
- Family 8 (Integrated) has zero Python code — documentation stubs only
- Legacy folders (combinatorial, continuous, multi_objective) have code but no READMEs
- The best-structured folder is `2_routing/tsp/` but even it lacks several gold-standard sections
- Applications have 33 standalone Python files + 9 domain subdirectories with phase-based READMEs
- GitHub Pages site in `docs/` has ~135 HTML files — must not be modified

**Created:** `AUDIT_MANIFEST.md` with complete inventory and status assessments.

**Next:** Phase 1 — Define gold standard template.

---

## 2026-04-04 — Phase 1: Gold Standard Template

**Created:** `TEMPLATE_STANDARD.md` with templates for problem READMEs (8 sections + notation conventions), variant READMEs (lighter 5-section format), application domain READMEs (9 sections), application phase READMEs, and code file standards.

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/flow_shop/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- Complete README rewrite from ~187 lines to ~400+ lines matching gold standard
- Added: formal Input/Decision/Objective/Constraints structure (Section 1)
- Added: notation table + 3 alternative formulations (recursion, position-based MILP, CP-SAT) (Section 2)
- Added: per-variant subsections with brief descriptions (Section 3)
- Added: Taillard benchmark library table, instance format, small illustrative instance (Section 4)
- Added: full method tables for ALL 11 heuristics and ALL 17 metaheuristics (README was listing only 6 heuristics and 5 metaheuristics — repo actually has 28+ algorithm files) (Section 5)
- Added: pseudocode for Johnson's Rule, NEH, and Iterated Greedy (Section 5)
- Added: implementation guide with Taillard acceleration, common pitfalls (Section 6)
- Added: scale guidance table (Section 7)
- Added: complete directory tree reflecting ALL 31 .py files and 8 test files (Section 8)
- Added: full references section with seminal papers, surveys, benchmark citation (Section 10)
- Preserved: algorithm taxonomy tree and key insights

**Open questions:** None — this is the best-covered problem folder in the repo.

---

## 2026-04-04 — Phase 2: `problems/2_routing/tsp/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- Complete README rewrite from ~113 lines to ~350+ lines matching gold standard
- Added: formal Input/Decision/Objective/Constraints structure
- Added: notation table + 3 formulations (DFJ with exponential SECs, MTZ compact, 1-tree relaxation)
- Added: per-variant subsections for ATSP, TSPTW, PCTSP, PDP
- Added: TSPLIB benchmark library reference + small illustrative instance
- Added: pseudocode for Held-Karp, Nearest Neighbor, 2-opt
- Added: all 7 metaheuristics documented (README was listing only 3; actual repo has TS, ACO, IG, VNS)
- Added: implementation guide (distance matrix tips, 2-opt evaluation, neighbor lists)
- Added: computational results table with scale guidance
- Updated: directory tree showing all 5 test files (was listing 1)
- Expanded: references with books and surveys

---

## 2026-04-04 — Phase 2: `problems/2_routing/cvrp/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- Complete README rewrite from ~99 lines to ~320+ lines matching gold standard
- Added: formal Input/Decision/Objective/Constraints structure
- Added: notation table + 2 formulations (2-index vehicle flow with rounded capacity cuts, 3-index vehicle flow)
- Added: per-variant brief descriptions for 3 of 10 variants
- Added: CVRPLIB reference with Augerat/Christofides/Uchoa sets + small instance
- Added: pseudocode for Clarke-Wright savings
- Added: all 7 metaheuristics documented (README had 2; actual repo has TS, ACO, IG, LS, VNS)
- Added: 2-opt* inter-route neighborhood explanation
- Added: giant-tour encoding + split procedure explanation
- Added: implementation guide with split DP, savings precomputation, pitfalls
- Added: computational results with scale guidance
- Updated: directory tree showing all 6 test files (was 1)
- Added: state-of-the-art references (ALNS, HGS)

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/single_machine/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- Complete rewrite matching gold standard
- Added: notation table, WSPT optimality proof sketch, ATC formula and pseudocode
- Added: pseudocode for Moore's Algorithm
- Added: all 6 metaheuristics documented (README had 2; added GA, IG, LS, VNS)
- Added: implementation guide, computational results, scale guidance
- Updated: directory tree showing all 6 test files (was 1)
- Added: full references (Smith, Moore, Jackson, Pinedo textbook)

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/job_shop/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- **CRITICAL FIX:** Removed phantom files from directory tree (exact/disjunctive_mip.py, exact/constraint_programming.py, heuristics/giffler_thompson.py — none of these exist in the repo)
- Added: all 6 metaheuristics documented (README had 3; added IG, local_search, VNS)
- Added: critical-path neighborhood detail (N1, N5)
- Added: benchmark instances table (ft06, ft10, ft20, Lawrence, Taillard)
- Added: Shifting Bottleneck description with quality notes
- Added: implementation guide (disjunctive graph, incremental evaluation)
- Added: computational results table
- Updated: directory tree showing all 6 test files (was 1)
- Noted: no exact method implementations currently exist

---

## 2026-04-04 — Phase 2: `problems/3_packing_cutting/knapsack/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

**Changes:**
- Complete rewrite matching gold standard
- Added: notation table, LP relaxation / Dantzig bound description
- Added: DP pseudocode, variant subsections (bounded, multidim, subset sum)
- Added: all 6 metaheuristics documented (README had 1; added SA, TS, IG, LS, VNS)
- Added: repair operator explanation, implementation guide, computational results
- Updated: directory tree showing all 6 test files (was 1), Pisinger/OR-Library benchmark refs

---

## 2026-04-04 — Phase 2: Batch — `bin_packing`, `cutting_stock`, `vrptw`

**All three:** `[GOOD]` → `[EXCELLENT]`

**bin_packing:** MILP formulation with symmetry-breaking, FFD pseudocode with tight 11/9 bound, all 6 metaheuristics, 6 test files.

**cutting_stock:** Gilmore-Gomory pattern-based formulation, column generation description, all 6 metaheuristics (README had 0), 7 test files.

**vrptw:** Time window propagation formulation, Solomon I1 description, all 7 metaheuristics (README had 2; added TS, ACO, IG, LS, VNS), forward time slack tip, Solomon benchmarks, 6 test files.

---

## 2026-04-04 — Phase 2: Batch — families 4-6 (assignment, location, network)

**All six:** `[GOOD]` or `[PARTIAL]` → `[EXCELLENT]`

**assignment:** Total unimodularity explanation, Hungarian description, 3 variant directories.

**facility_location:** MILP formulation, LP relaxation note, all 6 metaheuristics documented (README had SA only), 1.488 approximation.

**p_median:** MILP formulation, Teitz-Bart description, all 6 metaheuristics documented (README had 0), 7 test files.

**shortest_path:** Dijkstra + Bellman-Ford pseudocode, LP formulation (TU matrix), APSP variant.

**max_flow:** Max-Flow Min-Cut theorem, LP duality, added Dinic's algorithm (missing from README), 2 test files.

**min_spanning_tree:** Kruskal + Prim pseudocode, cut/cycle properties, Steiner tree variant.

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/parallel_machine/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

- Complete rewrite with notation table, MIP formulation, LPT pseudocode
- All 6 metaheuristics documented (README had GA only)
- Updated directory tree (6 test files, not 1), added MULTIFIT explanation
- Added implementation guide, computational results, approximation ratios

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/flexible_job_shop/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

- CRITICAL FIX: Removed phantom files (exact/mip_fjsp.py, metaheuristics/nsga2.py — don't exist)
- Noted empty exact/ directory explicitly
- All 6 metaheuristics documented (README had GA/SA only)
- Added Brandimarte/Barnes/Hurink benchmark references
- Updated directory tree (7 test files, not 1)

---

## 2026-04-04 — Phase 2: `problems/1_scheduling/rcpsp/`

**Status change:** `[GOOD]` → `[EXCELLENT]`

- CRITICAL FIX: Removed phantom files (exact/mip_time_indexed.py, exact/constraint_programming.py, heuristics/priority_rules.py — don't exist)
- Noted empty exact/ directory explicitly
- All 6 metaheuristics documented (README had GA/SA only)
- Added Serial SGS pseudocode, PSPLIB benchmark table
- Added activity-list encoding explanation
- Updated directory tree (7 test files, not 1)

---

## 2026-04-04 — Phase 2: `problems/9_uncertainty_modeling/` (batch — 8 remaining)

**All eight:** `[GOOD]` → `[EXCELLENT]`

**two_stage_sp:** Added notation table (9 symbols), deterministic equivalent extensive form, newsvendor-as-2SSP illustrative instance, SAA description, 4 factory methods documented.

**robust_shortest_path:** Added notation table, min-max cost/regret/expected cost criteria table with complexity, label-setting description, 4-node illustrative instance, 4 solution methods with complexity.

**stochastic_knapsack:** Added notation table, chance-constrained formulation, expected-capacity variant, 4-item illustrative instance showing feasibility probability check, 3 methods documented.

**chance_constrained_fl:** Added full MILP formulation (5 constraint groups), notation table (10 symbols), 2-facility illustrative instance, capacity_violation_prob() documented, 3 methods.

**robust_portfolio:** Added Markowitz and robust SOCP formulations, ellipsoidal uncertainty set explanation, 3-asset illustrative instance, DeMiguel 1/n insight, 5 methods documented.

**stochastic_vrp:** Added notation table, chance-constrained formulation, recourse policy description, 4-customer illustrative instance, overflow probability mechanics, 3 methods.

**robust_scheduling:** Added scheduling notation (1|uncertain p_j|min max-regret ΣwjCj), WSPT optimality per scenario, 3-job illustrative instance, 2-approximation note, 4 methods.

**dro:** Added notation table, Wasserstein dual reformulation, moment-based ambiguity set, 2-variable illustrative instance, regularization interpretation, 3 methods.

---

## 2026-04-04 — Phase 2: `problems/7_inventory_lotsizing/` (all 7 — READMEs from scratch)

**All seven:** `[EMPTY]` → `[EXCELLENT]`

**eoq:** Classic EOQ formula, backorder variant, quantity discount variant. Notation table, $O(1)$/$O(B \log B)$ complexity. Harris (1913) reference.

**lot_sizing:** Dynamic lot sizing with ZIO property. Wagner-Whitin DP formulation, Silver-Meal and Part-Period Balancing heuristics. 4-period illustrative instance.

**wagner_whitin:** Standalone Wagner-Whitin DP. ZIO property explanation, $O(T^2)$ DP recurrence, 5-period illustrative instance.

**capacitated_lot_sizing:** Full MILP formulation (5 constraint groups), NP-hardness citation. MIP via HiGHS, two greedy heuristics. 7 Python files, 3 test files documented.

**multi_echelon_inventory:** Serial supply chain with $L$ echelons. Echelon base-stock policy, powers-of-two (2% optimality), greedy allocation. Clark-Scarf (1960) reference.

**safety_stock:** Analytical $\sigma_{\text{DDLT}}$ formula, safety factor $z = \Phi^{-1}(\text{SL})$, reorder point. Demand + lead-time variability convolution.

**vehicle_loading:** Dual-capacity bin packing (weight + volume). MILP formulation, FFD heuristic. 5-item illustrative instance.

---

## 2026-04-04 — Phase 2: `problems/combinatorial/` (all 7 — READMEs from scratch)

**All seven:** `[EMPTY]` → `[EXCELLENT]`

**graph_coloring:** ILP formulation, DSatur description, chromatic number notation. Brélaz (1979) reference.

**graph_partitioning:** Balanced k-way formulation, Kernighan-Lin algorithm description. $O(n^2 \log n)$ complexity.

**job_sequencing:** Scheduling notation 1|d_j|ΣwjUj, ILP formulation, greedy unit-time algorithm. Moore (1968) reference.

**max_clique:** ILP formulation, Bron-Kerbosch with pivoting, relationship to MIS. $O(3^{n/3})$ worst case.

**max_independent_set:** ILP formulation, B&B + greedy heuristic, Gallai's theorem. Inapproximability result.

**maximum_satisfiability:** ILP with clause satisfaction, greedy 1/2-approximation. Goemans-Williamson SDP reference.

**vertex_cover:** ILP formulation, 2-approximation via maximal matching, LP half-integrality. Bar-Yehuda & Even (1981).

---

## 2026-04-04 — Phase 2: `problems/continuous/` + `problems/multi_objective/` (7 READMEs from scratch)

**All seven:** `[EMPTY]` → `[EXCELLENT]`

**linear_programming:** Standard LP form, sensitivity analysis (shadow prices, reduced costs, allowable ranges). HiGHS solver. Dantzig/Bertsimas references.

**quadratic_programming:** Convex QP standard form, KKT conditions. SLSQP solver. Nocedal & Wright, Boyd & Vandenberghe references.

**nonlinear_programming:** General NLP form with callable constraints, KKT conditions. SciPy minimize interface.

**semidefinite_relaxation:** MAX-CUT via GW SDP relaxation, 0.878 approximation ratio, random hyperplane rounding. Eigendecomposition proxy.

**bi_objective_knapsack:** Pareto front formulation, epsilon-constraint method, 4-item illustrative instance. Ehrgott/Bazgan references.

**multi_objective_tsp:** $k$ distance matrices, weighted-sum scalarization, nearest-neighbor per weight vector. Jaszkiewicz reference.

**multi_objective_shortest_path:** Pareto-optimal paths, label-setting (multi-objective Dijkstra), dominance pruning. Hansen/Martins references.

---

## 2026-04-04 — Phase 2: Remaining problem directories (21 READMEs from scratch)

**All twenty-one:** `[EMPTY]` → `[EXCELLENT]`

### Scheduling (5)

**assembly_line_balancing:** SALBP-1 formulation, cycle time constraint, RPW heuristic, Helgeson-Birnie reference.

**batch_scheduling:** 1|batch,s_fam|ΣwjCj notation, family grouping formulation, BATC dispatching rule, Ikura-Potts reference.

**nurse_scheduling:** NSP formulation, shift coverage + pattern constraints, greedy roster heuristic, Warner (1976) reference.

**project_scheduling:** Multi-project RCPSP (MPSP), cross-project resource sharing, priority SGS, Gonçalves (2008) reference.

**workforce_scheduling:** WS|skills,availability|min uncovered formulation, skill-coverage constraints, greedy cover heuristic.

### Routing (5)

**arc_routing:** CARP formulation, edge-demand constraints, path scanning heuristic, Golden-Wong reference.

**chinese_postman:** CPP formulation, polynomial solution via minimum-weight matching on odd-degree vertices, Edmonds-Johnson reference.

**dial_a_ride:** DARP formulation, pickup-delivery pairing + ride-time constraints, insertion heuristic, Cordeau-Laporte reference.

**multi_depot_vrp:** MDVRP formulation, depot assignment + routing, nearest-depot heuristic, Renaud (1996) reference.

**vrp_pickup_delivery:** VRPPD formulation, precedence + capacity constraints, insertion heuristic, Savelsbergh-Sol reference.

### Packing & Cutting (4)

**bin_packing_2d:** 2D-BPP formulation, shelf algorithms (NFDH, FFDH), Coffman et al. reference.

**multidim_knapsack:** MdKP formulation, m capacity constraints, pseudo-utility greedy, Chu-Beasley reference.

**multiple_knapsack:** MKP formulation, item-to-bin assignment, ILP + density greedy, Martello-Toth reference.

**strip_packing:** 2D-SPP formulation, NFDH/FFDH shelf heuristics, Coffman et al. reference.

### Assignment (1)

**quadratic_assignment:** QAP Koopmans-Beckmann formulation, Gilmore-Lawler bound, greedy + SA, Sahni-Gonzalez NP-hardness.

### Location & Covering (4)

**hub_location:** p-Hub Median formulation, O'Kelly linearization, enumeration + greedy heuristic, O'Kelly (1987) reference.

**max_coverage:** Submodular maximization, (1-1/e) greedy approximation, ILP formulation, Nemhauser et al. (1978) reference.

**set_covering:** SCP formulation, ILP via HiGHS, greedy ln(m)+1 approximation, Chvatal (1979) reference.

**set_packing:** Max-weight set packing formulation, greedy 1/k-approximation, Hurkens-Schrijver reference.

### Network (2)

**multi_commodity_flow:** MCFP arc-commodity formulation, shared capacity constraints, LP via linprog, Ahuja-Magnanti-Orlin reference.

**network_design:** FCNDP formulation, big-M capacity linking, greedy edge opening heuristic, Magnanti-Wong reference.

**Total: 0 missing READMEs remaining in problems/ directory.**

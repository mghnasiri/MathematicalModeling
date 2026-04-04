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

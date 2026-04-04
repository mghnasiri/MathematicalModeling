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

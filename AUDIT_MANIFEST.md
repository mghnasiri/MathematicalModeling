# AUDIT MANIFEST

> Generated: 2026-04-04 | Phase 0 Reconnaissance
>
> This document inventories every folder in `/problems/` and `/applications/`,
> assesses its current state, and tracks enrichment progress through Phase 2+.

## Status Legend

| Code | Meaning |
|------|---------|
| `[EMPTY]` | Directory exists but contains no README or only placeholder |
| `[STUB]` | README exists but is < 30 lines with minimal content |
| `[PARTIAL]` | README has some sections but significant gaps vs. gold standard |
| `[GOOD]` | README covers most sections; code implementations exist and work |
| `[EXCELLENT]` | Fully matches gold standard template; all code verified |

## Summary Statistics

| Metric | Count |
|--------|-------|
| Problem families | 13 (incl. 2 legacy) |
| Problem subfamilies | 74 |
| Variant folders | 48+ (across 17 variant parent dirs) |
| Total Python files (problems/) | 623 |
| Total READMEs (problems/) | 84 |
| Missing READMEs (subfamilies) | 42 |
| Application domains | 9 |
| Application Python files | 34 |
| Application READMEs | 49 |
| GitHub Pages HTML files | ~135 (in docs/) |

---

## 1. PROBLEMS INVENTORY

### Family 1: Scheduling (`1_scheduling/`) — 206 .py, 36 .md, 82 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `flow_shop/` | Yes | ~20 | 3 test files (130 tests) | 10 (blocking, distributed, hybrid, lot_streaming, no_wait, open_shop, setup_times, stochastic, tardiness) | `[GOOD]` |
| `single_machine/` | Yes | ~7 | 1 test file (55 tests) | 2 (batch, preemptive) | `[GOOD]` |
| `parallel_machine/` | Yes | ~7 | 1 test file (43 tests) | 2 (sdst, unrelated_tardiness) | `[GOOD]` |
| `job_shop/` | Yes | ~5 | 1 test file (41 tests) | 3 (flexible_tardiness, no_wait, weighted_tardiness) | `[GOOD]` |
| `flexible_job_shop/` | Yes | ~4 | 1 test file (37 tests) | 0 | `[GOOD]` |
| `rcpsp/` | Yes | ~4 | 1 test file (35 tests) | 1 (multi_mode) | `[GOOD]` |
| `assembly_line_balancing/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `batch_scheduling/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `nurse_scheduling/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `project_scheduling/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `workforce_scheduling/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

**Variant READMEs:** Most variants have short STUB READMEs (~15-25 lines).

### Family 2: Routing (`2_routing/`) — 125 .py, 18 .md, 62 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `tsp/` | Yes | ~8 | 1 test file (55 tests) | 4 (asymmetric, pickup_delivery, prize_collecting, time_windows) | `[GOOD]` |
| `cvrp/` | Yes | ~5 | 1 test file (41 tests) | 10 (backhaul, backhauls, cumulative, electric, multi_compartment, multi_depot, multi_trip, open_vrp, periodic, split_delivery) | `[GOOD]` |
| `vrptw/` | Yes | ~4 | 1 test file (31 tests) | 1 (soft_time_windows) | `[GOOD]` |
| `arc_routing/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `chinese_postman/` | **NO** | ~2 | 1 test file | 0 | `[EMPTY]` |
| `dial_a_ride/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `multi_depot_vrp/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `vrp_pickup_delivery/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

### Family 3: Packing & Cutting (`3_packing_cutting/`) — 90 .py, 11 .md, 46 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `knapsack/` | Yes | ~5 | 1 test file (37 tests) | 4 (bounded, multidimensional, multiple, subset_sum) | `[GOOD]` |
| `bin_packing/` | Yes | ~3 | 1 test file (29 tests) | 3 (online, two_dimensional, variable_size) | `[GOOD]` |
| `cutting_stock/` | Yes | ~2 | 1 test file (21 tests) | 1 (two_dimensional) | `[GOOD]` |
| `bin_packing_2d/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `multidim_knapsack/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `multiple_knapsack/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `strip_packing/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

### Family 4: Assignment & Matching (`4_assignment_matching/`) — 19 .py, 6 .md, 22 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `assignment/` | Yes | ~3 | 1 test file (17 tests) | 3 (generalized, max_weight_matching, quadratic) | `[GOOD]` |
| `linear_assignment/` | Yes | ~2 | 1 test file | 0 | `[PARTIAL]` |
| `matching/` | Yes | ~2 | 1 test file | 0 | `[PARTIAL]` |
| `quadratic_assignment/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

### Family 5: Location & Covering (`5_location_covering/`) — 50 .py, 4 .md, 35 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `facility_location/` | Yes | ~3 | 1 test file (16 tests) | 1 (capacitated) | `[GOOD]` |
| `p_median/` | Yes | ~2 | 1 test file (13 tests) | 1 (capacitated) | `[GOOD]` |
| `hub_location/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `max_coverage/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `set_covering/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `set_packing/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

### Family 6: Network Flow & Design (`6_network_flow_design/`) — 27 .py, 6 .md, 25 dirs

| Subfolder | README | .py files | Tests | Variants | Status |
|-----------|--------|-----------|-------|----------|--------|
| `shortest_path/` | Yes | ~3 | 1 test file (21 tests) | 1 (all_pairs) | `[GOOD]` |
| `max_flow/` | Yes | ~2 | 1 test file (16 tests) | 1 (min_cost_flow) | `[GOOD]` |
| `min_spanning_tree/` | Yes | ~2 | 1 test file (16 tests) | 1 (steiner_tree) | `[GOOD]` |
| `multi_commodity_flow/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |
| `network_design/` | **NO** | ~3 | 1 test file | 0 | `[EMPTY]` |

### Family 7: Inventory & Lot Sizing (`7_inventory_lotsizing/`) — 27 .py, 0 .md, 47 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| `eoq/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `lot_sizing/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `wagner_whitin/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `capacitated_lot_sizing/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `multi_echelon_inventory/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `safety_stock/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `vehicle_loading/` | **NO** | ~3 | 1 test file | `[EMPTY]` |

**Note:** Entire family has ZERO READMEs. All 7 subfamilies need READMEs from scratch.

### Family 8: Integrated Structural (`8_integrated_structural/`) — 0 .py, 4 .md, 4 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| (root) | Yes | 0 | 0 | `[STUB]` |
| `assembly_line_balancing/` | Yes | 0 | 0 | `[STUB]` |
| `inventory_routing/` | Yes | 0 | 0 | `[STUB]` |
| `location_routing/` | Yes | 0 | 0 | `[STUB]` |

**Note:** All 3 subfamilies are documentation-only stubs — no implementations at all.

### Family 9: Uncertainty Modeling (`9_uncertainty_modeling/`) — 36 .py, 9 .md, 73 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| `newsvendor/` | Yes | ~3 | 1 test file (13 tests) | `[EXCELLENT]` |
| `two_stage_sp/` | Yes | ~3 | 1 test file (10 tests) | `[EXCELLENT]` |
| `robust_shortest_path/` | Yes | ~3 | 1 test file (13 tests) | `[EXCELLENT]` |
| `stochastic_knapsack/` | Yes | ~3 | 1 test file (11 tests) | `[EXCELLENT]` |
| `chance_constrained_fl/` | Yes | ~3 | 1 test file (11 tests) | `[EXCELLENT]` |
| `robust_portfolio/` | Yes | ~3 | 1 test file (14 tests) | `[EXCELLENT]` |
| `stochastic_vrp/` | Yes | ~3 | 1 test file (13 tests) | `[EXCELLENT]` |
| `robust_scheduling/` | Yes | ~3 | 1 test file (13 tests) | `[EXCELLENT]` |
| `dro/` | Yes | ~3 | 1 test file (12 tests) | `[EXCELLENT]` |

### Legacy: Combinatorial (`combinatorial/`) — 22 .py, 1 .md, 45 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| (root) | Yes (taxonomy mapping) | 0 | 0 | `[STUB]` |
| `graph_coloring/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `graph_partitioning/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `job_sequencing/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `max_clique/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `max_independent_set/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `maximum_satisfiability/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `vertex_cover/` | **NO** | ~3 | 1 test file | `[EMPTY]` |

### Legacy: Continuous Optimization (`continuous/`) — 12 .py, 0 .md, 25 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| `linear_programming/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `quadratic_programming/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `nonlinear_programming/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `semidefinite_relaxation/` | **NO** | ~3 | 1 test file | `[EMPTY]` |

### Legacy: Multi-Objective (`multi_objective/`) — 9 .py, 0 .md, 19 dirs

| Subfolder | README | .py files | Tests | Status |
|-----------|--------|-----------|-------|--------|
| `bi_objective_knapsack/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `multi_objective_tsp/` | **NO** | ~3 | 1 test file | `[EMPTY]` |
| `multi_objective_shortest_path/` | **NO** | ~3 | 1 test file | `[EMPTY]` |

### Legacy: Location & Network (`location_network/`) — 0 .py, 1 .md

Empty redirect folder. All content moved to families 4-6.

---

## 2. APPLICATIONS INVENTORY

### Root-level Python files (not in subdirectories)

| File | Lines | Status |
|------|-------|--------|
| `agriculture_crop_selection.py` | ~460 | `[GOOD]` — runnable |
| `agriculture_crop_transport.py` | ~320 | `[GOOD]` |
| `agriculture_distribution_center.py` | ~420 | `[GOOD]` |
| `agriculture_equipment_scheduling.py` | ~390 | `[GOOD]` |
| `agriculture_fertilizer_routing.py` | ~340 | `[GOOD]` |
| `agriculture_field_assignment.py` | ~440 | `[GOOD]` |
| `agriculture_harvest_scheduling.py` | ~460 | `[GOOD]` |
| `agriculture_irrigation_network.py` | ~350 | `[GOOD]` |
| `agriculture_pest_control.py` | ~360 | `[GOOD]` |
| `agriculture_seed_inventory.py` | ~340 | `[GOOD]` |
| `agriculture_silo_packing.py` | ~350 | `[GOOD]` |
| `agriculture_water_allocation.py` | ~400 | `[GOOD]` |
| `cloud_resource_packing.py` | ~310 | `[GOOD]` |
| `delivery_routing.py` | ~350 | `[GOOD]` |
| `energy_grid.py` | ~450 | `[GOOD]` |
| `finance_portfolio.py` | ~340 | `[GOOD]` |
| `healthcare_ambulance_location.py` | ~440 | `[GOOD]` |
| `healthcare_bed_management.py` | ~430 | `[GOOD]` |
| `healthcare_clinical_trial.py` | ~400 | `[GOOD]` |
| `healthcare_emergency_network.py` | ~400 | `[GOOD]` |
| `healthcare_home_visits.py` | ~420 | `[GOOD]` |
| `healthcare_nurse_assignment.py` | ~320 | `[GOOD]` |
| `healthcare_or_scheduling.py` | ~480 | `[GOOD]` |
| `healthcare_parallel_or.py` | ~320 | `[GOOD]` |
| `healthcare_patient_flow.py` | ~330 | `[GOOD]` |
| `healthcare_supply_delivery.py` | ~330 | `[GOOD]` |
| `logistics_gate_assignment.py` | ~410 | `[GOOD]` |
| `manufacturing_scheduling.py` | ~320 | `[GOOD]` |
| `retail_inventory.py` | ~440 | `[GOOD]` |
| `supply_chain_network.py` | ~310 | `[GOOD]` |
| `telecom_network.py` | ~390 | `[GOOD]` |
| `warehouse_location.py` | ~390 | `[GOOD]` |
| `workforce_assignment.py` | ~380 | `[GOOD]` |

### Subdirectory Domains

| Domain | Subdirs | README (root) | Phase READMEs | Status |
|--------|---------|---------------|---------------|--------|
| `agriculture/` | 7 phases | Yes | 7 | `[PARTIAL]` |
| `construction/` | 5 phases | Yes | 5 | `[PARTIAL]` |
| `energy/` | 5 phases | Yes | 5 | `[PARTIAL]` |
| `healthcare/` | 5 phases | Yes | 5 | `[PARTIAL]` |
| `manufacturing/` | 6 phases | Yes | 6 | `[PARTIAL]` |
| `public_services/` | 4 phases | Yes | 4 | `[PARTIAL]` |
| `retail_ecommerce/` | 4 phases | Yes | 4 | `[PARTIAL]` |
| `transportation_logistics/` | 5 phases | Yes | 5 | `[PARTIAL]` |
| `tests/` | 0 | No | 0 | `[STUB]` |

---

## 3. GITHUB PAGES SAFETY MAP

The `docs/` directory contains ~135 HTML files serving the GitHub Pages site:

- `docs/index.html` — main landing page
- `docs/families/*.html` — 14 family pages (scheduling, routing, packing, etc.)
- `docs/applications/*.html` — ~120 application pages
- `docs/taxonomy.md` — taxonomy specification
- `docs/og-image.png` — OpenGraph social image

**CRITICAL:** These HTML files must NOT be modified. They reference:
- Internal links between family and application pages
- Potentially links to `/problems/` and `/applications/` raw Markdown on GitHub
- CSS/JS likely inline or in the HTML files themselves (no separate /assets/ dir)

---

## 4. GOLD STANDARD CANDIDATE

**Best-structured folder:** `problems/2_routing/tsp/`

Reasons:
- README has: Problem Definition, Mathematical Formulation, Complexity table, Solution Approaches (categorized), Implementation directory tree, Benchmark references
- Has exact/, heuristics/, metaheuristics/ subdirectories with working code
- 55 tests passing
- 4 well-structured variants with their own READMEs and code
- Clean instance.py with dataclass pattern

**Gaps even in the gold standard:**
- No Benchmark Instances section with BKS table
- No Implementation Guide (solver modeling)
- No Computational Results Summary
- No Key References section with full citations
- Variant READMEs are stubs (~15 lines each)

---

## 5. PRIORITY QUEUE (suggested enrichment order)

### Tier 1 — Upgrade existing [GOOD] to [EXCELLENT]
These have working code + tests but need README enrichment to gold standard:

1. `1_scheduling/flow_shop/` (flagship, most code)
2. `2_routing/tsp/` (current best structure)
3. `2_routing/cvrp/` (most variants: 10)
4. `1_scheduling/single_machine/`
5. `1_scheduling/job_shop/`
6. `3_packing_cutting/knapsack/`

### Tier 2 — Create READMEs for [EMPTY] folders that have code
These have Python implementations but no documentation:

7. `7_inventory_lotsizing/*` (entire family, 7 subfamilies, 0 READMEs)
8. `combinatorial/*` (7 subfamilies, 0 READMEs)
9. `continuous/*` (4 subfamilies, 0 READMEs)
10. `multi_objective/*` (3 subfamilies, 0 READMEs)
11. Remaining `[EMPTY]` in families 1-6

### Tier 3 — Enrich variant READMEs from [STUB] to [GOOD]
All 48+ variant folders currently have stub READMEs.

### Tier 4 — Applications enrichment
All 9 application domains from [PARTIAL] to [EXCELLENT].

### Tier 5 — Integrated Structural (Family 8)
Documentation-only stubs with no code — need implementations or explicit scope marking.

---

## 6. ENRICHMENT TRACKING

| Folder | Phase 2 Start | Phase 2 End | Status |
|--------|--------------|-------------|--------|
| *(to be filled as work progresses)* | | | |

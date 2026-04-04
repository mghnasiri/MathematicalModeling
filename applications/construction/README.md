# Construction — Decision Chain

> **Status:** Structure in place — decision points mapped across all 5 phases.

## About this sector

Construction projects involve site selection, project scheduling under resource and precedence constraints, procurement of materials, site operations management, and project closeout. OR methods are central to project scheduling (RCPSP), resource leveling, material procurement (lot sizing), crew assignment, and equipment routing. The sector features heavy use of precedence-constrained scheduling, multi-resource allocation, and logistics optimization.

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Initiation | Site Selection | Facility Location / Multi-Criteria | — | Planned |
| 02 Planning | Project Scheduling | RCPSP | — | Planned |
| 02 Planning | Resource Leveling | LP / Smoothing | — | Planned |
| 03 Procurement | Material Procurement | EOQ / Lot Sizing | — | Planned |
| 03 Procurement | Subcontractor Selection | Set Covering / Assignment | — | Planned |
| 04 Operations | Crew Assignment | Assignment / Nurse Scheduling | — | Planned |
| 04 Operations | Equipment Routing | VRP | — | Planned |
| 04 Operations | Concrete Delivery Scheduling | Batch Scheduling | — | Planned |
| 05 Closeout | Punchlist Scheduling | Single Machine Scheduling | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| RCPSP | Scheduling | `problems/1_scheduling/rcpsp/` |
| Single Machine | Scheduling | `problems/1_scheduling/single_machine/` |
| Batch Scheduling | Scheduling | `problems/1_scheduling/batch_scheduling/` |
| VRP | Routing | `problems/2_routing/cvrp/` |
| Assignment | Assignment | `problems/4_assignment_matching/assignment/` |
| Facility Location | Location | `problems/5_location_covering/facility_location/` |
| Set Covering | Location | `problems/5_location_covering/set_covering/` |
| EOQ / Lot Sizing | Inventory | `problems/7_inventory_lotsizing/eoq/` |

## Existing implementations

No dedicated construction application files yet. See the construction project page in `docs/`.

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

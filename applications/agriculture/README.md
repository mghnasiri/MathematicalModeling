# Agriculture — Decision Chain

> **Status:** Live — 12 implementations across all 7 phases.

## About this sector

Agriculture involves sequential decisions from strategic land-use and crop selection through planting, growing, harvest, post-harvest storage and processing, and market sales. Operations research optimizes resource allocation, scheduling, routing, and inventory at every phase. The sector showcases the full breadth of OR — from portfolio optimization (crop mix under yield uncertainty) through vehicle routing (fertilizer delivery) to bin packing (silo loading).

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Strategic | Land Use & Crop Selection | LP/MIP, Robust Portfolio | `agriculture_crop_selection.py` | Live |
| 01 Strategic | Field Assignment | Assignment | `agriculture_field_assignment.py` | Live |
| 03 Planting | Seed & Input Procurement | EOQ / Newsvendor | `agriculture_seed_inventory.py` | Live |
| 04 Growing | Irrigation Scheduling | LP / Network Flow | `agriculture_irrigation_network.py` | Live |
| 04 Growing | Fertilizer Application Routing | VRP | `agriculture_fertilizer_routing.py` | Live |
| 04 Growing | Pest Control Routing | VRP | `agriculture_pest_control.py` | Live |
| 04 Growing | Water Allocation | LP / Network Flow | `agriculture_water_allocation.py` | Live |
| 05 Harvest | Harvest Scheduling | Machine Scheduling | `agriculture_harvest_scheduling.py` | Live |
| 05 Harvest | Equipment Scheduling | Parallel Machine | `agriculture_equipment_scheduling.py` | Live |
| 06 Post-Harvest | Silo Packing | Bin Packing | `agriculture_silo_packing.py` | Live |
| 06 Post-Harvest | Crop Transport | CVRP | `agriculture_crop_transport.py` | Live |
| 06 Post-Harvest | Distribution Center | Facility Location | `agriculture_distribution_center.py` | Live |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| Robust Portfolio | Uncertainty | `problems/9_uncertainty_modeling/robust_portfolio/` |
| Assignment | Assignment | `problems/4_assignment_matching/assignment/` |
| EOQ / Newsvendor | Inventory / Uncertainty | `problems/7_inventory_lotsizing/eoq/` |
| Network Flow | Network | `problems/6_network_flow_design/max_flow/` |
| VRP | Routing | `problems/2_routing/cvrp/` |
| Single/Parallel Machine | Scheduling | `problems/1_scheduling/parallel_machine/` |
| Bin Packing | Packing | `problems/3_packing_cutting/bin_packing/` |
| CVRP | Routing | `problems/2_routing/cvrp/` |
| Facility Location | Location | `problems/5_location_covering/facility_location/` |
| LP | Continuous | `problems/continuous/linear_programming/` |

## Existing implementations
- `agriculture_crop_selection.py` — Crop portfolio selection as robust portfolio optimization
- `agriculture_crop_transport.py` — Farm-to-market distribution as CVRP
- `agriculture_distribution_center.py` — Distribution center placement as facility location
- `agriculture_equipment_scheduling.py` — Equipment scheduling as parallel machine scheduling
- `agriculture_fertilizer_routing.py` — Fertilizer application routing as VRP
- `agriculture_field_assignment.py` — Field assignment as assignment problem
- `agriculture_harvest_scheduling.py` — Harvest scheduling as machine scheduling
- `agriculture_irrigation_network.py` — Irrigation scheduling as network flow
- `agriculture_pest_control.py` — Pest control routing as VRP
- `agriculture_seed_inventory.py` — Seed inventory as EOQ/Newsvendor
- `agriculture_silo_packing.py` — Silo packing as bin packing
- `agriculture_water_allocation.py` — Water allocation as LP/network flow

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

# Agriculture — Decision Chain

> **Status:** Structure in place — decision points being populated progressively.

## About this sector
Agriculture involves sequential decisions from strategic land-use and crop selection through planting, growing, harvest, post-harvest storage and processing, and market sales. Operations research optimizes resource allocation, scheduling, routing, and inventory at every phase.

## Decision chain

| Phase | Decision point | OR problem | Status |
|---|---|---|---|
| 01 Strategic | Land Use & Crop Selection | LP/MIP, Robust Portfolio | Live |
| 01 Strategic | Field Assignment | Assignment | Live |
| 03 Planting | Seed & Input Procurement | EOQ/Newsvendor | Live |
| 04 Growing | Irrigation Scheduling | LP/Network Flow | Live |
| 04 Growing | Fertilizer Application Routing | VRP | Live |
| 04 Growing | Pest Control Routing | VRP | Live |
| 04 Growing | Water Allocation | LP/Network Flow | Live |
| 05 Harvest | Harvest Scheduling | Machine Scheduling | Live |
| 05 Harvest | Equipment Scheduling | Parallel Machine | Live |
| 06 Post-Harvest | Silo Packing | Bin Packing | Live |
| 06 Post-Harvest | Crop Transport | CVRP | Live |
| 06 Post-Harvest | Distribution Center | Facility Location | Live |

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

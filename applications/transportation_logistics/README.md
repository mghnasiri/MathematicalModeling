# Transportation & Logistics — Decision Chain

> **Status:** Live — 2 implementations; decision points mapped across all 5 phases.

## About this sector

Transportation and logistics involves network design, fleet management, operational routing, crew scheduling, and real-time dispatch. OR methods optimize hub-and-spoke networks, fleet sizing, vehicle routing, driver scheduling, and dynamic re-optimization. This sector is the natural home of VRP variants, which are among the most studied problems in OR — the CVRP alone has generated thousands of papers since Dantzig & Ramser (1959).

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Network | Hub / Depot Location | Hub Location / Facility Location | — | Planned |
| 01 Network | Route Network Design | Network Design | — | Planned |
| 02 Fleet | Fleet Sizing & Mix | Bin Packing / MIP | — | Planned |
| 02 Fleet | Gate Assignment | Assignment / QAP | `logistics_gate_assignment.py` | Live |
| 03 Operational | Parcel Delivery Routing | CVRP / VRPTW | `delivery_routing.py` | Live |
| 03 Operational | Multi-Depot Routing | MDVRP | — | Planned |
| 04 Crew | Driver / Crew Scheduling | Set Covering / Nurse Scheduling | — | Planned |
| 04 Crew | Crew Rostering | Workforce Scheduling | — | Planned |
| 05 Dynamic | Real-Time Dispatch | Dynamic VRP | — | Planned |
| 05 Dynamic | Disruption Re-Routing | Robust Shortest Path | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| CVRP | Routing | `problems/2_routing/cvrp/` |
| VRPTW | Routing | `problems/2_routing/vrptw/` |
| Multi-Depot VRP | Routing | `problems/2_routing/multi_depot_vrp/` |
| Hub Location | Location | `problems/5_location_covering/hub_location/` |
| Facility Location | Location | `problems/5_location_covering/facility_location/` |
| Network Design | Network | `problems/6_network_flow_design/network_design/` |
| Set Covering | Location | `problems/5_location_covering/set_covering/` |
| Assignment / QAP | Assignment | `problems/4_assignment_matching/assignment/` |
| Robust Shortest Path | Uncertainty | `problems/9_uncertainty_modeling/robust_shortest_path/` |

## Existing implementations
- `delivery_routing.py` — Parcel delivery routing as CVRP/VRPTW
- `logistics_gate_assignment.py` — Gate assignment problem

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

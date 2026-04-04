# Public Services — Decision Chain

> **Status:** Live — 1 implementation; decision points mapped across all 4 phases.

## About this sector

Public services include infrastructure planning (schools, fire stations, libraries, transit networks), service scheduling (waste collection, transit timetabling), emergency management (evacuation, disaster response), and resource allocation under budget constraints. OR methods optimize facility coverage, minimize response times, and allocate limited public budgets. Location and covering problems are central — the p-median and MCLP models originated from public-sector applications.

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Infrastructure | Public Facility Location | p-Median / MCLP | — | Planned |
| 01 Infrastructure | Telecom Network Design | Network Design | `telecom_network.py` | Live |
| 01 Infrastructure | Transit Network Design | Shortest Path / Network Design | — | Planned |
| 02 Service Sched. | Waste Collection Routing | Arc Routing (CARP) | — | Planned |
| 02 Service Sched. | Transit Timetabling | Scheduling / Set Covering | — | Planned |
| 03 Emergency | Evacuation Routing | Max Flow / Shortest Path | — | Planned |
| 03 Emergency | Emergency Vehicle Deployment | p-Median / MCLP | — | Planned |
| 04 Resource Alloc. | Budget Allocation | Knapsack / LP | — | Planned |
| 04 Resource Alloc. | Workforce Assignment | Assignment / Set Covering | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| p-Median | Location | `problems/5_location_covering/p_median/` |
| Max Coverage (MCLP) | Location | `problems/5_location_covering/max_coverage/` |
| Network Design | Network | `problems/6_network_flow_design/network_design/` |
| Shortest Path | Network | `problems/6_network_flow_design/shortest_path/` |
| Max Flow | Network | `problems/6_network_flow_design/max_flow/` |
| Arc Routing (CARP) | Routing | `problems/2_routing/arc_routing/` |
| Set Covering | Location | `problems/5_location_covering/set_covering/` |
| Knapsack | Packing | `problems/3_packing_cutting/knapsack/` |
| LP | Continuous | `problems/continuous/linear_programming/` |

## Existing implementations
- `telecom_network.py` — Network design (public infrastructure analogy)

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

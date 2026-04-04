# Healthcare — Decision Chain

> **Status:** Live — 10 implementations across all 5 phases.

## About this sector

Healthcare operations span strategic facility design, staff scheduling, patient flow optimization, supply chain management, and emergency response. OR methods optimize resource utilization, minimize wait times, and improve patient outcomes while respecting regulatory and safety constraints. The sector features highly constrained scheduling (operating rooms, nurses), location problems (ambulance stations), and routing (home visits, supply delivery).

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Strategic | Clinical Trial Planning | Capacity Planning / LP | `healthcare_clinical_trial.py` | Live |
| 02 Staff Planning | Nurse Assignment | Nurse Scheduling / Assignment | `healthcare_nurse_assignment.py` | Live |
| 03 Patient Flow | Operating Room Scheduling | Flexible Job Shop (FJSP) | `healthcare_or_scheduling.py` | Live |
| 03 Patient Flow | Parallel OR Scheduling | Parallel Machine Scheduling | `healthcare_parallel_or.py` | Live |
| 03 Patient Flow | Bed Management | Bin Packing + Scheduling | `healthcare_bed_management.py` | Live |
| 03 Patient Flow | Patient Flow Routing | Network Flow | `healthcare_patient_flow.py` | Live |
| 03 Patient Flow | Home Visit Routing | VRP with Time Windows | `healthcare_home_visits.py` | Live |
| 04 Supply Chain | Medical Supply Delivery | CVRP | `healthcare_supply_delivery.py` | Live |
| 05 Emergency | Ambulance Station Location | MCLP / p-Median | `healthcare_ambulance_location.py` | Live |
| 05 Emergency | Emergency Network Routing | Dynamic VRP | `healthcare_emergency_network.py` | Live |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| FJSP | Scheduling | `problems/1_scheduling/flexible_job_shop/` |
| Parallel Machine | Scheduling | `problems/1_scheduling/parallel_machine/` |
| Nurse Scheduling | Scheduling | `problems/1_scheduling/nurse_scheduling/` |
| CVRP | Routing | `problems/2_routing/cvrp/` |
| VRP with Time Windows | Routing | `problems/2_routing/vrptw/` |
| Bin Packing | Packing | `problems/3_packing_cutting/bin_packing/` |
| Assignment | Assignment | `problems/4_assignment_matching/assignment/` |
| Facility Location (MCLP) | Location | `problems/5_location_covering/max_coverage/` |
| p-Median | Location | `problems/5_location_covering/p_median/` |
| Network Flow | Network | `problems/6_network_flow_design/max_flow/` |

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

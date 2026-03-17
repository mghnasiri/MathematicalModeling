# Healthcare — Decision Chain

> **Status:** Structure in place — decision points being populated progressively.

## About this sector
Healthcare operations involve facility design, staff scheduling, patient flow optimization, supply chain management, and emergency response. OR methods optimize resource utilization while maintaining quality of care.

## Decision chain

| Phase | Decision point | OR problem | Status |
|---|---|---|---|
| 03 Patient Flow | Operating Room Scheduling | FJSP | Live |
| 05 Emergency | Ambulance Location | MCLP/p-Median | Live |
| 05 Emergency | Emergency Network Routing | Dynamic VRP | Live |

## Existing implementations
- `healthcare_ambulance_location.py` — Ambulance station placement as maximum covering location
- `healthcare_bed_management.py` — Bed management as bin packing + scheduling
- `healthcare_clinical_trial.py` — Clinical trial planning as capacity planning
- `healthcare_emergency_network.py` — Emergency network routing as dynamic VRP
- `healthcare_home_visits.py` — Home visit routing as VRP
- `healthcare_nurse_assignment.py` — Nurse assignment as scheduling
- `healthcare_or_scheduling.py` — Operating room scheduling as FJSP
- `healthcare_parallel_or.py` — Parallel OR scheduling
- `healthcare_patient_flow.py` — Patient flow as network flow
- `healthcare_supply_delivery.py` — Medical supply delivery as CVRP

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations

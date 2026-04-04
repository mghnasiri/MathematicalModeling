# Patient Flow Operations

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Healthcare
**Phase:** Patient Flow
**Decision-maker:** OR scheduler / bed manager / care coordinator
**Decision frequency:** Daily

## Decision question
How should patients be admitted, scheduled for surgery, and routed through care pathways?

## OR problem mapping
**Canonical problem(s):** FJSP (OR scheduling), Bin Packing (beds), Network Flow (pathways)
**Implementation:** See:
- [`../../healthcare_or_scheduling.py`](../../healthcare_or_scheduling.py)
- [`../../healthcare_parallel_or.py`](../../healthcare_parallel_or.py)
- [`../../healthcare_bed_management.py`](../../healthcare_bed_management.py)
- [`../../healthcare_patient_flow.py`](../../healthcare_patient_flow.py)
- [`../../healthcare_home_visits.py`](../../healthcare_home_visits.py)

## Key modeling aspects
- Operating room scheduling maps to Flexible Job Shop: surgeries (jobs) require specific equipment/teams (machines) with sequence-dependent setup (room turnover)
- Bed management is a online bin packing problem where patients (items) of varying length-of-stay (size) must fit into ward beds (bins) of fixed capacity
- Home visit routing is a VRPTW: caregivers (vehicles) visit patients (customers) within appointment windows while respecting shift duration

## Data requirements
- Surgery list with estimated durations, resource needs, and surgeon availability
- Ward bed counts, current occupancy, and predicted length-of-stay distributions
- Patient home locations, visit time windows, and service durations
- Travel time matrix between patient addresses for route planning

## Canonical problem
- [Flexible Job Shop Scheduling](../../../problems/1_scheduling/flexible_job_shop/README.md)
- [Bin Packing](../../../problems/3_packing_cutting/bin_packing/README.md)
- [Max Flow / Network Flow](../../../problems/6_network_flow_design/max_flow/README.md)
- [VRPTW](../../../problems/2_routing/vrptw/README.md)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

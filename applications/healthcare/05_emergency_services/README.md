# Emergency Services

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Healthcare
**Phase:** Emergency Services
**Decision-maker:** EMS director / dispatch coordinator
**Decision frequency:** Real-time / strategic

## Decision question
Where should ambulance stations be located, and how should ambulances be dispatched?

## OR problem mapping
**Canonical problem(s):** MCLP / p-Median (location), Dynamic VRP (dispatch)
**Implementation:** See:
- [`../../healthcare_ambulance_location.py`](../../healthcare_ambulance_location.py)
- [`../../healthcare_emergency_network.py`](../../healthcare_emergency_network.py)

## Key modeling aspects
- Ambulance station placement is a covering/p-median problem: locate p stations to minimize worst-case or average response time across demand zones
- Real-time dispatch is a dynamic VRP where new emergency calls arrive stochastically and idle ambulances must be re-positioned for coverage
- Response time targets (e.g., 8-minute threshold) translate into maximum covering constraints with probabilistic demand

## Data requirements
- Geographic demand zones with historical call volumes and temporal patterns
- Candidate station sites with fixed and operating costs
- Road-network travel times (peak and off-peak) between zones and stations
- Fleet size, ambulance availability, and average on-scene service duration

## Canonical problem
- [p-Median](../../../problems/5_location_covering/p_median/README.md)
- [Facility Location (MCLP)](../../../problems/5_location_covering/facility_location/README.md)
- [CVRP / Dynamic VRP](../../../problems/2_routing/cvrp/README.md)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

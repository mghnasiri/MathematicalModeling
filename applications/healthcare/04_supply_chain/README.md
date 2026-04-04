# Healthcare Supply Chain

> **Status:** Placeholder — see existing implementation.

## Sector context
**Sector:** Healthcare
**Phase:** Supply Chain
**Decision-maker:** Supply chain manager
**Decision frequency:** Weekly

## Decision question
How should medical supplies be distributed across facilities?

## OR problem mapping
**Canonical problem(s):** CVRP / IRP
**Implementation:** See [`../../healthcare_supply_delivery.py`](../../healthcare_supply_delivery.py)

## Key modeling aspects
- Medical supply delivery is a CVRP: delivery vehicles with weight/volume limits serve clinics from a central warehouse, minimizing total travel cost
- Temperature-sensitive items (vaccines, blood products) add time-window constraints, pushing the model toward VRPTW
- Demand uncertainty for consumables (PPE, medications) requires safety stock or stochastic recourse to avoid stockouts

## Data requirements
- Facility locations (warehouse, clinics) and road-network distance/time matrix
- Weekly demand volumes per facility and item category
- Vehicle fleet size, payload capacity, and operating cost per kilometer
- Item storage constraints (cold chain, hazmat) and delivery time windows

## Canonical problem
- [Capacitated VRP (CVRP)](../../../problems/2_routing/cvrp/README.md)
- [VRPTW](../../../problems/2_routing/vrptw/README.md)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

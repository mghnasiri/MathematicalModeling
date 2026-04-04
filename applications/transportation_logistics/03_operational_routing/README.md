# Operational Routing

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Transportation & Logistics
**Phase:** Operational Routing
**Decision-maker:** Dispatch planner
**Decision frequency:** Daily

## Decision question
How should vehicles be routed to deliver parcels efficiently?

## OR problem mapping
**Canonical problem(s):** CVRP / VRPTW
**Implementation:** See [`../../delivery_routing.py`](../../delivery_routing.py)

## Key modeling aspects
- Daily parcel delivery is a capacitated vehicle routing problem: partition customers into vehicle-feasible routes minimizing total travel distance
- Time window commitments (e.g., morning delivery slots) add VRPTW constraints requiring careful sequencing within each route
- Stochastic demand and travel times introduce recourse decisions, linking to stochastic VRP formulations

## Data requirements
- Customer locations (geocoded addresses) and daily demand volumes
- Vehicle fleet capacity, count, and depot location(s)
- Delivery time windows and service time per stop
- Road network travel times and distances (or distance matrix)

## Canonical problem
- [CVRP](../../../problems/2_routing/cvrp/README.md)
- [VRPTW](../../../problems/2_routing/vrptw/README.md)
- [Stochastic VRP](../../../problems/9_uncertainty_modeling/stochastic_vrp/README.md)

## Status
This decision point is part of the **Transportation & Logistics** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

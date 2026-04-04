# Last-Mile Delivery

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Retail & E-Commerce
**Phase:** Last-Mile Delivery
**Decision-maker:** Logistics manager
**Decision frequency:** Daily

## Decision question
How should delivery routes be planned to reach customers efficiently within time windows?

## OR problem mapping
**Canonical problem(s):** VRPTW

## Key modeling aspects

- Last-mile delivery is a capacitated vehicle routing problem: partition customers into routes for a fleet of vehicles subject to payload limits.
- Customer delivery windows (e.g., "2-4 PM") introduce time-window constraints, turning the problem into a VRPTW with waiting time and late-arrival penalties.
- Heterogeneous fleets (vans, bikes, drones) and multi-depot setups extend the base CVRP with vehicle-type-dependent costs and range limits.

## Data requirements

- **Delivery requests** -- geocoded customer addresses, parcel counts/weights, and requested time windows.
- **Fleet profile** -- number of vehicles per type, capacity (weight and volume), speed profiles, and operating cost per km.
- **Road network** -- travel-time and distance matrices (or API-based routing) accounting for traffic patterns by time of day.
- **Depot locations** -- coordinates and operating hours of dispatch points (warehouses or micro-hubs).

## Canonical problem

- [CVRP](../../../problems/2_routing/cvrp/README.md)
- [VRPTW](../../../problems/2_routing/vrptw/README.md)
- [Multi-Depot VRP](../../../problems/2_routing/multi_depot_vrp/README.md)

## Status
This decision point is part of the **Retail & E-Commerce** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

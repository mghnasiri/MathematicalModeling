# Service Scheduling

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Public Services
**Phase:** Service Scheduling
**Decision-maker:** Service manager
**Decision frequency:** Monthly

## Decision question
How should court hearings, permit reviews, or public services be scheduled?

## OR problem mapping
**Canonical problem(s):** Parallel Machine Scheduling

## Key modeling aspects
- Waste collection routes must traverse every street (arc) in a district, mapping to the Capacitated Arc Routing Problem (CARP) where vehicle capacity limits route length
- Transit timetabling requires selecting a minimum set of departure times that cover all passenger origin-destination pairs, a Set Covering formulation
- Court and permit scheduling assigns cases to available rooms/officers over time slots, naturally modeled as Parallel Machine Scheduling with due dates

## Data requirements
- Street network graph with arc distances and service requirements (for waste collection)
- Passenger demand matrix with origin-destination pairs and time preferences (for timetabling)
- Case/task list with processing times, due dates, and resource requirements
- Staff and equipment availability calendars

## Canonical problem
- [CVRP](../../../problems/2_routing/cvrp/README.md) -- capacitated vehicle routing as a node-routing proxy for collection routes
- [Set Covering](../../../problems/5_location_covering/set_covering/README.md) -- select minimum-cost set of services to cover all demands
- [Parallel Machine Scheduling](../../../problems/1_scheduling/parallel_machine/README.md) -- assign tasks to parallel resources to meet deadlines

## Status
This decision point is part of the **Public Services** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

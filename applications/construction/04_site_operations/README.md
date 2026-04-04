# Site Operations

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Construction
**Phase:** Site Operations
**Decision-maker:** Site manager
**Decision frequency:** Daily

## Decision question
How should crews and equipment be scheduled and routed across the construction site?

## OR problem mapping
**Canonical problem(s):** GAP / VRP

## Key modeling aspects
- Assigning specialized crews to tasks across multiple zones is a generalized assignment problem (GAP) with skill and availability constraints.
- Routing heavy equipment (cranes, concrete pumps) between zones minimizes idle travel time — a capacitated VRP with time windows.
- Daily re-planning is needed as weather, material delays, and task completions change the feasible set.

## Data requirements
- Crew roster with skill sets, availability windows, and hourly costs.
- Task list with location, duration, required skills, and time windows.
- Site layout graph with travel times between work zones and staging areas.
- Equipment fleet with capacities and mobilization costs.

## Canonical problem
See [CVRP](../../../problems/2_routing/cvrp/README.md)

## Status
This decision point is part of the **Construction** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

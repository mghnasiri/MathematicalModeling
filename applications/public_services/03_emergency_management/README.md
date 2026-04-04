# Emergency Management

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Public Services
**Phase:** Emergency Management
**Decision-maker:** Emergency director
**Decision frequency:** Ongoing

## Decision question
How should patrol routes and evacuation plans be designed to maximize coverage and safety?

## OR problem mapping
**Canonical problem(s):** CARP / Network Flow

## Key modeling aspects
- Evacuation planning seeks the maximum number of people moved from danger zones to shelters per unit time, directly modeled as a Max Flow problem on the road network
- Optimal evacuation paths for individuals or groups correspond to Shortest Path queries on time-dependent or congestion-aware graphs
- Pre-positioning emergency vehicles (ambulances, fire trucks) to minimize worst-case response time is a p-Median location problem with coverage constraints

## Data requirements
- Road network graph with edge capacities (lanes, speed limits) and travel times
- Population density at risk zones and shelter locations with capacities
- Emergency vehicle fleet size, depot locations, and response time targets
- Historical incident data for demand forecasting and scenario generation

## Canonical problem
- [Max Flow](../../../problems/6_network_flow_design/max_flow/README.md) -- maximize evacuation throughput from sources to sinks
- [Shortest Path](../../../problems/6_network_flow_design/shortest_path/README.md) -- find fastest evacuation routes under network constraints
- [p-Median](../../../problems/5_location_covering/p_median/README.md) -- position emergency vehicles to minimize response distance

## Status
This decision point is part of the **Public Services** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

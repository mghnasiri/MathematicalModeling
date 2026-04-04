# Energy Infrastructure

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Energy
**Phase:** Infrastructure
**Decision-maker:** Energy planner
**Decision frequency:** Multi-year

## Decision question
Where should power plants and network infrastructure be sited?

## OR problem mapping
**Canonical problem(s):** Facility Location

## Key modeling aspects
- Power plant siting is a classic **uncapacitated facility location** problem: choose which candidate sites to open, minimizing fixed construction costs plus transmission costs to demand nodes
- Transmission network design maps to **minimum spanning tree / network design**: connect substations with minimum total cable length while ensuring connectivity and redundancy
- Long planning horizons introduce multi-period variants where facility opening decisions are staged over years, linking to capacitated facility location with time-indexed variables

## Data requirements
- Candidate plant sites with fixed construction costs and generation capacities
- Customer demand nodes (cities, industrial zones) with peak load estimates (MW)
- Pairwise transmission distances or costs between candidate sites and demand nodes
- Terrain, regulatory, and environmental constraints that restrict eligible locations

## Canonical problem
See [Facility Location](../../../problems/5_location_covering/facility_location/README.md) and [Minimum Spanning Tree](../../../problems/6_network_flow_design/min_spanning_tree/README.md)

## Status
This decision point is part of the **Energy** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

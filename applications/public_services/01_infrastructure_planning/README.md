# Infrastructure Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Public Services
**Phase:** Infrastructure Planning
**Decision-maker:** City planner
**Decision frequency:** Multi-year

## Decision question
Where should public facilities (schools, fire stations, libraries) be located to maximize coverage?

## OR problem mapping
**Canonical problem(s):** p-Median / MCLP

## Key modeling aspects
- Siting decisions are binary (open/close) with distance-dependent service quality, mapping directly to p-Median and Maximal Covering Location Problem (MCLP) formulations
- Transit network expansion involves selecting edges (routes) and nodes (stops) to maximize connectivity under budget, a classic Network Design problem
- Equity constraints (e.g., every resident within k minutes of a fire station) translate to covering constraints in the integer program

## Data requirements
- Candidate facility locations with fixed opening costs
- Demand points (population centroids, census blocks) with weights
- Travel time or distance matrix between demand points and candidate sites
- Budget caps, capacity limits, and minimum coverage thresholds

## Canonical problem
- [p-Median](../../../problems/5_location_covering/p_median/README.md) -- minimize weighted distance to nearest facility
- [Facility Location](../../../problems/5_location_covering/facility_location/README.md) -- uncapacitated facility location with fixed costs
- [Network Design](../../../problems/6_network_flow_design/network_design/README.md) -- select edges to build a cost-effective transit network

## Status
This decision point is part of the **Public Services** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

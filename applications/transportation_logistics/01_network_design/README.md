# Network Design

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Transportation & Logistics
**Phase:** Network Design
**Decision-maker:** Network planner
**Decision frequency:** Multi-year

## Decision question
Where should hubs and depots be located to minimize total network cost?

## OR problem mapping
**Canonical problem(s):** Hub Location / CFLP

## Key modeling aspects
- Hub/depot placement is a facility location problem: trade off fixed opening costs against transportation costs to minimize total network cost
- Route network topology (direct vs. hub-and-spoke) maps to network design with capacity constraints on arcs and nodes
- Multi-year horizon introduces demand forecasting uncertainty, linking to stochastic and robust facility location variants

## Data requirements
- Candidate site locations with fixed opening costs and capacities
- Origin-destination demand forecasts (volume, weight) by time period
- Transportation cost rates per unit-distance between all node pairs
- Infrastructure constraints (zoning, permits, maximum facility counts)

## Canonical problem
- [Facility Location (UFLP)](../../../problems/5_location_covering/facility_location/README.md)
- [p-Median](../../../problems/5_location_covering/p_median/README.md)
- [Network Design](../../../problems/6_network_flow_design/network_design/README.md)

## Status
This decision point is part of the **Transportation & Logistics** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

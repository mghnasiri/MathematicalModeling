# Plant Location & Design

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Manufacturing
**Phase:** Strategic Planning
**Decision-maker:** Plant manager
**Decision frequency:** Annual

## Decision question
Where should manufacturing plants be built or expanded?

## OR problem mapping
**Canonical problem(s):** CFLP / Network Design

## Key modeling aspects
- Plant location is a capacitated facility location problem (CFLP): select which sites to open while respecting production-capacity limits at each site
- Fixed costs (land, construction) vs. variable transport costs create the classic trade-off optimized by facility-location models
- Demand uncertainty and long planning horizons often require stochastic or robust extensions (scenario-based CFLP)

## Data requirements
- Candidate site locations with fixed opening costs and capacity limits
- Customer/demand-zone locations with forecasted annual demand volumes
- Per-unit transportation costs (or distances) between each site-customer pair
- Optional: labor costs, tax incentives, and lead-time constraints per site

## Canonical problem
- [Facility Location (UFLP)](../../../problems/5_location_covering/facility_location/README.md)
- [p-Median](../../../problems/5_location_covering/p_median/README.md)
- [Network Design](../../../problems/6_network_flow_design/network_design/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

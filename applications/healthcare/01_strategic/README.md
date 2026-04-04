# Strategic Healthcare Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Healthcare
**Phase:** Strategic Planning
**Decision-maker:** Hospital administrator / health system planner
**Decision frequency:** Annual / multi-year

## Decision question
Where should healthcare facilities be located, and how should department capacities be planned?

## OR problem mapping
**Canonical problem(s):** CFLP / Hub Location, Stochastic Programming

## Key modeling aspects
- Facility location decisions (where to build, what capacity) map directly to Capacitated Facility Location with fixed opening costs and demand coverage constraints
- Clinical trial resource planning is a two-stage stochastic program: commit resources (stage 1) before uncertain patient enrollment and outcomes (stage 2)
- Long planning horizons require demand forecasting under uncertainty, making stochastic and robust formulations essential

## Data requirements
- Population demographics and projected patient demand by region
- Candidate site locations, construction/lease costs, and maximum capacities
- Travel time or distance matrices between population centers and candidate sites
- Historical demand variability for stochastic scenario generation

## Canonical problem
- [Facility Location (CFLP)](../../../problems/5_location_covering/facility_location/README.md)
- [Two-Stage Stochastic Programming](../../../problems/9_uncertainty_modeling/two_stage_sp/README.md)
- [Linear Programming](../../../problems/continuous/linear_programming/README.md)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

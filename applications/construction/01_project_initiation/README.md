# Project Initiation

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Construction
**Phase:** Project Initiation
**Decision-maker:** Project director
**Decision frequency:** Per project

## Decision question
Which site should be selected for a new construction project?

## OR problem mapping
**Canonical problem(s):** Facility Location / Knapsack

## Key modeling aspects
- Each candidate site has a fixed opening cost and variable costs that depend on which demands it serves — a classic uncapacitated facility location structure.
- Site selection often includes a budget or cardinality constraint (open at most k sites), adding a knapsack-like dimension.
- Proximity to labor, suppliers, and transport links translates directly to assignment costs in the UFLP objective.

## Data requirements
- List of candidate sites with fixed opening costs (land, permits, mobilization).
- Demand points (project zones or client locations) with estimated volumes.
- Distance or travel-time matrix between candidates and demand points.
- Budget ceiling or maximum number of sites to open.

## Canonical problem
See [Facility Location](../../../problems/5_location_covering/facility_location/README.md)

## Status
This decision point is part of the **Construction** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

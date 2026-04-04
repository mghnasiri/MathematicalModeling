# Crew Scheduling

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Transportation & Logistics
**Phase:** Crew Scheduling
**Decision-maker:** Operations manager
**Decision frequency:** Monthly

## Decision question
How should driver duties and shifts be assigned to cover all routes?

## OR problem mapping
**Canonical problem(s):** Set Partitioning

## Key modeling aspects
- Driver duty generation and selection is a set covering/partitioning problem: find a minimum-cost collection of feasible shifts that covers all scheduled route legs
- Rostering over a planning horizon must respect labor regulations (maximum hours, mandatory rest, consecutive days off), adding side constraints to the partitioning model
- Crew pairing links trips into multi-day sequences, combining combinatorial enumeration of feasible pairings with set covering to select the final roster

## Data requirements
- Route timetable with trip start/end times and locations
- Driver pool with qualifications, home bases, and contract types
- Labor rules (max driving hours, minimum rest periods, overtime thresholds)
- Pay rates and penalty costs for overtime, split shifts, and deadheading

## Canonical problem
- [Set Covering](../../../problems/5_location_covering/set_covering/README.md)
- [Assignment (LAP)](../../../problems/4_assignment_matching/assignment/README.md)

## Status
This decision point is part of the **Transportation & Logistics** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

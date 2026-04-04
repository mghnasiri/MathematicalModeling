# Fleet Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Transportation & Logistics
**Phase:** Fleet Planning
**Decision-maker:** Fleet manager
**Decision frequency:** Annual

## Decision question
What should the fleet size and vehicle composition be to meet demand efficiently?

## OR problem mapping
**Canonical problem(s):** MIP

## Key modeling aspects
- Fleet sizing determines the minimum number of vehicles of each type needed to cover projected demand, mapping to bin packing (items = demand, bins = vehicles) and MIP formulations
- Gate/dock assignment allocates arriving vehicles to limited terminal resources, a classic assignment or quadratic assignment problem minimizing turnaround time
- Vehicle mix decisions balance acquisition cost against operational flexibility, requiring multi-period capacity planning under demand uncertainty

## Data requirements
- Demand forecasts by route, season, and vehicle type
- Vehicle specifications (capacity, operating cost, range, maintenance schedule)
- Terminal gate/dock counts, dimensions, and availability windows
- Capital and leasing cost structures for each vehicle class

## Canonical problem
- [Bin Packing](../../../problems/3_packing_cutting/bin_packing/README.md)
- [Assignment (LAP)](../../../problems/4_assignment_matching/assignment/README.md)
- [Quadratic Assignment (QAP)](../../../problems/4_assignment_matching/quadratic_assignment/README.md)

## Status
This decision point is part of the **Transportation & Logistics** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

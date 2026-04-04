# Production Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Manufacturing
**Phase:** Production Planning
**Decision-maker:** Production planner
**Decision frequency:** Weekly / Monthly

## Decision question
What products should be produced and when?

## OR problem mapping
**Canonical problem(s):** LP / CLSP

## Key modeling aspects
- Deciding production quantities and timing maps directly to lot-sizing models (CLSP/DLSP), balancing setup costs against inventory holding costs
- Material requirements planning (MRP) decomposes finished-goods demand into component needs via bill-of-materials explosion
- Capacity constraints on machines or labor convert the uncapacitated model into the capacitated lot-sizing problem (CLSP)

## Data requirements
- Demand forecasts per product per planning period (weekly/monthly)
- Setup costs and setup times for each product on each production line
- Per-unit holding costs and production costs
- Bill-of-materials structure and component lead times

## Canonical problem
- [Capacitated Lot Sizing (CLSP)](../../../problems/7_inventory_lotsizing/capacitated_lot_sizing/README.md)
- [Wagner-Whitin](../../../problems/7_inventory_lotsizing/wagner_whitin/README.md)
- [Lot Sizing (Silver-Meal)](../../../problems/7_inventory_lotsizing/lot_sizing/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

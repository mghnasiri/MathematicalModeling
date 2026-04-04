# Order Fulfillment

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Retail & E-Commerce
**Phase:** Fulfillment
**Decision-maker:** Warehouse manager
**Decision frequency:** Daily

## Decision question
How should orders be picked, packed, and staged for shipment?

## OR problem mapping
**Canonical problem(s):** TSP / Bin Packing

## Key modeling aspects

- Order picking maps to a TSP/assignment hybrid: determine the picker route through warehouse aisles (TSP) and assign orders to pickers or waves (assignment).
- Packing multi-item orders into shipping boxes is a bin packing problem: minimize the number of cartons while respecting weight and volume limits.
- Batch scheduling of pick waves and staging lanes is a parallel-machine scheduling problem with release dates and due-date (carrier pickup) constraints.

## Data requirements

- **Order pool** -- current open orders with item lists, quantities, weights, and promised ship-by times.
- **Warehouse layout** -- aisle/slot locations for each SKU, zone definitions, and travel-time matrix between pick locations.
- **Packing specs** -- available carton dimensions and weight limits, item dimensions and fragility flags.
- **Resource availability** -- number of pickers, packing stations, and staging lanes per shift.

## Canonical problem

- [Bin Packing](../../../problems/3_packing_cutting/bin_packing/README.md)
- [Assignment](../../../problems/4_assignment_matching/assignment/README.md)
- [TSP](../../../problems/2_routing/tsp/README.md)
- [Batch Scheduling](../../../problems/1_scheduling/batch_scheduling/README.md)

## Status
This decision point is part of the **Retail & E-Commerce** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

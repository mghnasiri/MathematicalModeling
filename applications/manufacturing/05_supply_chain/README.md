# Manufacturing Supply Chain

> **Status:** Placeholder — see existing implementation.

## Sector context
**Sector:** Manufacturing
**Phase:** Supply Chain
**Decision-maker:** Procurement manager
**Decision frequency:** Weekly

## Decision question
How should suppliers be selected and materials transported to the plant?

## OR problem mapping
**Canonical problem(s):** CVRP / QAP
**Implementation:** See [`../../supply_chain_network.py`](../../supply_chain_network.py)

## Key modeling aspects
- Supplier selection and inbound logistics combine facility location (which warehouses to use) with vehicle routing (how to transport materials)
- Network design determines the number, location, and capacity of distribution centers to minimize total logistics cost
- Demand variability in multi-echelon networks requires safety-stock placement and inventory positioning decisions

## Data requirements
- Supplier locations, capacities, lead times, and per-unit costs
- Warehouse/DC candidate sites with fixed costs and throughput limits
- Customer demand volumes and geographic locations
- Transportation costs, vehicle capacities, and delivery time windows

## Canonical problem
- [Facility Location (UFLP)](../../../problems/5_location_covering/facility_location/README.md)
- [CVRP](../../../problems/2_routing/cvrp/README.md)
- [Multi-Echelon Inventory](../../../problems/7_inventory_lotsizing/multi_echelon_inventory/README.md)
- [Network Design](../../../problems/6_network_flow_design/network_design/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

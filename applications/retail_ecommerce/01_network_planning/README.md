# Network Planning

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Retail & E-Commerce
**Phase:** Network Planning
**Decision-maker:** Supply chain director
**Decision frequency:** Annual

## Decision question
Where should stores and warehouses be located to serve customers efficiently?

## OR problem mapping
**Canonical problem(s):** p-Median / CFLP
**Implementation:** See [`../../warehouse_location.py`](../../warehouse_location.py)

## Key modeling aspects

- Warehouse placement is a classical facility location problem: trade off fixed opening costs against transportation costs to demand zones.
- Demand aggregation at the ZIP/region level lets you model customer coverage as weighted assignment to the nearest open facility.
- Capacity constraints on warehouse throughput convert the uncapacitated (UFLP) formulation into a capacitated one (CFLP), requiring additional binary/continuous variables.

## Data requirements

- **Candidate sites** -- coordinates, fixed annual costs, and throughput capacity for each potential warehouse.
- **Customer demand** -- geocoded demand points (or regional aggregates) with expected annual volume.
- **Transportation costs** -- per-unit shipping cost matrix between each candidate site and each demand zone.
- **Service-level constraints** -- maximum acceptable delivery distance or lead time from warehouse to customer.

## Canonical problem

- [Facility Location (UFLP)](../../../problems/5_location_covering/facility_location/README.md)
- [Capacitated Facility Location](../../../problems/5_location_covering/facility_location/variants/capacitated/README.md)
- [p-Median](../../../problems/5_location_covering/p_median/README.md)

## Status
This decision point is part of the **Retail & E-Commerce** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

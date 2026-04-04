# Post-Harvest Operations

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Agriculture
**Phase:** Post-Harvest
**Decision-maker:** Logistics and storage manager
**Decision frequency:** Weekly

## Decision question
How should harvested crops be stored, packed, and distributed to markets?

## OR problem mapping
**Canonical problem(s):** Bin Packing (silo), CVRP (transport), Facility Location (distribution center)
**Implementation:** See:
- [`../../agriculture_silo_packing.py`](../../agriculture_silo_packing.py)
- [`../../agriculture_crop_transport.py`](../../agriculture_crop_transport.py)
- [`../../agriculture_distribution_center.py`](../../agriculture_distribution_center.py)

## Key modeling aspects
- Assigning crop lots of varying sizes to fixed-capacity silos is a bin packing problem minimizing wasted space
- Delivering crops from farm to markets/processors with truck capacity limits is a CVRP
- Choosing which distribution centers to open balances fixed facility costs against transport distances

## Data requirements
- Crop lot volumes/weights per field and silo capacities with compatibility constraints
- Truck fleet size, payload capacity, and travel distance/time matrix between locations
- Candidate distribution center sites with opening costs and throughput capacities
- Customer/market locations and weekly demand volumes

## Canonical problem
- [Bin Packing](../../../problems/3_packing_cutting/bin_packing/README.md) -- silo allocation for harvested crop lots
- [CVRP](../../../problems/2_routing/cvrp/README.md) -- crop transport routing under vehicle capacity
- [Facility Location](../../../problems/5_location_covering/facility_location/README.md) -- distribution center selection

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

# Seed & Input Procurement

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Agriculture
**Phase:** Planting Season
**Decision-maker:** Procurement officer
**Decision frequency:** Once per planting season

## Decision question
How much seed and input materials should be ordered, and when, to minimize procurement and holding costs?

## OR problem mapping
**Canonical problem(s):** EOQ / Lot Sizing / Newsvendor
**Implementation:** See [`../../agriculture_seed_inventory.py`](../../agriculture_seed_inventory.py)

## Key modeling aspects
- Seed order quantity under uncertain planting-window weather maps directly to the newsvendor critical fractile
- Bulk discount tiers and storage capacity limits extend the problem to EOQ with quantity breaks
- Lead times from suppliers combined with narrow planting windows create lot-sizing trade-offs

## Data requirements
- Seed demand per crop (acres planned x seeding rate) and demand variability
- Supplier pricing (unit cost, bulk discount tiers, shipping lead times)
- Holding cost for seed storage and shelf-life / viability decay rates
- Planting window calendar and weather-driven demand uncertainty distributions

## Canonical problem
- [Newsvendor](../../../problems/9_uncertainty_modeling/newsvendor/README.md) -- single-period order quantity under demand uncertainty
- [EOQ](../../../problems/7_inventory_lotsizing/eoq/README.md) -- classic order quantity with holding and ordering costs
- [Wagner-Whitin](../../../problems/7_inventory_lotsizing/wagner_whitin/README.md) -- dynamic lot sizing over the planting horizon

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

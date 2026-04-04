# Procurement

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Construction
**Phase:** Procurement
**Decision-maker:** Procurement officer
**Decision frequency:** Monthly

## Decision question
How much material should be ordered and when to minimize costs and avoid delays?

## OR problem mapping
**Canonical problem(s):** Lot Sizing / EOQ

## Key modeling aspects
- Material orders must balance ordering costs (purchase orders, delivery fees) against holding costs (on-site storage, spoilage) — the classic EOQ trade-off.
- Time-varying demand driven by the project schedule turns the problem into dynamic lot sizing (Wagner-Whitin / Silver-Meal).
- Capacity constraints on storage yards or supplier lead times add feasibility restrictions beyond the basic EOQ model.

## Data requirements
- Bill of materials with per-period demand quantities derived from the project schedule.
- Per-item ordering cost, unit cost, and holding cost rate.
- Supplier lead times and any quantity-discount schedules.
- On-site storage capacity limits.

## Canonical problem
See [EOQ](../../../problems/7_inventory_lotsizing/eoq/README.md)

## Status
This decision point is part of the **Construction** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

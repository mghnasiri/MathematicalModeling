# Inventory Planning

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Retail & E-Commerce
**Phase:** Inventory Planning
**Decision-maker:** Category manager
**Decision frequency:** Weekly

## Decision question
How much inventory should be ordered and when to minimize costs while meeting demand?

## OR problem mapping
**Canonical problem(s):** Newsvendor / (s,S)
**Implementation:** See [`../../retail_inventory.py`](../../retail_inventory.py)

## Key modeling aspects

- Perishable and seasonal SKUs with a single ordering opportunity map directly to the newsvendor model: balance overage versus underage cost under demand uncertainty.
- Replenishable SKUs with recurring orders follow an (s,S) or EOQ policy, where reorder points and order-up-to levels are set from demand forecasts and lead-time distributions.
- Safety-stock calculations bridge the gap between deterministic lot-sizing and stochastic demand, ensuring a target service level (e.g., 95% fill rate).

## Data requirements

- **Demand history** -- point-of-sale data per SKU/location at daily or weekly granularity for forecast generation.
- **Cost parameters** -- unit purchase cost, holding cost rate, stockout/backorder penalty, and salvage value.
- **Lead times** -- supplier lead-time distributions (mean and variance) for each SKU or vendor.
- **Storage constraints** -- shelf-space or warehouse capacity limits that cap order quantities.

## Canonical problem

- [Newsvendor](../../../problems/9_uncertainty_modeling/newsvendor/README.md)
- [EOQ](../../../problems/7_inventory_lotsizing/eoq/README.md)
- [Safety Stock](../../../problems/7_inventory_lotsizing/safety_stock/README.md)
- [Lot Sizing](../../../problems/7_inventory_lotsizing/lot_sizing/README.md)

## Status
This decision point is part of the **Retail & E-Commerce** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

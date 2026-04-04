# Market Timing & Sales

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Agriculture
**Phase:** Market & Sales
**Decision-maker:** Sales manager / cooperative director
**Decision frequency:** Weekly to monthly

## Decision question
When and at what price should harvested crops be sold to maximize revenue under price uncertainty?

## OR problem mapping
**Canonical problem(s):** Stochastic DP / Robust Optimization

## Key modeling aspects
- Deciding when to sell stored crops under price volatility is a stochastic dynamic programming problem
- Robust optimization hedges against worst-case price scenarios when distributions are ambiguous
- Inventory holding costs for unsold crops create a newsvendor-like trade-off between selling now and waiting

## Data requirements
- Historical and forecast commodity price series with volatility estimates
- Storage costs (per unit per period) and spoilage/quality degradation rates
- Contract options (spot price, forward contracts, minimum quantities)
- Market demand forecasts and any seasonal price patterns

## Canonical problem
- [Newsvendor](../../../problems/9_uncertainty_modeling/newsvendor/README.md) -- sell-now vs. hold-for-later under price uncertainty
- [Robust Portfolio](../../../problems/9_uncertainty_modeling/robust_portfolio/README.md) -- hedging across multiple market channels
- [DRO](../../../problems/9_uncertainty_modeling/dro/README.md) -- distributionally robust pricing when price distribution is ambiguous

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

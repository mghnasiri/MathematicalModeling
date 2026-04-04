# Seasonal Workforce Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Agriculture
**Phase:** Workforce Planning
**Decision-maker:** Farm operations manager
**Decision frequency:** Once per season

## Decision question
How many seasonal workers should be hired, and when, to meet labor demands across planting, growing, and harvest phases?

## OR problem mapping
**Canonical problem(s):** LP / Stochastic Programming

## Key modeling aspects
- Seasonal labor demand profiles create a multi-period staffing problem solvable as an LP or stochastic program
- Hiring/firing costs and minimum contract lengths introduce integer constraints similar to lot sizing
- Uncertain weather shifts peak labor windows, making two-stage stochastic programming a natural fit

## Data requirements
- Projected labor demand per phase (planting, growing, harvest) in worker-hours
- Hiring, training, and overtime cost rates for each worker category
- Historical weather data to model demand variability across scenarios
- Legal constraints on working hours, contract duration, and seasonal visa limits

## Canonical problem
- [Two-Stage Stochastic Programming](../../../problems/9_uncertainty_modeling/two_stage_sp/README.md) -- hire first stage, adjust second stage
- [Linear Programming](../../../problems/continuous/linear_programming/README.md) -- deterministic workforce allocation
- [Lot Sizing](../../../problems/7_inventory_lotsizing/lot_sizing/README.md) -- analogous hiring/release cost structure

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

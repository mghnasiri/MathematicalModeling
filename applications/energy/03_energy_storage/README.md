# Energy Storage

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Energy
**Phase:** Energy Storage
**Decision-maker:** Storage operator
**Decision frequency:** Hourly

## Decision question
When should energy be stored and when should it be discharged to the grid?

## OR problem mapping
**Canonical problem(s):** LP / MIP

## Key modeling aspects
- **Battery dispatch** is a multi-period LP/inventory problem: decide charge/discharge quantities each hour to maximize arbitrage revenue (buy low, sell high) subject to state-of-charge dynamics
- **Storage sizing** maps to a knapsack-like problem: select battery capacity (MW/MWh) from discrete options to maximize net present value subject to a capital budget
- Uncertainty in renewable output and electricity prices makes this a natural candidate for stochastic or robust formulations with recourse actions at each time step

## Data requirements
- Hourly electricity price forecasts or historical price series
- Battery parameters: power rating (MW), energy capacity (MWh), round-trip efficiency, degradation rates
- Renewable generation forecasts co-located with storage (if applicable)

## Canonical problem
See [Knapsack](../../../problems/3_packing_cutting/knapsack/README.md) and [Linear Programming](../../../problems/continuous/linear_programming/README.md)

## Status
This decision point is part of the **Energy** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

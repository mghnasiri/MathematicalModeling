# Generation Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Energy
**Phase:** Generation Planning
**Decision-maker:** Grid operator
**Decision frequency:** Daily

## Decision question
Which generating units should be committed and how should power be dispatched?

## OR problem mapping
**Canonical problem(s):** MIP / LP

## Key modeling aspects
- **Unit commitment** is a large-scale MIP: binary on/off decisions for each generator at each time period, subject to minimum up/down times, ramp rates, and demand balance
- **Economic dispatch** (given committed units) reduces to an LP that allocates load across online generators to minimize fuel cost while respecting capacity and transmission limits
- Long-term **capacity expansion** decides which new generation assets to build over a planning horizon, combining network design with LP relaxations of investment and operating costs

## Data requirements
- Generator fleet data: capacity (MW), heat rates, fuel costs, ramp rates, min up/down times
- Hourly or sub-hourly load forecast for the planning horizon
- Renewable generation profiles (wind, solar) with intermittency scenarios
- Transmission network topology and line capacities

## Canonical problem
See [Linear Programming](../../../problems/continuous/linear_programming/README.md)

## Status
This decision point is part of the **Energy** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

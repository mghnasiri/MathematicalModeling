# Distribution

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Energy
**Phase:** Distribution
**Decision-maker:** Grid engineer
**Decision frequency:** Real-time

## Decision question
How should electrical load be balanced across the distribution network?

## OR problem mapping
**Canonical problem(s):** Network Flow
**Implementation:** See [`../../energy_grid.py`](../../energy_grid.py)

## Key modeling aspects
- **Load balancing** across feeders and substations is a network flow problem: route power from supply nodes (generators, substations) to demand nodes (customers) through a capacitated distribution graph
- Real-time switching and reconfiguration decisions minimize losses and prevent overloads, mapping to min-cost flow with side constraints on voltage and thermal limits
- Fault isolation and service restoration after outages reduce to maximum-flow / minimum-cut analysis on the distribution topology

## Data requirements
- Distribution network topology: buses, feeders, transformers, and line impedances
- Real-time load measurements (MW, MVAR) at substations and major customers
- Line capacity ratings (thermal limits) and voltage bounds per bus
- Generator injection points and their available output

## Canonical problem
See [Max Flow](../../../problems/6_network_flow_design/max_flow/README.md)

## Status
This decision point is part of the **Energy** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

# Dynamic Operations

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Transportation & Logistics
**Phase:** Dynamic Operations
**Decision-maker:** Dispatcher
**Decision frequency:** Real-time

## Decision question
How should vehicles be dispatched in real-time to respond to new requests?

## OR problem mapping
**Canonical problem(s):** Dynamic VRP

## Key modeling aspects
- Real-time dispatch assigns incoming service requests to active vehicles, requiring fast re-optimization of routes as new orders arrive (dynamic VRP)
- Disruption re-routing finds alternative paths when links fail or delays occur, mapping to robust shortest path under uncertain arc costs
- Rolling-horizon re-planning balances solution quality against computation time, often using insertion heuristics warm-started from the current plan

## Data requirements
- Live vehicle positions, remaining capacity, and current route plans
- Incoming order stream with pickup/delivery locations and urgency levels
- Real-time traffic and road network conditions (travel time updates)
- Historical disruption data for scenario generation and robustness calibration

## Canonical problem
- [CVRP](../../../problems/2_routing/cvrp/README.md)
- [VRPTW](../../../problems/2_routing/vrptw/README.md)
- [Robust Shortest Path](../../../problems/9_uncertainty_modeling/robust_shortest_path/README.md)

## Status
This decision point is part of the **Transportation & Logistics** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

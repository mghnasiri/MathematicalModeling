# Growing Season Operations

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Agriculture
**Phase:** Growing Season
**Decision-maker:** Farm operations manager
**Decision frequency:** Daily to weekly during growing season

## Decision question
How should irrigation, fertilizer application, and pest control be scheduled and routed across fields?

## OR problem mapping
**Canonical problem(s):** LP/Network Flow (irrigation), VRP (fertilizer/pest routing)
**Implementation:** See:
- [`../../agriculture_irrigation_network.py`](../../agriculture_irrigation_network.py)
- [`../../agriculture_fertilizer_routing.py`](../../agriculture_fertilizer_routing.py)
- [`../../agriculture_pest_control.py`](../../agriculture_pest_control.py)
- [`../../agriculture_water_allocation.py`](../../agriculture_water_allocation.py)

## Key modeling aspects
- Irrigation canal allocation is a network flow problem maximizing water delivery across field nodes
- Fertilizer and pest-control vehicles visiting multiple fields map to capacitated VRP with time windows
- Water allocation among competing crops under scarcity is an LP with resource-sharing constraints

## Data requirements
- Irrigation network topology (canals, pumps, valves) with flow capacities
- Field locations, crop water/fertilizer requirements, and application time windows
- Vehicle fleet size, tank capacities, and travel time matrix between fields
- Water supply forecasts (reservoir levels, rainfall) and regulatory allocation limits

## Canonical problem
- [Max Flow](../../../problems/6_network_flow_design/max_flow/README.md) -- irrigation network capacity allocation
- [CVRP](../../../problems/2_routing/cvrp/README.md) -- fertilizer and pest-control vehicle routing
- [VRPTW](../../../problems/2_routing/vrptw/README.md) -- routing with application time windows
- [Linear Programming](../../../problems/continuous/linear_programming/README.md) -- water allocation among crops

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

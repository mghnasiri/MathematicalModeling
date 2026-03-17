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

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

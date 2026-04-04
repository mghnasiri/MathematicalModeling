# Harvest Scheduling

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Agriculture
**Phase:** Harvest Season
**Decision-maker:** Harvest operations manager
**Decision frequency:** Daily during harvest

## Decision question
In what sequence should fields be harvested, and how should equipment be allocated?

## OR problem mapping
**Canonical problem(s):** Machine Scheduling, Parallel Machine Scheduling
**Implementation:** See:
- [`../../agriculture_harvest_scheduling.py`](../../agriculture_harvest_scheduling.py)
- [`../../agriculture_equipment_scheduling.py`](../../agriculture_equipment_scheduling.py)

## Key modeling aspects
- Fields are jobs with crop-dependent processing times and maturity-window due dates, forming a scheduling problem
- Multiple harvesters operating simultaneously map to identical/uniform parallel machine scheduling
- Crop spoilage penalties for late harvest translate to weighted tardiness objectives

## Data requirements
- Field list with estimated harvest duration, crop maturity windows, and spoilage cost rates
- Equipment fleet: number of harvesters, speed ratings, and fuel/maintenance constraints
- Field-to-field travel times and any precedence constraints (e.g., access roads)
- Weather forecasts affecting available working hours per day

## Canonical problem
- [Single Machine Scheduling](../../../problems/1_scheduling/single_machine/README.md) -- sequencing fields on one harvester
- [Parallel Machine Scheduling](../../../problems/1_scheduling/parallel_machine/README.md) -- allocating fields across multiple harvesters
- [Flow Shop Scheduling](../../../problems/1_scheduling/flow_shop/README.md) -- multi-stage harvest-then-transport pipelines

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

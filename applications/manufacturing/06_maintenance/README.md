# Maintenance Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Manufacturing
**Phase:** Maintenance
**Decision-maker:** Maintenance manager
**Decision frequency:** Monthly

## Decision question
When should preventive maintenance be scheduled to minimize downtime?

## OR problem mapping
**Canonical problem(s):** Scheduling

## Key modeling aspects
- Preventive-maintenance scheduling trades off planned downtime costs against the risk and cost of unplanned breakdowns
- Grouping maintenance jobs into batches reduces setup overhead, mapping to batch-scheduling models on a single machine or parallel machines
- Age-based and condition-based policies determine when to trigger maintenance, linking to single-machine scheduling with release dates and deadlines

## Data requirements
- Equipment list with failure-rate distributions (MTBF/MTTR) or degradation models
- Maintenance task durations and required crew/resource availability
- Cost of planned maintenance vs. cost of unplanned failure (downtime, lost output)
- Production schedule constraints defining available maintenance windows

## Canonical problem
- [Single Machine Scheduling](../../../problems/1_scheduling/single_machine/README.md)
- [Batch Scheduling](../../../problems/1_scheduling/batch_scheduling/README.md)
- [Parallel Machine Scheduling](../../../problems/1_scheduling/parallel_machine/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

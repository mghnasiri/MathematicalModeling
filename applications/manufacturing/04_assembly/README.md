# Assembly Operations

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Manufacturing
**Phase:** Assembly
**Decision-maker:** Line manager
**Decision frequency:** Per shift

## Decision question
How should tasks be assigned to assembly stations?

## OR problem mapping
**Canonical problem(s):** SALBP

## Key modeling aspects
- Assigning tasks to workstations under precedence and cycle-time constraints is the Simple Assembly Line Balancing Problem (SALBP)
- SALBP-1 minimizes the number of stations for a given cycle time; SALBP-2 minimizes cycle time for a given number of stations
- Task durations and precedence graphs are the core inputs; extensions include U-shaped lines, mixed-model lines, and stochastic task times

## Data requirements
- Task list with deterministic (or stochastic) processing times
- Precedence graph defining required task orderings
- Target cycle time or number of available workstations
- Optional: equipment/zone restrictions per task, ergonomic constraints

## Canonical problem
- [Assembly Line Balancing (SALBP)](../../../problems/1_scheduling/assembly_line_balancing/README.md)
- [Parallel Machine Scheduling](../../../problems/1_scheduling/parallel_machine/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

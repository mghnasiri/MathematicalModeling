# Project Planning

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Construction
**Phase:** Project Planning
**Decision-maker:** Project manager
**Decision frequency:** Per project

## Decision question
How should project activities be scheduled under resource and precedence constraints?

## OR problem mapping
**Canonical problem(s):** RCPSP

## Key modeling aspects
- Construction activities form a precedence DAG (e.g., foundation before framing) with renewable resource constraints (crews, cranes).
- The objective is to minimize project makespan (Cmax) while respecting both precedence and per-period resource capacities.
- Schedule Generation Schemes (serial/parallel SGS) decode priority lists into feasible schedules, enabling metaheuristic search.

## Data requirements
- Activity list with durations and precedence relationships.
- Renewable resource types (e.g., crews, equipment) with per-period availability.
- Per-activity resource requirements for each resource type.

## Canonical problem
See [RCPSP](../../../problems/1_scheduling/rcpsp/README.md)

## Status
This decision point is part of the **Construction** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

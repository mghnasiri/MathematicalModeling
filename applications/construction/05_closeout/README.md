# Project Closeout

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Construction
**Phase:** Project Closeout
**Decision-maker:** Project manager
**Decision frequency:** End of project

## Decision question
How should remaining punch list items be sequenced to minimize project completion time?

## OR problem mapping
**Canonical problem(s):** Single Machine Scheduling

## Key modeling aspects
- Punch list items are independent tasks processed by a single inspection crew — a 1 | | gamma single machine scheduling problem.
- Minimizing total weighted tardiness (1 || sum wjTj) captures contractual penalties for late deficiency corrections.
- Due dates derive from contractual milestones; weights reflect severity (safety-critical vs. cosmetic).

## Data requirements
- Punch list items with estimated correction durations.
- Due dates or contractual deadlines per item.
- Priority weights reflecting penalty severity or client importance.

## Canonical problem
See [Single Machine Scheduling](../../../problems/1_scheduling/single_machine/README.md)

## Status
This decision point is part of the **Construction** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

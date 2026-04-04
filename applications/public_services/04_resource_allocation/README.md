# Resource Allocation

> **Status:** Placeholder — full documentation not yet available.

## Sector context
**Sector:** Public Services
**Phase:** Resource Allocation
**Decision-maker:** Budget officer
**Decision frequency:** Annual

## Decision question
How should public budgets be allocated across programs and projects?

## OR problem mapping
**Canonical problem(s):** MIP / Knapsack

## Key modeling aspects
- Selecting which programs to fund under a fixed budget, where each program has a cost and an estimated benefit, maps to the 0-1 Knapsack Problem
- When funding levels are continuous (partial allocations allowed), the problem becomes a Linear Program maximizing total benefit subject to budget and policy constraints
- Assigning staff members to departments or projects to minimize skill mismatch or maximize coverage is a classic Assignment Problem

## Data requirements
- Program/project list with estimated costs and benefit scores
- Total budget ceiling and any per-category spending floors or caps
- Staff roster with skill profiles, availability, and assignment preferences
- Performance metrics or historical outcomes for benefit estimation

## Canonical problem
- [Knapsack](../../../problems/3_packing_cutting/knapsack/README.md) -- select programs under a budget constraint to maximize benefit
- [Linear Programming](../../../problems/continuous/linear_programming/README.md) -- continuous budget allocation with linear objectives
- [Assignment](../../../problems/4_assignment_matching/assignment/README.md) -- optimally assign workforce to roles minimizing total cost

## Status
This decision point is part of the **Public Services** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

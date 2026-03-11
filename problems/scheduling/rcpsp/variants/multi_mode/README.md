# Multi-Mode RCPSP (MPS | prec | Cmax)

## Problem Definition

Extends RCPSP by allowing each activity to be executed in one of several modes. Each mode has a different duration and resource requirement. Choose modes and schedule activities to minimize project makespan while respecting precedence and resource constraints.

## Complexity

Strongly NP-hard.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Serial SGS (shortest mode) | Heuristic | Sprecher & Drexl (1998) |
| Simulated Annealing | Metaheuristic | Hartmann & Briskorn (2010) |

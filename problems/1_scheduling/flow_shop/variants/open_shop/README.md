# Open Shop Scheduling (Om || Cmax)

## Problem Definition

Each job must be processed on all machines, but the order of operations within each job is free (no fixed routing). The objective is to minimize the makespan.

**Objective:** min Cmax

**Key difference from flow shop:** No fixed machine order — each job can visit machines in any order.

## Complexity

NP-hard for m >= 3 (Gonzalez & Sahni, 1976). O(2)||Cmax is polynomial.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| LPT dispatching | Heuristic | Gonzalez & Sahni (1976) |
| Greedy earliest-start | Heuristic | Gonzalez & Sahni (1976) |
| Simulated Annealing | Metaheuristic | Liaw (2000) |

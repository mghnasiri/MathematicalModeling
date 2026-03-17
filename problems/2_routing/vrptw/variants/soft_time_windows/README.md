# VRPTW with Soft Time Windows

## Problem Definition

Extends VRPTW by allowing time window violations with a penalty cost proportional to the delay. Early arrivals still wait, but late arrivals incur a per-unit penalty rather than being infeasible. Minimizes travel distance plus total penalty.

## Complexity

NP-hard (generalizes VRPTW).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor | Heuristic | — |
| Simulated Annealing | Metaheuristic | Taillard et al. (1997) |

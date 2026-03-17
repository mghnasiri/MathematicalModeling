# Prize-Collecting TSP (PCTSP)

## Problem Definition

Given n cities with prizes and travel costs, select a subset of cities and find a tour through them. Minimize travel cost minus collected prizes, subject to collecting at least a minimum total prize.

## Complexity

NP-hard (generalizes TSP). 2-approximation known via primal-dual methods (Goemans & Williamson, 1995).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy Prize Insertion | Heuristic | — |
| Nearest Neighbor PCTSP | Heuristic | — |
| Simulated Annealing | Metaheuristic | Balas (1989) |

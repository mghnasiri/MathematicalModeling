# Quadratic Assignment Problem (QAP)

## Problem Definition

Given n facilities and n locations, a flow matrix F (flow between facilities) and a distance matrix D (distance between locations), assign each facility to a unique location to minimize total cost: Σ_i Σ_j f_ij × d_π(i),π(j).

## Complexity

NP-hard (one of the hardest combinatorial optimization problems). No constant-factor approximation is known unless P=NP.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy Construction | Heuristic | — |
| 2-opt Local Search | Heuristic | Burkard et al. (2009) |
| Simulated Annealing | Metaheuristic | Burkard & Rendl (1984) |

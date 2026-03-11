# Stochastic Flow Shop (Fm | prmu, stoch | E[Cmax])

## Problem Definition

Extends the permutation flow shop by modeling processing times as random variables. Processing times follow truncated normal distributions with known means and standard deviations. The objective is to minimize the expected makespan.

**Objective:** min E[Cmax] over all job permutations.

## Complexity

NP-hard (generalizes deterministic PFSP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| NEH (deterministic proxy) | Heuristic | Nawaz, Enscore & Ham (1983) |
| NEH (Monte Carlo) | Heuristic | Framinan & Perez-Gonzalez (2015) |
| Simulated Annealing | Metaheuristic | Gourgand, Grangeon & Norre (2000) |

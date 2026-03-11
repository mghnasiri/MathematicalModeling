# Hybrid Flow Shop (HFm | prmu | Cmax)

## Problem Definition

Extends the permutation flow shop by having multiple identical parallel machines at each stage. Each job visits all stages in sequence, but at each stage can be processed on any available machine. Minimize makespan.

## Complexity

NP-hard even for 2 stages with 1 and 2 machines respectively.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| NEH-HFS | Heuristic | Adapted from Nawaz et al. (1983) |
| LPT-HFS | Heuristic | — |
| SPT-HFS | Heuristic | — |
| Simulated Annealing | Metaheuristic | Ruiz & Vázquez-Rodríguez (2010) |

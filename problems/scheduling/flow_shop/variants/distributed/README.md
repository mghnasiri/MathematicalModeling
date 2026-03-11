# Distributed Permutation Flow Shop (DFm | prmu | Cmax)

## Problem Definition

Multiple identical factories, each with the same flow shop configuration (m machines). Jobs must be assigned to factories and sequenced within each. Minimize the overall makespan (maximum completion across all factories).

## Complexity

NP-hard (generalizes PFSP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| NEH-DPFSP | Heuristic | Naderi & Ruiz (2010) |
| Round-Robin | Heuristic | — |
| Simulated Annealing | Metaheuristic | Naderi & Ruiz (2010) |

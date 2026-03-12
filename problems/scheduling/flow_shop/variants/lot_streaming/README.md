# Lot Streaming Flow Shop (Fm | prmu, lot-streaming | Cmax)

## Problem Definition

Each job is split into equal sublots that transfer between machines independently, allowing overlapping operations of the same job on consecutive machines. Reduces makespan compared to standard flow shop.

## Complexity

NP-hard for m >= 3 with discrete sublots.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| NEH-LS | Heuristic | Trietsch & Baker (1993) |
| LPT-LS | Heuristic | — |
| Simulated Annealing | Metaheuristic | Potts & Baker (2000) |

# Flow Shop with Total Weighted Tardiness (Fm | prmu | ΣwjTj)

## Problem Definition

Standard permutation flow shop with due dates and job weights. Minimize total weighted tardiness: ΣwjTj where Tj = max(0, Cj - dj).

## Complexity

NP-hard (even for single machine).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| EDD Rule | Heuristic | Jackson (1955) |
| WSPT Rule | Heuristic | Smith (1956) |
| NEH-Tardiness | Heuristic | Kim (1993) |
| Simulated Annealing | Metaheuristic | Vallada & Ruiz (2010) |

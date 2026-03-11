# Asymmetric Traveling Salesman Problem (ATSP)

## Problem Definition

Given n cities with directed distances d(i,j) ≠ d(j,i), find the shortest directed Hamiltonian cycle. Unlike symmetric TSP, reversal of segments changes arc directions.

## Complexity

NP-hard. Best known approximation: O(log n / log log n) (Asadpour et al., 2017).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor | Heuristic | — |
| Multi-Start NN | Heuristic | — |
| Simulated Annealing | Metaheuristic | Kanellakis & Papadimitriou (1980) |

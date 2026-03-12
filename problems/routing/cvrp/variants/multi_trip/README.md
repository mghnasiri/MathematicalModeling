# Multi-Trip VRP (MTVRP)

## Problem Definition

Extends CVRP by allowing vehicles to make multiple trips. After completing a route, a vehicle returns to the depot and can start a new trip. Useful when the number of vehicles is limited relative to the number of customers.

**Objective:** Minimize total travel distance.

**Constraints:**
- Each customer visited exactly once
- Route capacity respected per trip
- Number of vehicles limited

## Complexity

NP-hard (generalizes CVRP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy NN + round-robin | Heuristic | Taillard, Laporte & Gendreau (1996) |
| Simulated Annealing | Metaheuristic | Olivera & Viera (2007) |

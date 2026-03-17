# Periodic Vehicle Routing Problem (PVRP)

## Problem Definition

Over a planning horizon of T days, each customer requires a specified number of visits. Select which days to visit each customer and build capacity-feasible routes for each day, minimizing total distance.

## Complexity

NP-hard (generalizes CVRP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Spread-then-Route | Heuristic | Christofides & Beasley (1984) |
| Simulated Annealing | Metaheuristic | Cordeau et al. (1997) |

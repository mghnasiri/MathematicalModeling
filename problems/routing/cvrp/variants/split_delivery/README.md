# Split Delivery Vehicle Routing Problem (SDVRP)

## Problem Definition

Extends CVRP by allowing each customer to be served by multiple vehicles (split deliveries). A customer's demand can be split across different routes, potentially reducing total distance and number of vehicles.

## Complexity

NP-hard (generalizes CVRP). Dror & Trudeau (1989) showed savings of up to 50% compared to non-split CVRP.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor Split | Heuristic | — |
| Savings Split | Heuristic | Adapted from Clarke-Wright |
| Simulated Annealing | Metaheuristic | Archetti et al. (2006) |

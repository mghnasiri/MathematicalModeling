# Open Vehicle Routing Problem (OVRP)

## Problem Definition

Like CVRP but vehicles do not return to the depot. Each route starts at the depot and ends at the last customer served.

```
min  Σ_k route_distance(k)
s.t. demand(route_k) ≤ Q        ∀ k  (capacity)
     each customer visited once
     routes start at depot, end at last customer
```

## Complexity

NP-hard (generalizes TSP).

## Applications

- **Courier delivery**: drivers go home after last delivery
- **School bus routing**: buses end at the school
- **Home healthcare**: nurses visit patients then go home
- **Newspaper delivery**: carriers end their route at home

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor OVRP | Heuristic | Sariklis & Powell (2000) |
| Modified Savings | Heuristic | Sariklis & Powell (2000) |
| Simulated Annealing | Metaheuristic | Li et al. (2007) |

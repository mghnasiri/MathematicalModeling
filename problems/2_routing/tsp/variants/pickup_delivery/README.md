# Pickup and Delivery Problem (PDP)

## Problem Definition

Given n pickup-delivery pairs and a depot, find the shortest tour visiting all 2n+1 locations exactly once, such that each pickup is visited before its corresponding delivery.

```
min  Σ d(π(k), π(k+1))
s.t. pos(pickup_i) < pos(delivery_i)   ∀ i  (precedence)
     π is a Hamiltonian cycle starting at depot
```

## Complexity

NP-hard (generalizes TSP).

## Applications

- **Courier services**: pickup packages and deliver to destinations
- **Ride-sharing**: pick up and drop off passengers
- **Supply chain**: collect from suppliers, deliver to warehouses
- **Dial-a-ride**: transport requests with origin-destination pairs

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Feasible Neighbor | Heuristic | Savelsbergh & Sol (1995) |
| Cheapest Pair Insertion | Heuristic | Savelsbergh & Sol (1995) |
| Simulated Annealing | Metaheuristic | Kirkpatrick et al. (1983) |

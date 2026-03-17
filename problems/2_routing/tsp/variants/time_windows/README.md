# TSP with Time Windows (TSPTW)

## Problem Definition

Find the shortest Hamiltonian cycle visiting each of **n** cities exactly once, where each city i must be visited within its time window **[e_i, l_i]**. Early arrivals wait; late arrivals are infeasible.

```
min  Σ d(π(k), π(k+1))
s.t. e_i ≤ t_i ≤ l_i          ∀ i (time windows)
     t_j ≥ t_i + s_i + d(i,j)  (travel + service time)
     π is a Hamiltonian cycle
```

## Complexity

NP-hard (generalizes TSP).

## Applications

- **Delivery scheduling**: packages with customer-specified time slots
- **Technician routing**: service calls with appointment windows
- **Dial-a-ride**: passenger pickup/dropoff with time constraints
- **Aircraft landing**: runway scheduling with approach windows
- **Exam scheduling**: students must visit exam rooms within allocated periods

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Feasible Neighbor | Heuristic | Solomon (1987) |
| Earliest Deadline Insertion | Heuristic | Gendreau et al. (1998) |
| Simulated Annealing | Metaheuristic | Ohlmann & Thomas (2007) |

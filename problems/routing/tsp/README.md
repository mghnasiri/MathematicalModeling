# Traveling Salesman Problem (TSP)

## Problem Definition

Given a set of $n$ cities and pairwise distances $d_{ij}$, find the shortest **Hamiltonian cycle** (tour) visiting each city exactly once and returning to the starting city.

**Symmetric TSP**: $d_{ij} = d_{ji}$ for all $i, j$ (undirected graph).
**Asymmetric TSP (ATSP)**: $d_{ij} \neq d_{ji}$ in general (directed graph).

## Mathematical Formulation

**Parameters:**
- $n$: number of cities
- $d_{ij}$: distance from city $i$ to city $j$

**Decision variables:**
- $x_{ij} \in \{0, 1\}$: 1 if edge $(i,j)$ is in the tour

**Objective:**

$$\min \sum_{i=1}^{n} \sum_{j=1}^{n} d_{ij} x_{ij}$$

**Subject to:**

$$\sum_{j=1}^{n} x_{ij} = 1 \quad \forall i \quad \text{(leave each city once)}$$

$$\sum_{i=1}^{n} x_{ij} = 1 \quad \forall j \quad \text{(enter each city once)}$$

$$\text{Subtour elimination constraints (SECs)}$$

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| General TSP | NP-hard | Karp (1972) |
| Metric TSP | NP-hard, 3/2-approximable | Christofides (1976) |
| Euclidean TSP | NP-hard, PTAS exists | Arora (1998) |
| 2-city TSP | O(1) | Trivial |

## Solution Approaches

### Exact Methods

| Method | Complexity | Practical Limit | Description |
|--------|-----------|----------------|-------------|
| Held-Karp DP | $O(2^n \cdot n^2)$ | $n \leq 23$ | Bitmask DP over subsets |
| Branch & Bound | $O(n!)$ worst | $n \leq 25$ | DFS with 1-tree lower bound |

### Constructive Heuristics

| Method | Complexity | Approx. Ratio | Description |
|--------|-----------|---------------|-------------|
| Nearest Neighbor | $O(n^2)$ | $O(\log n)$ | Greedily visit nearest city |
| Cheapest Insertion | $O(n^3)$ | 2 (metric) | Insert cheapest city-position pair |
| Farthest Insertion | $O(n^3)$ | $O(\log n)$ | Insert farthest city at cheapest position |
| Nearest Insertion | $O(n^3)$ | 2 (metric) | Insert nearest city at cheapest position |
| Greedy (Nearest Edge) | $O(n^2 \log n)$ | $O(\log n)$ | Add shortest feasible edges |

### Improvement Heuristics & Metaheuristics

| Method | Neighborhood | Description |
|--------|-------------|-------------|
| 2-opt | Segment reversal | Remove 2 edges, reconnect by reversing |
| Or-opt | Segment relocation | Move 1-3 consecutive cities |
| VND | 2-opt + Or-opt | Variable Neighborhood Descent |
| Simulated Annealing | 2-opt | Boltzmann acceptance, auto-calibrated temperature |
| Genetic Algorithm | OX crossover | Order crossover, swap mutation, tournament selection |

## Implementations in This Repository

```
tsp/
├── instance.py              # TSPInstance, TSPSolution, benchmark instances
├── exact/
│   ├── held_karp.py         # Held-Karp DP — O(2^n * n^2)
│   └── branch_and_bound.py  # B&B with 1-tree lower bound
├── heuristics/
│   ├── nearest_neighbor.py  # NN + multi-start
│   ├── cheapest_insertion.py # Cheapest/farthest/nearest insertion
│   └── greedy.py            # Greedy nearest-edge
├── metaheuristics/
│   ├── local_search.py      # 2-opt, Or-opt, VND
│   ├── simulated_annealing.py
│   └── genetic_algorithm.py
└── tests/
    └── test_tsp.py          # 55 tests, 10 test classes
```

## Benchmark Instances

| Instance | Cities | Optimal | Source |
|----------|--------|---------|--------|
| small4 | 4 | 9 | Handcrafted |
| small5 | 5 | 19 | Handcrafted |
| gr17 | 17 | 2016 | TSPLIB (Groetschel) |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Asymmetric TSP (ATSP)](variants/asymmetric/) | `variants/asymmetric/` | Directed distances, $d(i,j) \neq d(j,i)$ |
| [TSP with Time Windows (TSPTW)](variants/time_windows/) | `variants/time_windows/` | Each city has a service time window $[e_i, l_i]$ |
| [Prize-Collecting TSP (PCTSP)](variants/prize_collecting/) | `variants/prize_collecting/` | Collect prizes from a subset of cities; trade off travel cost vs. prize |
| [Pickup and Delivery (PDP)](variants/pickup_delivery/) | `variants/pickup_delivery/` | Paired pickup-delivery requests with precedence constraints |

## Key References

- Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, 85-103. https://doi.org/10.1007/978-1-4684-2001-2_9
- Held, M. & Karp, R.M. (1962). A dynamic programming approach to sequencing problems. *SIAM J.*, 10(1), 196-210. https://doi.org/10.1137/0110015
- Croes, G.A. (1958). A method for solving traveling salesman problems. *Oper. Res.*, 6(6), 791-812. https://doi.org/10.1287/opre.6.6.791
- Rosenkrantz, D.J. et al. (1977). An analysis of several heuristics for the TSP. *SIAM J. Comput.*, 6(3), 563-581. https://doi.org/10.1137/0206041
- Applegate, D.L. et al. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton. https://doi.org/10.1515/9781400841103

# Traveling Salesman Problem (TSP)

## 1. Problem Definition

- **Input:**
  - A set $V = \{1, 2, \ldots, n\}$ of cities (vertices)
  - Pairwise distances $d_{ij} \geq 0$ for each pair $(i, j) \in V \times V$
- **Decision:** Find a permutation $\pi = (\pi(1), \pi(2), \ldots, \pi(n))$ of cities
- **Objective:** Minimize total tour length $\sum_{k=1}^{n-1} d_{\pi(k),\pi(k+1)} + d_{\pi(n),\pi(1)}$
- **Constraints:** Visit each city exactly once and return to the starting city (Hamiltonian cycle)
- **Classification:** Combinatorial optimization (discrete permutation)

**Symmetric TSP:** $d_{ij} = d_{ji}$ for all $i, j$ (undirected complete graph $K_n$).
**Asymmetric TSP (ATSP):** $d_{ij} \neq d_{ji}$ in general (directed complete graph).
**Metric TSP:** Distances satisfy the triangle inequality $d_{ij} \leq d_{ik} + d_{kj}$.
**Euclidean TSP:** Cities are points in $\mathbb{R}^2$; distances are Euclidean.

### Complexity

| Variant | Complexity | Approximation | Reference |
|---------|-----------|---------------|-----------|
| General TSP | NP-hard | No constant-factor unless P=NP | Karp (1972) |
| Metric TSP | NP-hard | 3/2-approximation | Christofides (1976) |
| Euclidean TSP | NP-hard | PTAS exists | Arora (1998) |
| $(1,2)$-TSP | NP-hard | 8/7-approximation | Berman & Karpinski (2006) |

The decision version ("Is there a tour of length $\leq L$?") is NP-complete by reduction from Hamiltonian Cycle. The optimization version is NP-hard. No constant-factor approximation exists for the general (non-metric) case unless P = NP (Sahni & Gonzalez, 1976).

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of cities | $\mathbb{Z}^+$ |
| $d_{ij}$ | Distance from city $i$ to city $j$ | $\mathbb{R}_{\geq 0}$ |
| $x_{ij}$ | 1 if the tour traverses edge $(i,j)$ | $\{0, 1\}$ |
| $u_i$ | Subtour elimination variable (MTZ) or flow | $\mathbb{R}$ |
| $S$ | Subset of cities | $S \subset V,\; 2 \leq |S| \leq n{-}1$ |

### Formulation A: Dantzig-Fulkerson-Johnson (DFJ) — Subtour Elimination

$$\min \sum_{i=1}^{n} \sum_{j=1, j \neq i}^{n} d_{ij}\, x_{ij} \tag{1}$$

$$\text{s.t.} \quad \sum_{j \neq i} x_{ij} = 1 \quad \forall\, i \in V \quad \text{(leave each city once)} \tag{2}$$

$$\sum_{i \neq j} x_{ij} = 1 \quad \forall\, j \in V \quad \text{(enter each city once)} \tag{3}$$

$$\sum_{i \in S} \sum_{j \in S, j \neq i} x_{ij} \leq |S| - 1 \quad \forall\, S \subset V,\; 2 \leq |S| \leq n-1 \quad \text{(SECs)} \tag{4}$$

$$x_{ij} \in \{0, 1\} \tag{5}$$

**Strengths:** Tightest LP relaxation of any standard TSP formulation.
**Weaknesses:** Exponentially many constraints ($2^n - 2$ subtour elimination constraints). In practice, these are added as lazy constraints via callback — only violated SECs are generated.

### Formulation B: Miller-Tucker-Zemlin (MTZ) — Compact

Replace the exponential SECs with $O(n^2)$ constraints using auxiliary variables $u_i$:

$$u_i - u_j + n\, x_{ij} \leq n - 1 \quad \forall\, i, j \in V \setminus \{1\},\; i \neq j \tag{6}$$

$$1 \leq u_i \leq n - 1 \quad \forall\, i \in V \setminus \{1\} \tag{7}$$

**Strengths:** Polynomial number of constraints. Easy to implement.
**Weaknesses:** Much weaker LP relaxation than DFJ. Solver typically needs longer to close the gap.

### Formulation C: 1-Tree Relaxation (Lower Bound)

The minimum 1-tree (Held & Karp, 1970): compute the MST of $V \setminus \{1\}$ plus the two shortest edges incident to city 1. This yields a lower bound on the optimal tour. Lagrangian relaxation of the degree constraints can be applied via subgradient optimization to tighten this bound. This is the standard B&B lower bound for TSP.

---

## 3. Variants

| Notation | Variant | Directory | Description |
|----------|---------|-----------|-------------|
| ATSP | Asymmetric TSP | `variants/asymmetric/` | Directed distances |
| TSPTW | TSP with Time Windows | `variants/time_windows/` | Service time window $[e_i, l_i]$ per city |
| PCTSP | Prize-Collecting TSP | `variants/prize_collecting/` | Trade off travel cost vs. collected prizes |
| PDP | Pickup and Delivery | `variants/pickup_delivery/` | Paired requests with precedence |

### 3.1 Asymmetric TSP (ATSP)

Directed graph with $d_{ij} \neq d_{ji}$. Can be transformed to symmetric TSP of size $2n$ via the Jonker-Volgenant transformation, but specialized algorithms (assignment relaxation + patching) are more efficient.

### 3.2 TSP with Time Windows (TSPTW)

Each city $i$ has a time window $[e_i, l_i]$. The salesman must arrive at city $i$ by time $l_i$; if arriving before $e_i$, they wait. Feasibility itself is NP-complete.

### 3.3 Prize-Collecting TSP (PCTSP)

Not all cities need to be visited. Each city $i$ has a prize $\pi_i$; the objective combines minimizing travel cost and maximizing collected prizes. Models selective visiting decisions.

### 3.4 Pickup and Delivery Problem (PDP)

Paired requests $(p_i, d_i)$ where item $i$ must be picked up at $p_i$ and delivered to $d_i$. Precedence constraint: $p_i$ before $d_i$ in the tour.

---

## 4. Benchmark Instances

### TSPLIB

The standard TSP benchmark library (Reinelt, 1991) contains 111 instances from 14 to 85,900 cities. All optimal solutions are known for instances up to ~10,000 cities.

**URL:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

### Instances in This Repository

| Instance | Cities | Optimal | Source |
|----------|--------|---------|--------|
| small4 | 4 | 9 | Handcrafted |
| small5 | 5 | 19 | Handcrafted |
| gr17 | 17 | 2016 | TSPLIB (Groetschel) |

### Small Illustrative Instance

A 5-city Euclidean instance:

```
Cities: (0,0), (1,3), (4,3), (4,0), (2,1)
Optimal tour: 0 → 1 → 2 → 3 → 4 → 0
Distance: 13.6 (Euclidean)
```

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Held-Karp Dynamic Programming (Held & Karp, 1962)

**Idea:** Define $f(S, j)$ = minimum cost path visiting all cities in subset $S$, ending at $j$. Recurrence: $f(S, j) = \min_{k \in S \setminus \{j\}} [f(S \setminus \{j\}, k) + d_{kj}]$. Enumerate over all $2^n$ subsets.

**Complexity:** $O(2^n \cdot n^2)$ time, $O(2^n \cdot n)$ space. **Practical limit:** $n \leq 23$.

```
ALGORITHM HeldKarp(d[1..n][1..n])
  FOR each subset S ⊆ {2,...,n} with |S|=1:
    f[S][j] ← d[1][j]  for j ∈ S
  FOR s = 2 TO n-1:
    FOR each S ⊆ {2,...,n} with |S|=s:
      FOR j ∈ S:
        f[S][j] ← min over k ∈ S\{j} of (f[S\{j}][k] + d[k][j])
  RETURN min over j of (f[{2,...,n}][j] + d[j][1])
```

#### Branch and Bound (1-Tree Lower Bound)

**Idea:** DFS search tree fixing edges in/out of the tour. Lower bound at each node via minimum 1-tree. Warm-start upper bound from nearest-neighbor heuristic.

**Practical limit:** $n \leq 25$ (highly instance-dependent).

### 5.2 Constructive Heuristics

| # | Method | Complexity | Approx. Ratio | Key Idea |
|---|--------|-----------|---------------|----------|
| 1 | Nearest Neighbor | $O(n^2)$ | $O(\log n)$ | Greedily visit nearest unvisited city |
| 2 | Cheapest Insertion | $O(n^3)$ | 2 (metric) | Insert city-position pair of minimum cost |
| 3 | Farthest Insertion | $O(n^3)$ | $O(\log n)$ | Insert farthest city at cheapest position |
| 4 | Nearest Insertion | $O(n^3)$ | 2 (metric) | Insert nearest city at cheapest position |
| 5 | Greedy (Nearest Edge) | $O(n^2 \log n)$ | $O(\log n)$ | Add shortest edges not violating degree/subtour |

```
ALGORITHM NearestNeighbor(d[1..n][1..n])
  tour ← [1],  visited ← {1}
  FOR step = 2 TO n:
    last ← tour[-1]
    next ← argmin over j ∉ visited of d[last][j]
    Append next to tour, add to visited
  RETURN tour
```

**Multi-start NN:** Run NN from each starting city, return the best tour. Still $O(n^3)$.

### 5.3 Improvement Heuristics / Local Search

| Neighborhood | Move | Size | Description |
|-------------|------|------|-------------|
| 2-opt | Reverse segment | $O(n^2)$ | Remove edges $(i, i{+}1)$ and $(j, j{+}1)$, reconnect by reversing segment $[i{+}1, j]$ |
| Or-opt | Relocate segment | $O(n^2)$ | Move 1, 2, or 3 consecutive cities to another position |
| 3-opt | Recombine 3 segments | $O(n^3)$ | Remove 3 edges, reconnect (8 possible reconnections) |
| VND | 2-opt + Or-opt | Variable | Apply neighborhoods in sequence; restart on improvement |

**2-opt** is the workhorse of TSP local search. A single 2-opt pass is $O(n^2)$; in practice, tours converge in $O(n)$ iterations of the full neighborhood scan.

```
ALGORITHM TwoOpt(tour)
  improved ← TRUE
  WHILE improved:
    improved ← FALSE
    FOR i = 1 TO n-2:
      FOR j = i+2 TO n:
        Δ ← d[tour[i]][tour[j]] + d[tour[i+1]][tour[j+1]]
             - d[tour[i]][tour[i+1]] - d[tour[j]][tour[j+1]]
        IF Δ < 0:
          Reverse tour[i+1 .. j]
          improved ← TRUE
  RETURN tour
```

### 5.4 Metaheuristics

This repository implements **7 metaheuristics** for TSP:

| # | Method | Year | Category | Key Feature |
|---|--------|------|----------|-------------|
| 1 | Simulated Annealing (SA) | — | Trajectory | 2-opt moves, auto-calibrated temperature |
| 2 | Tabu Search (TS) | — | Trajectory | 2-opt with recency-based tabu list |
| 3 | Iterated Greedy (IG) | — | Trajectory | Remove + reinsert $d$ cities via cheapest insertion |
| 4 | Variable Neighborhood Search (VNS) | — | Trajectory | Systematic neighborhood change (2-opt → Or-opt → 3-opt) |
| 5 | Genetic Algorithm (GA) | — | Population | Order Crossover (OX), swap mutation |
| 6 | Ant Colony Optimization (ACO) | — | Population | Pheromone trails + visibility heuristic |
| 7 | Local Search (2-opt, Or-opt, VND) | — | Improvement | Foundation for all trajectory methods |

**ACO** is particularly natural for TSP: ants probabilistically build tours edge-by-edge, biased by pheromone intensity (exploitation) and inverse distance (greedy heuristic). MMAS bounds prevent stagnation.

### 5.5 Hybrid and Advanced Methods

- **LKH (Lin-Kernighan-Helsgott):** State-of-the-art TSP solver. Sequential $k$-opt with candidate lists and backtracking. Not implemented here but referenced as the gold standard.
- **Concorde:** Exact solver using branch-and-cut with DFJ SECs. Solves instances with ~85,000 cities optimally.

---

## 6. Implementation Guide

### Modeling Tips

- **Distance matrix:** Precompute the full $n \times n$ distance matrix. For Euclidean instances, round to integers (as TSPLIB does) to avoid floating-point issues.
- **2-opt evaluation:** The cost change $\Delta$ for a 2-opt move can be computed in $O(1)$ from four edge costs. Never recompute the full tour cost.
- **Neighbor lists:** For large instances, maintain sorted neighbor lists (nearest $k$ cities per city). Only consider 2-opt moves involving at least one short edge — this prunes the $O(n^2)$ neighborhood without sacrificing solution quality.

### Common Pitfalls

- **Asymmetric distances:** Euclidean TSP is symmetric, but real-world road distances are often asymmetric (one-way streets). Ensure algorithms don't assume $d_{ij} = d_{ji}$.
- **Starting city:** For symmetric TSP, the starting city doesn't matter (any rotation of the tour is equivalent). For asymmetric TSP, it does.
- **Tour representation:** This repo uses ordered lists $[0, 3, 1, 2]$ meaning visit city 0, then 3, then 1, then 2, then back to 0.

---

## 7. Computational Results Summary

| Method | Category | Typical Gap (small, $n{\leq}50$) | Typical Gap (large, $n{>}100$) |
|--------|----------|------|------|
| Held-Karp | Exact | 0% ($n \leq 23$) | Infeasible |
| B&B (1-tree) | Exact | 0% ($n \leq 25$) | Infeasible |
| Nearest Neighbor | Heuristic | 15-25% | 15-25% |
| Cheapest Insertion | Heuristic | 10-20% | 10-20% |
| NN + 2-opt | Heuristic+LS | 2-5% | 3-8% |
| SA | Metaheuristic | <1% | 1-3% |
| GA + 2-opt | Metaheuristic | <1% | 1-3% |
| ACO | Metaheuristic | <1% | 1-3% |
| LKH (reference) | State-of-art | 0% | <0.1% |

**Scale guidance:**
- $n \leq 23$: Held-Karp gives the optimum.
- $n \leq 50$: B&B feasible; NN + 2-opt gets within 2-5%.
- $n = 100{-}1000$: Metaheuristics (SA, GA, ACO) with 2-opt local search. Expect 1-3% from optimal.
- $n > 1000$: LKH is the reference. Simple metaheuristics with 2-opt still perform well.

---

## 8. Implementations in This Repository

```
tsp/
├── instance.py                    # TSPInstance, TSPSolution, benchmark instances
│
├── exact/
│   ├── held_karp.py               # Held-Karp DP — O(2^n × n^2)
│   └── branch_and_bound.py        # B&B with 1-tree lower bound, NN warm-start
│
├── heuristics/
│   ├── nearest_neighbor.py        # NN + multi-start
│   ├── cheapest_insertion.py      # Cheapest / farthest / nearest insertion
│   └── greedy.py                  # Greedy nearest-edge
│
├── metaheuristics/
│   ├── local_search.py            # 2-opt, Or-opt, VND
│   ├── simulated_annealing.py     # SA with 2-opt moves
│   ├── tabu_search.py             # TS with recency-based tabu list
│   ├── iterated_greedy.py         # IG with remove/reinsert
│   ├── vns.py                     # VNS with systematic neighborhood change
│   ├── genetic_algorithm.py       # GA: OX crossover, swap mutation
│   └── ant_colony.py              # ACO/MMAS: pheromone trails
│
├── variants/
│   ├── asymmetric/                # ATSP
│   ├── time_windows/              # TSPTW
│   ├── prize_collecting/          # PCTSP
│   └── pickup_delivery/           # PDP
│
└── tests/                         # 5 test files
    ├── test_tsp.py                # Core algorithms
    ├── test_tsp_ts.py             # Tabu Search
    ├── test_tsp_aco.py            # Ant Colony
    ├── test_tsp_ig.py             # Iterated Greedy
    └── test_tsp_vns.py            # VNS
```

**Total:** 2 exact methods, 5 constructive heuristics (in 3 files), 7 metaheuristics/LS, 4 variants, 5 test files.

---

## 9. Key References

### Seminal Papers

- Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, 85-103.
- Held, M. & Karp, R.M. (1962). A dynamic programming approach to sequencing problems. *SIAM Journal on Applied Mathematics*, 10(1), 196-210.
- Christofides, N. (1976). Worst-case analysis of a new heuristic for the travelling salesman problem. *Report 388, Graduate School of Industrial Administration, CMU*.
- Croes, G.A. (1958). A method for solving traveling salesman problems. *Operations Research*, 6(6), 791-812.
- Lin, S. & Kernighan, B.W. (1973). An effective heuristic algorithm for the traveling-salesman problem. *Operations Research*, 21(2), 498-516.
- Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *Journal of the ACM*, 45(5), 753-782.

### Books

- Applegate, D.L., Bixby, R.E., Chvatal, V. & Cook, W.J. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.
- Gutin, G. & Punnen, A.P., eds. (2007). *The Traveling Salesman Problem and Its Variations*. Springer.

### Surveys

- Rosenkrantz, D.J., Stearns, R.E. & Lewis, P.M. (1977). An analysis of several heuristics for the traveling salesman problem. *SIAM Journal on Computing*, 6(3), 563-581.
- Laporte, G. (1992). The traveling salesman problem: An overview of exact and approximate algorithms. *European Journal of Operational Research*, 59(2), 231-247.

### Benchmark

- Reinelt, G. (1991). TSPLIB — a traveling salesman problem library. *ORSA Journal on Computing*, 3(4), 376-384.

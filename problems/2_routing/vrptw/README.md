# Vehicle Routing Problem with Time Windows (VRPTW)

## 1. Problem Definition

- **Input:**
  - A depot (node 0) with time window $[e_0, l_0]$ (planning horizon)
  - $n$ customers with demands $q_i$, time windows $[e_i, l_i]$, and service times $s_i$
  - Vehicle capacity $Q$, fleet size $K$
  - Pairwise distances $d_{ij}$ and travel times $t_{ij}$
- **Decision:** Partition customers into routes; determine visit sequences
- **Objective:** Minimize total travel distance (or number of vehicles + distance)
- **Constraints:**
  - Each customer visited exactly once
  - Route capacity $\leq Q$
  - Arrive at customer $i$ by $l_i$; if before $e_i$, wait until $e_i$
  - Return to depot within $[e_0, l_0]$
- **Classification:** NP-hard (generalizes CVRP)

### Complexity

NP-hard (generalizes CVRP which generalizes TSP). Even feasibility testing (does a feasible solution exist with $K$ vehicles?) is NP-complete when time windows are tight.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $[e_i, l_i]$ | Time window for customer $i$ | $\mathbb{R}^2$ |
| $s_i$ | Service time at customer $i$ | $\mathbb{R}_{\geq 0}$ |
| $t_{ij}$ | Travel time from $i$ to $j$ | $\mathbb{R}_{\geq 0}$ |
| $a_i$ | Arrival time at customer $i$ | $\mathbb{R}_{\geq 0}$ |
| $b_i$ | Begin-service time: $b_i = \max(a_i, e_i)$ | $\mathbb{R}_{\geq 0}$ |

### Time Window Propagation

For consecutive customers $i \to j$ in a route:

$$a_j = b_i + s_i + t_{ij} \tag{1}$$

$$b_j = \max(a_j, e_j) \tag{2}$$

Feasibility check: $b_j \leq l_j$ (arrive before window closes).

### MILP (extends CVRP with time variables)

Add time window constraints to the CVRP formulation:

$$b_i + s_i + t_{ij} - b_j \leq M(1 - x_{ij}) \quad \forall i,j \quad \text{(time propagation)} \tag{3}$$

$$e_i \leq b_i \leq l_i \quad \forall i \quad \text{(time window feasibility)} \tag{4}$$

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Soft Time Windows | `variants/soft_time_windows/` | Penalties for early/late arrival instead of hard constraints |

### Soft Time Windows

Arrival outside $[e_i, l_i]$ incurs a penalty rather than infeasibility. Objective includes penalty costs, enabling trade-off between routing cost and punctuality.

---

## 4. Benchmark Instances

### Solomon Benchmark (Solomon, 1987)

The standard VRPTW test set with 56 instances across 3 classes:

| Class | Customer Distribution | Time Windows |
|-------|----------------------|-------------|
| C (clustered) | Clustered groups | Narrow |
| R (random) | Uniform random | Wide |
| RC (mixed) | Random + clustered | Mixed |

Each class has 100-customer and 25-customer variants.

**URL:** https://www.sintef.no/projectweb/top/vrptw/

### Instances in This Repository

| Instance | Customers | Type |
|----------|-----------|------|
| solomon_c101_mini | 8 | Clustered (from C101) |
| tight_tw5 | 5 | Narrow windows |

---

## 5. Solution Methods

### 5.1 Constructive Heuristics

#### Solomon I1 Insertion (Solomon, 1987)

**Idea:** Start with a seed customer (farthest unrouted). Iteratively insert the customer-position pair that minimizes a composite criterion: $\alpha_1 \cdot \text{distance increase} + \alpha_2 \cdot \text{urgency}$. When no more insertions are feasible, start a new route.

**Complexity:** $O(n^2 K)$ where $K$ is the number of routes.

#### Nearest Neighbor with Time Windows

Greedy: from the current position, go to the nearest *feasible* unvisited customer (respecting capacity and time windows).

### 5.2 Metaheuristics

This repository implements **7 metaheuristics** for VRPTW:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Relocate/swap with TW feasibility |
| 2 | Simulated Annealing (SA) | Trajectory | Relocate/swap with TW checks |
| 3 | Tabu Search (TS) | Trajectory | Customer-route tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove customers + Solomon reinsert |
| 5 | Variable Neighborhood Search (VNS) | Trajectory | Relocate → swap → cross-exchange |
| 6 | Genetic Algorithm (GA) | Population | Giant-tour + TW-aware split decoder |
| 7 | Ant Colony Optimization (ACO) | Population | Pheromone + TW urgency heuristic |

**TW-feasibility check:** All inter-route moves must verify that the new route respects time windows for every customer downstream of the insertion point. This requires $O(k)$ propagation per move (where $k$ is route length).

### 5.3 State-of-the-Art

**Adaptive Large Neighborhood Search** (Ropke & Pisinger, 2006) and **Hybrid Genetic Search** (Vidal et al., 2013) achieve state-of-the-art on Solomon and Gehring-Homberger benchmarks.

---

## 6. Implementation Guide

- **Forward time slack:** Precompute slack at each position to check TW feasibility in $O(1)$ for relocate/swap moves instead of $O(k)$.
- **Hierarchical objective:** Minimize vehicles first, then distance. This avoids solutions with many half-empty routes.
- **Waiting time:** When a vehicle arrives early, it waits. This increases route duration but not cost (distance-based). Some formulations penalize waiting.

---

## 7. Implementations in This Repository

```
vrptw/
├── instance.py                    # VRPTWInstance, VRPTWSolution, validation
├── heuristics/
│   └── solomon_insertion.py       # Solomon I1 + nearest neighbor TW
├── metaheuristics/
│   ├── local_search.py            # Relocate/swap with TW feasibility
│   ├── simulated_annealing.py     # SA with TW checks
│   ├── tabu_search.py             # TS with customer-route tabu
│   ├── iterated_greedy.py         # IG: remove + Solomon reinsert
│   ├── vns.py                     # VNS: relocate → swap → cross
│   ├── genetic_algorithm.py       # GA: giant-tour + TW split
│   └── ant_colony.py              # ACO with TW urgency
├── variants/
│   └── soft_time_windows/         # Soft TW penalties
└── tests/                         # 6 test files
    ├── test_vrptw.py              # Core algorithms
    ├── test_vrptw_ts.py           # TS
    ├── test_vrptw_aco.py          # ACO
    ├── test_vrptw_ig.py           # IG
    ├── test_vrptw_ls.py           # LS
    └── test_vrptw_vns.py          # VNS
```

**Total:** 2 heuristics (1 file), 7 metaheuristics/LS, 1 variant, 6 test files.

---

## 8. Key References

- Solomon, M.M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254-265.
- Gehring, H. & Homberger, J. (1999). A parallel hybrid evolutionary metaheuristic for the vehicle routing problem with time windows. *Proceedings of EUROGEN*, 57-64.
- Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455-472.
- Vidal, T., Crainic, T.G., Gendreau, M., Lahrichi, N. & Rei, W. (2013). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. *Operations Research*, 60(3), 611-624.
- Toth, P. & Vigo, D., eds. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). SIAM.

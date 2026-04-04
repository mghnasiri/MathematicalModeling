# Capacitated Vehicle Routing Problem (CVRP)

## 1. Problem Definition

- **Input:**
  - A depot (node 0) and a set $N = \{1, 2, \ldots, n\}$ of customers
  - Demands $q_i > 0$ for each customer $i \in N$
  - Pairwise distances $d_{ij} \geq 0$ between all nodes (including depot)
  - Vehicle capacity $Q > 0$ (identical fleet)
  - Number of available vehicles $K$ (sometimes unlimited)
- **Decision:** Partition customers into routes and determine the visit sequence within each route
- **Objective:** Minimize total travel distance across all routes
- **Constraints:**
  - Each customer visited exactly once by exactly one vehicle
  - Each route starts and ends at the depot
  - Total demand on each route does not exceed $Q$
- **Classification:** Combinatorial optimization (NP-hard)

**Complexity:** NP-hard — generalizes both the TSP (single vehicle, no capacity) and the Bin Packing Problem (assignment only, no routing). Even determining the minimum number of vehicles is NP-hard.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of customers | $\mathbb{Z}^+$ |
| $K$ | Number of vehicles | $\mathbb{Z}^+$ |
| $Q$ | Vehicle capacity | $\mathbb{R}_{> 0}$ |
| $q_i$ | Demand of customer $i$ | $\mathbb{R}_{> 0}$ |
| $d_{ij}$ | Distance from node $i$ to node $j$ | $\mathbb{R}_{\geq 0}$ |
| $x_{ij}$ | Number of times edge $(i,j)$ is traversed (2-index) | $\{0, 1, 2\}$ |
| $x_{ijk}$ | 1 if vehicle $k$ traverses arc $(i,j)$ (3-index) | $\{0, 1\}$ |

### Formulation A: Two-Index Vehicle Flow

$$\min \sum_{i=0}^{n} \sum_{j=0, j \neq i}^{n} d_{ij}\, x_{ij} \tag{1}$$

$$\text{s.t.} \quad \sum_{j=1}^{n} x_{0j} = 2K \quad \text{(exactly } K \text{ vehicles leave depot)} \tag{2}$$

$$\sum_{j=0, j \neq i}^{n} x_{ij} = 2 \quad \forall\, i \in N \quad \text{(each customer has degree 2)} \tag{3}$$

$$\sum_{i \in S} \sum_{j \in S, j \neq i} x_{ij} \leq |S| - \lceil \sum_{i \in S} q_i / Q \rceil \quad \forall\, S \subseteq N,\; |S| \geq 2 \quad \text{(rounded capacity cuts)} \tag{4}$$

$$x_{ij} \in \{0, 1\} \;\forall i,j \neq 0;\quad x_{0j} \in \{0, 1, 2\} \tag{5}$$

**Strengths:** Compact (no vehicle index). Constraint (4) simultaneously eliminates subtours and enforces capacity.
**Weaknesses:** Exponentially many rounded capacity cuts — added via lazy constraint generation.

### Formulation B: Three-Index Vehicle Flow

$$\min \sum_{k=1}^{K} \sum_{i=0}^{n} \sum_{j=0, j \neq i}^{n} d_{ij}\, x_{ijk} \tag{6}$$

$$\text{s.t.} \quad \sum_{k=1}^{K} \sum_{j=0}^{n} x_{ijk} = 1 \quad \forall\, i \in N \quad \text{(visit each customer once)} \tag{7}$$

$$\sum_{j=0}^{n} x_{0jk} = 1 \quad \forall\, k \quad \text{(each vehicle leaves depot)} \tag{8}$$

$$\sum_{i=0}^{n} x_{ihk} - \sum_{j=0}^{n} x_{hjk} = 0 \quad \forall\, h \in N,\, k \quad \text{(flow conservation)} \tag{9}$$

$$\sum_{i=1}^{n} q_i \sum_{j=0}^{n} x_{ijk} \leq Q \quad \forall\, k \quad \text{(capacity)} \tag{10}$$

**Strengths:** Explicit vehicle assignment — easy to extract routes.
**Weaknesses:** Symmetry among identical vehicles; needs symmetry-breaking constraints for efficient solving.

---

## 3. Variants

This repository implements **10 CVRP variants** — one of the most comprehensive variant collections:

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Multi-Depot (MDVRP) | `variants/multi_depot/` | Multiple depots; vehicles assigned to home depots |
| Split Delivery (SDVRP) | `variants/split_delivery/` | Customer demand can be split across vehicles |
| Open VRP (OVRP) | `variants/open_vrp/` | Vehicles need not return to depot |
| Periodic (PVRP) | `variants/periodic/` | Service over multi-day planning horizon |
| Cumulative (CumVRP) | `variants/cumulative/` | Minimize sum of arrival times (latency) |
| Backhaul (VRPB) | `variants/backhaul/` | Linehaul deliveries before backhaul pickups |
| Backhauls (VRPB+) | `variants/backhauls/` | Mixed linehaul/backhaul customers |
| Electric (EVRP) | `variants/electric/` | Battery constraints + recharging stations |
| Multi-Compartment (MCVRP) | `variants/multi_compartment/` | Separate compartments for incompatible goods |
| Multi-Trip (MTVRP) | `variants/multi_trip/` | Vehicles can perform multiple trips |

### 3.1 Split Delivery VRP

Relaxes the single-visit constraint — a customer's demand can be served by multiple vehicles. Can reduce total distance by up to 50% vs. CVRP. NP-hard but admits a 2-approximation.

### 3.2 Electric VRP

Vehicles have limited battery range. Routes must include visits to recharging stations when energy drops too low. Models the growing electric vehicle logistics sector.

### 3.3 Multi-Depot VRP

Multiple depots, each with its own fleet. Each vehicle starts and ends at its home depot. Combines facility assignment with routing.

See each variant's `README.md` for detailed formulations.

---

## 4. Benchmark Instances

### CVRPLIB

The standard CVRP benchmark repository with instances from multiple sources:

| Set | Author | Year | Instances | Size Range |
|-----|--------|------|-----------|-----------|
| A/B/P/E | Augerat et al. | 1995 | 82 | 16-101 customers |
| M | Christofides & Eilon | 1969 | 14 | 51-199 customers |
| X | Uchoa et al. | 2017 | 100 | 100-1000 customers |

**URL:** http://vrp.atd-lab.inf.puc-rio.br/index.php/en/

### Instances in This Repository

| Instance | Customers | Capacity | Source |
|----------|-----------|----------|--------|
| small6 | 6 | 15 | Handcrafted (two clusters) |
| christofides1 | 5 | 6 | Clarke-Wright example |
| medium12 | 12 | 40 | Random with mixed demands |

### Small Illustrative Instance

```
Depot: (0, 0)
Customers: (2,4), (5,2), (7,4), (3,7), (6,7), (8,1)
Demands:    3,     5,     4,     6,     3,     4
Capacity: 15
```

---

## 5. Solution Methods

### 5.1 Exact Methods

Branch-and-cut with rounded capacity cuts (constraint 4 above) and 2-path cuts. The Concorde-based CVRPSEP library provides separation routines. State-of-the-art: Branch-Cut-and-Price (Fukasawa et al., 2006) solves instances up to ~300 customers.

### 5.2 Constructive Heuristics

#### Clarke-Wright Savings (Clarke & Wright, 1964)

**Idea:** Start with $n$ direct depot-customer-depot routes. Merge route pairs $(i, j)$ by decreasing savings $s(i,j) = d(0,i) + d(0,j) - d(i,j)$, subject to capacity feasibility and only merging at route endpoints.

**Complexity:** $O(n^2 \log n)$ for savings computation + sorting.

```
ALGORITHM ClarkeWrightSavings(d, q, Q)
  routes ← {[0, i, 0] for each customer i}
  savings ← {(s(i,j), i, j) for all i < j}
  Sort savings by decreasing value
  FOR each (s, i, j) in savings:
    IF i is route endpoint AND j is route endpoint
       AND routes(i) ≠ routes(j)
       AND demand(route_i) + demand(route_j) ≤ Q:
      Merge routes at i and j
  RETURN routes
```

#### Sweep Algorithm (Gillett & Miller, 1974)

**Idea:** Compute polar angle of each customer from depot. Sweep clockwise, adding customers to the current route until capacity exceeded; then start a new route.

**Complexity:** $O(n \log n)$. Multi-start: try multiple starting angles.

### 5.3 Improvement Heuristics / Local Search

| Neighborhood | Move | Scope |
|-------------|------|-------|
| Relocate | Move one customer to another route | Inter-route |
| Swap | Exchange customers between two routes | Inter-route |
| 2-opt* | Cross two route segments | Inter-route |
| 2-opt | Reverse segment within one route | Intra-route |
| Or-opt | Move 1-3 consecutive customers | Intra-route |

**2-opt\*** (Potvin & Rousseau, 1995) is the inter-route version of 2-opt: given two routes, remove one edge from each and reconnect the resulting four segments in a different combination. Unlike intra-route 2-opt, this can transfer customers between routes.

### 5.4 Metaheuristics

This repository implements **7 metaheuristics** for CVRP:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Relocate, swap, 2-opt*, 2-opt, Or-opt |
| 2 | Simulated Annealing (SA) | Trajectory | Multi-neighborhood with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Recency-based tabu on relocated customers |
| 4 | Iterated Greedy (IG) | Trajectory | Remove $d$ customers + reinsert via cheapest insertion |
| 5 | Variable Neighborhood Search (VNS) | Trajectory | Systematic change among relocate/swap/2-opt* |
| 6 | Genetic Algorithm (GA) | Population | Giant-tour encoding, OX crossover, split decoder (Prins, 2004) |
| 7 | Ant Colony Optimization (ACO) | Population | Pheromone on customer-customer edges |

**Giant-tour encoding** (Prins, 2004) represents a CVRP solution as a single permutation of all customers. A split procedure (linear-time DP) optimally partitions this permutation into capacity-feasible routes. This allows using TSP-style crossover operators (OX) on CVRP.

### 5.5 State-of-the-Art

**Adaptive Large Neighborhood Search (ALNS)** (Ropke & Pisinger, 2006) and **Hybrid Genetic Search (HGS)** (Vidal et al., 2012) are the current state-of-the-art for CVRP. HGS consistently achieves <0.5% gap on standard benchmarks.

---

## 6. Implementation Guide

### Modeling Tips

- **Savings precomputation:** The Clarke-Wright savings matrix can be computed in $O(n^2)$ and reused across multi-start runs.
- **Route feasibility:** Maintain running demand totals per route for $O(1)$ feasibility checks during local search moves.
- **Split procedure:** Prins's split runs in $O(n)$ for a given giant tour. It finds the optimal split by dynamic programming: $f(k) = \min_{j < k} [f(j) + \text{cost}(\pi[j{+}1:k])]$ subject to capacity.

### Common Pitfalls

- **Depot handling:** Node 0 is the depot. Customers are nodes 1 through $n$. Distance matrices must include the depot row/column.
- **Empty routes:** Some formulations allow empty routes (vehicle stays at depot). Ensure this doesn't artificially inflate solution costs.
- **Symmetry:** The 3-index formulation has vehicle symmetry. Without symmetry-breaking, the solver explores many equivalent solutions.

---

## 7. Computational Results Summary

| Method | Category | Typical Gap (small, $n{\leq}50$) | Typical Gap (large, $n{>}100$) |
|--------|----------|------|------|
| Clarke-Wright | Heuristic | 5-15% | 5-15% |
| Sweep | Heuristic | 10-20% | 10-20% |
| CW + 2-opt* | Heuristic+LS | 2-5% | 3-8% |
| SA | Metaheuristic | 1-3% | 2-5% |
| GA (giant-tour) | Metaheuristic | 1-3% | 2-5% |
| ACO | Metaheuristic | 1-3% | 2-5% |
| HGS (reference) | State-of-art | <0.5% | <0.5% |

**Scale guidance:**
- $n \leq 50$: Exact methods feasible; CW + local search within 2-5%.
- $n = 50{-}200$: Metaheuristics (SA, GA, ACO). 1-5% from BKS.
- $n > 200$: ALNS or HGS. Standard metaheuristics may exceed 5% gap.

---

## 8. Implementations in This Repository

```
cvrp/
├── instance.py                    # CVRPInstance, CVRPSolution, validation
│
├── heuristics/
│   ├── clarke_wright.py           # Clarke-Wright savings (1964)
│   └── sweep.py                   # Angular sweep + multi-start (1974)
│
├── metaheuristics/
│   ├── local_search.py            # Relocate, swap, 2-opt*, 2-opt, Or-opt
│   ├── simulated_annealing.py     # SA with multi-neighborhood
│   ├── tabu_search.py             # TS with recency-based tabu
│   ├── iterated_greedy.py         # IG: remove + reinsert
│   ├── vns.py                     # VNS: systematic neighborhood change
│   ├── genetic_algorithm.py       # GA: giant-tour + split decoder (Prins)
│   └── ant_colony.py              # ACO/MMAS with pheromone trails
│
├── variants/
│   ├── multi_depot/               # MDVRP
│   ├── split_delivery/            # SDVRP
│   ├── open_vrp/                  # OVRP
│   ├── periodic/                  # PVRP
│   ├── cumulative/                # CumVRP
│   ├── backhaul/                  # VRPB
│   ├── backhauls/                 # VRPB+ (mixed)
│   ├── electric/                  # EVRP
│   ├── multi_compartment/         # MCVRP
│   └── multi_trip/                # MTVRP
│
└── tests/                         # 6 test files
    ├── test_cvrp.py               # Core algorithms (CW, sweep, SA, GA)
    ├── test_cvrp_ts.py            # Tabu Search
    ├── test_cvrp_aco.py           # Ant Colony
    ├── test_cvrp_ig.py            # Iterated Greedy
    ├── test_cvrp_ls.py            # Local Search
    └── test_cvrp_vns.py           # VNS
```

**Total:** 2 constructive heuristics, 7 metaheuristics/LS, 10 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Dantzig, G.B. & Ramser, J.H. (1959). The truck dispatching problem. *Management Science*, 6(1), 80-91.
- Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research*, 12(4), 568-581.
- Gillett, B.E. & Miller, L.R. (1974). A heuristic algorithm for the vehicle-dispatch problem. *Operations Research*, 22(2), 340-349.
- Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985-2002.

### Advanced Methods

- Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455-472.
- Vidal, T., Crainic, T.G., Gendreau, M. & Prins, C. (2012). A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows. *Computers & Operations Research*, 40(1), 475-489.

### Books and Surveys

- Toth, P. & Vigo, D., eds. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). SIAM.
- Laporte, G. (2009). Fifty years of vehicle routing. *Computers & Operations Research*, 36(11), 3068-3074.

### Benchmark

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T. & Subramanian, A. (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. *European Journal of Operational Research*, 257(3), 845-858.

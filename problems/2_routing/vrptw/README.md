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

**Decision variables:**

| Variable | Definition | Domain |
|----------|-----------|--------|
| $x_{ijk}$ | 1 if vehicle $k$ travels directly from $i$ to $j$ | $\{0,1\}$ |
| $b_{ik}$ | Begin-service time at node $i$ by vehicle $k$ | $\mathbb{R}_{\geq 0}$ |
| $y_{ik}$ | 1 if customer $i$ is served by vehicle $k$ | $\{0,1\}$ |

**Full formulation (two-index arc-flow):**

$$\min \sum_{k=1}^{K} \sum_{i=0}^{n} \sum_{j=0}^{n} d_{ij} \, x_{ijk} \tag{3}$$

Subject to:

$$\sum_{k=1}^{K} \sum_{j=0}^{n} x_{ijk} = 1 \quad \forall\, i \in \{1,\dots,n\} \quad \text{(visit once)} \tag{4}$$

$$\sum_{j=1}^{n} x_{0jk} \leq 1 \quad \forall\, k \quad \text{(each vehicle leaves depot at most once)} \tag{5}$$

$$\sum_{i=0}^{n} x_{ihk} = \sum_{j=0}^{n} x_{hjk} \quad \forall\, h,k \quad \text{(flow conservation)} \tag{6}$$

$$\sum_{i=1}^{n} q_i \sum_{j=0}^{n} x_{ijk} \leq Q \quad \forall\, k \quad \text{(capacity)} \tag{7}$$

$$b_{ik} + s_i + t_{ij} - b_{jk} \leq M(1 - x_{ijk}) \quad \forall\, i,j,k \quad \text{(time propagation)} \tag{8}$$

$$e_i \leq b_{ik} \leq l_i \quad \forall\, i,k \quad \text{(time window feasibility)} \tag{9}$$

where $M$ is a sufficiently large constant (e.g., $M = l_i + s_i + t_{ij} - e_j$).

**Hierarchical objective.** In practice VRPTW is often solved as a hierarchical bi-objective problem: minimize the number of vehicles first (primary), then minimize total distance (secondary). This avoids solutions with many lightly-loaded routes. The vehicle minimization component can be modeled by adding a large penalty $\Phi$ per vehicle used:

$$\min \;\Phi \sum_{k=1}^{K} \sum_{j=1}^{n} x_{0jk} \;+\; \sum_{k} \sum_{i} \sum_{j} d_{ij}\, x_{ijk} \tag{10}$$

with $\Phi \gg \max_{i,j} d_{ij}$ to enforce lexicographic priority on vehicle count.

**Model size:** $O(n^2 K)$ binary variables, $O(nK)$ continuous variables.
Practical for $n \leq 25$ with modern MIP solvers (CPLEX, Gurobi, HiGHS).
For larger instances, column generation (see Section 5.4) is the dominant exact approach.

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

Each class has 100-customer and 25-customer variants. Series 1 (e.g., C101-C109) has narrow time windows and a short planning horizon; Series 2 (e.g., C201-C208) has wide windows and a long horizon, allowing fewer vehicles.

**URL:** https://www.sintef.no/projectweb/top/vrptw/

### Best Known Solutions (100 customers)

| Instance | Class | Vehicles | Distance | Source |
|----------|-------|----------|----------|--------|
| C101 | Clustered, narrow TW | 10 | 828.94 | [TODO: verify BKS] |
| C201 | Clustered, wide TW | 3 | 591.56 | [TODO: verify BKS] |
| R101 | Random, narrow TW | 19 | 1645.79 | [TODO: verify BKS] |
| R201 | Random, wide TW | 4 | 1252.37 | [TODO: verify BKS] |
| RC101 | Mixed, narrow TW | 14 | 1696.94 | [TODO: verify BKS] |
| RC201 | Mixed, wide TW | 4 | 1406.91 | [TODO: verify BKS] |

### Best Known Solutions (25 customers)

| Instance | Class | Vehicles | Distance | Source |
|----------|-------|----------|----------|--------|
| C101-25 | Clustered, narrow TW | 3 | 191.3 | [TODO: verify BKS] |
| C201-25 | Clustered, wide TW | 2 | 214.7 | [TODO: verify BKS] |
| R101-25 | Random, narrow TW | 8 | 617.1 | [TODO: verify BKS] |
| R201-25 | Random, wide TW | 2 | 463.3 | [TODO: verify BKS] |
| RC101-25 | Mixed, narrow TW | 4 | 461.1 | [TODO: verify BKS] |
| RC201-25 | Mixed, wide TW | 2 | 360.2 | [TODO: verify BKS] |

### Gehring-Homberger Benchmark

Extended Solomon instances with 200, 400, 600, 800, and 1000 customers. Same C/R/RC classes. The standard large-scale VRPTW benchmark.

**URL:** https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/

### Instances in This Repository

| Instance | Customers | Type |
|----------|-----------|------|
| solomon_c101_mini | 8 | Clustered (from C101) |
| tight_tw5 | 5 | Narrow windows |

---

## 5. Solution Methods

### 5.1 Constructive Heuristics

#### Solomon I1 Insertion (Solomon, 1987)

**Idea:** Start with a seed customer (farthest unrouted). Iteratively insert the customer-position pair that minimizes a composite criterion combining distance increase and time urgency. When no more insertions are feasible, start a new route.

**Criterion c1 (insertion cost).** For inserting unrouted customer $u$ between consecutive nodes $i$ and $j$ in the current route:

$$c_{11}(i,u,j) = d_{iu} + d_{uj} - \mu \cdot d_{ij} \tag{distance increase}$$

$$c_{12}(i,u,j) = b_j^{\text{new}} - b_j^{\text{old}} \tag{push-forward at } j$$

$$c_1(i,u,j) = \alpha_1 \cdot c_{11}(i,u,j) + \alpha_2 \cdot c_{12}(i,u,j), \quad \alpha_1 + \alpha_2 = 1$$

The push-forward $c_{12}$ measures how much inserting $u$ delays the service start at subsequent customer $j$. An insertion is feasible only if the begin-service time $b_v \leq l_v$ for every customer $v$ downstream of the insertion point.

**Criterion c2 (customer selection).** Among all unrouted customers with a feasible insertion, choose the customer $u^*$ that maximizes:

$$c_2(u) = \lambda \cdot d_{0u} - c_1(i^*, u, j^*) \tag{urgency vs. cost}$$

where $(i^*, j^*)$ is the best insertion position for $u$. The $\lambda \cdot d_{0u}$ term favors customers far from the depot, which are harder to serve later.

**Pseudocode:**

```
SOLOMON-I1(instance, alpha1, alpha2, mu, lambda):
    unrouted <- {1, ..., n}
    routes <- []
    WHILE unrouted is not empty:
        seed <- argmax_{u in unrouted} d(0, u)      // farthest from depot
        route <- [seed]; remove seed from unrouted
        REPEAT:
            best_c2 <- -inf; best_u <- nil
            FOR each u in unrouted:
                best_c1 <- +inf; best_pos <- nil
                FOR each position p in {0, ..., |route|}:
                    IF insert(u, p) is feasible (capacity + TW):
                        c1 <- alpha1 * c11(i_p, u, j_p) + alpha2 * c12(i_p, u, j_p)
                        IF c1 < best_c1: best_c1 <- c1; best_pos <- p
                IF best_pos != nil:
                    c2 <- lambda * d(0, u) - best_c1
                    IF c2 > best_c2: best_c2 <- c2; best_u <- u
            IF best_u != nil:
                insert best_u at its best_pos in route
                remove best_u from unrouted
            ELSE: BREAK  // no more feasible insertions
        routes.append(route)
    RETURN routes
```

**Default parameters (this repository):** $\alpha_1 = 1.0$, $\alpha_2 = 0.0$, $\mu = 1.0$, $\lambda = 1.0$.

**Complexity:** $O(n^2 K)$ where $K$ is the number of routes.

#### Nearest Neighbor with Time Windows

Greedy: from the current position, go to the nearest *feasible* unvisited customer (respecting capacity and time windows). The feasibility check at each step verifies three conditions:

1. **Capacity:** current load + $q_u \leq Q$
2. **Time window:** arrival at $u$ does not exceed $l_u$
3. **Depot return:** after serving $u$, the vehicle can return to the depot before $l_0$

When no feasible neighbor exists, the current route closes and a new route begins from the depot.

**Complexity:** $O(n^2)$ per solution (each step scans remaining customers).

#### Push-Forward Insertion Cost

The **push-forward** at position $p$ in a route measures the cascading delay caused by inserting a customer. If customer $u$ is inserted between $i$ and $j$:

$$\text{PF}(u, p) = b_j^{\text{new}} - b_j^{\text{old}}$$

Because of time window waiting, the push-forward may be absorbed: if the vehicle was already waiting at $j$ (i.e., $a_j < e_j$), the delay is partially or fully absorbed by the slack. The **forward time slack** at position $p$ is:

$$F_p = \min_{p \leq k \leq |\text{route}|} \bigl(l_k - b_k\bigr) - \sum_{m=p}^{k-1} w_m$$

where $w_m = \max(0, e_m - a_m)$ is the waiting time at customer $m$. An insertion at position $p$ is feasible if and only if $\text{PF}(u, p) \leq F_p$. Precomputing forward time slack allows $O(1)$ feasibility checks for relocate and swap moves instead of the naive $O(k)$ propagation.

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

**TW-feasibility check:** All inter-route moves must verify that the new route respects time windows for every customer downstream of the insertion point. This requires $O(k)$ propagation per move (where $k$ is route length). Using precomputed forward time slack reduces this to $O(1)$ (see Section 5.1).

#### Metaheuristic Parameter Tables

**Simulated Annealing (SA)** -- Chiang & Russell (1996):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{\max}$ | 50,000 | Total SA iterations |
| Initial temperature | $T_0$ | $0.05 \cdot z_0$ | Auto-calibrated from initial cost $z_0$ |
| Cooling rate | $\alpha$ | 0.9995 | Geometric cooling: $T_{k+1} = \alpha \cdot T_k$ |
| Neighborhoods | -- | relocate, swap | Inter-route customer moves |
| Warm start | -- | Solomon I1 | Initial solution from constructive heuristic |

**Genetic Algorithm (GA)** -- Potvin & Bengio (1996):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Population size | $P$ | 50 | Number of individuals |
| Generations | $G$ | 300 | Number of GA generations |
| Mutation rate | $p_m$ | 0.15 | Probability of swap mutation |
| Tournament size | $k$ | 5 | Selection pressure parameter |
| Crossover | -- | OX | Order Crossover on giant tour |
| Encoding | -- | Giant tour | Permutation of all customers, split into routes |
| Warm start | -- | Solomon I1 | Seeds population[0] |

**Ant Colony Optimization (ACO)** -- Gambardella et al. (1999):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Number of ants | $m$ | $\min(n, 15)$ | Ants per iteration |
| Max iterations | $I_{\max}$ | 200 | Total ACO iterations |
| Pheromone importance | $\alpha$ | 1.0 | Exponent on pheromone trail |
| Heuristic importance | $\beta$ | 2.0 | Exponent on $1/d_{ij}$ heuristic |
| Evaporation rate | $\rho$ | 0.1 | Pheromone decay factor |
| Pheromone bounds | -- | MMAS | MAX-MIN Ant System bounds |
| Warm start | -- | Solomon I1 | Initial pheromone from best heuristic |

**Tabu Search (TS)** -- Cordeau et al. (2001):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{\max}$ | 3,000 | Total TS iterations |
| Tabu tenure | $\theta$ | $\lceil\sqrt{n}\rceil$ | Iterations a move stays tabu |
| Neighborhoods | -- | relocate, swap | Inter-route customer moves |
| Aspiration | -- | global best | Override tabu if new best found |
| Warm start | -- | Solomon I1 | Initial solution |

### 5.3 State-of-the-Art

**Adaptive Large Neighborhood Search (ALNS)** (Ropke & Pisinger, 2006): selects destroy/repair operators adaptively using roulette-wheel weights updated by performance. Typical destroy operators: random removal, worst removal (highest cost impact), Shaw removal (related customers). Repair operators: greedy insertion, regret-$k$ insertion. Achieves competitive results on Solomon 100-customer instances.

**Hybrid Genetic Search (HGS)** (Vidal et al., 2012, 2013): combines a GA with local search (education) and population diversity management. Uses a giant-tour representation with an optimal split procedure. Local search applies relocate, swap, 2-opt*, and Or-opt inter- and intra-route moves. Maintains feasible and infeasible sub-populations with penalized fitness. Currently the leading metaheuristic on Solomon and Gehring-Homberger benchmarks.

### 5.4 Column Generation for Vehicle Minimization

For exact or near-exact vehicle minimization, **column generation** (Desrochers et al., 1992) decomposes the VRPTW into a master problem and a pricing sub-problem.

**Master problem (set-partitioning relaxation):** select a minimum-cost subset of feasible routes covering all customers exactly once:

$$\min \sum_{r \in \Omega} c_r \, \theta_r$$

$$\text{s.t.} \quad \sum_{r \in \Omega} a_{ir} \, \theta_r = 1 \quad \forall\, i \in \{1,\dots,n\}$$

$$\theta_r \in \{0,1\} \quad \forall\, r$$

where $\Omega$ is the set of all feasible routes, $c_r$ is the cost of route $r$, $a_{ir} = 1$ if customer $i$ is in route $r$, and $\theta_r = 1$ if route $r$ is selected.

**Pricing sub-problem:** given dual prices $\pi_i$ from the LP relaxation of the master, generate a new column (route) with negative reduced cost:

$$\bar{c}_r = c_r - \sum_{i=1}^{n} \pi_i \, a_{ir} < 0$$

This is an **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)**: find a minimum-reduced-cost elementary path from depot to depot in the customer graph, subject to capacity $Q$, time windows $[e_i, l_i]$, and elementarity (no repeated customers). ESPPRC is NP-hard in general but solved efficiently via dynamic programming with resource extension and dominance pruning.

**Practical notes:**
- The LP relaxation of the master provides a strong lower bound on vehicle count
- Branch-and-price (B&P) embeds column generation within a branch-and-bound tree
- For large instances ($n > 100$), relaxing elementarity (allowing customer repetition in the sub-problem) improves tractability at the cost of weaker bounds
- State-of-the-art B&P solvers can optimally solve Solomon instances with 100 customers

---

## 6. Implementation Guide

### 6.1 Forward Time Slack

Precompute forward time slack $F_p$ at each position $p$ in every route. This allows $O(1)$ feasibility checks for relocate and swap moves instead of $O(k)$ full propagation. Recompute slack only for the affected route after each accepted move. See Section 5.1 for the formula.

### 6.2 Hierarchical Objective

Minimize vehicles first, then distance. This avoids solutions with many half-empty routes. In the implementation, use a weighted objective $\Phi \cdot K + D$ where $\Phi$ is a large constant (e.g., $10 \times$ the maximum single-route distance) and $K$ is the vehicle count. Compare solutions lexicographically: fewer vehicles always wins, distance breaks ties.

### 6.3 Waiting Time

When a vehicle arrives before $e_i$, it waits until $e_i$ to begin service. Waiting increases route duration but not distance cost. Some formulations penalize waiting to encourage tighter schedules. In this repository, waiting is free (distance-based objective).

### 6.4 Solution Encoding

This repository uses two encodings:

| Encoding | Used by | Description |
|----------|---------|-------------|
| Route list | SA, TS, LS, VNS, IG, ACO | Explicit list of routes, each a sequence of customer indices |
| Giant tour | GA | Single permutation of all customers, split into feasible routes |

The **giant-tour split** in the GA decoder scans left-to-right, starting a new route when adding the next customer would violate capacity, time windows, or depot return feasibility.

### 6.5 Neighborhood Operators

| Operator | Type | Description | Used by |
|----------|------|-------------|---------|
| Relocate | Inter-route | Move one customer from route $r_1$ to route $r_2$ | SA, TS, LS, VNS |
| Swap | Inter-route | Exchange one customer in $r_1$ with one in $r_2$ | SA, TS, LS, VNS |
| 2-opt* | Inter-route | Exchange tails of two routes | VNS |
| Cross-exchange | Inter-route | Exchange segments between two routes | VNS |
| Or-opt | Intra-route | Relocate a segment of 1-3 customers within a route | LS |
| 2-opt | Intra-route | Reverse a sub-sequence within a route | LS |

All inter-route operators require TW feasibility checks on both affected routes.

### 6.6 Data Structures

- **`VRPTWInstance`**: stores $n$, capacity $Q$, demand array, $(n+1) \times (n+1)$ distance matrix, $(n+1) \times 2$ time window array, $(n+1)$ service time array, and optional coordinates.
- **`VRPTWSolution`**: stores a list of routes (each a list of 1-indexed customer IDs) and the total distance. The `num_vehicles` property counts non-empty routes.
- **`validate_solution()`**: verifies all-visited-once, capacity, time windows, and depot return for every route. Returns `(is_valid, error_list)`.

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

1. Solomon, M.M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254-265. https://doi.org/10.1287/opre.35.2.254 -- Introduces the I1 insertion heuristic, the C/R/RC benchmark instances, and the time window propagation framework used throughout the VRPTW literature.

2. Desrochers, M., Desrosiers, J. & Solomon, M. (1992). A new optimization algorithm for the vehicle routing problem with time windows. *Operations Research*, 40(2), 342-354. https://doi.org/10.1287/opre.40.2.342 -- Column generation approach using ESPPRC pricing; foundation for branch-and-price methods.

3. Potvin, J.-Y. & Bengio, S. (1996). The vehicle routing problem with time windows part II: Genetic search. *INFORMS Journal on Computing*, 8(2), 165-172. https://doi.org/10.1287/ijoc.8.2.165 -- GA with route-based crossover for VRPTW; demonstrates population-based search effectiveness.

4. Chiang, W.-C. & Russell, R.A. (1996). Simulated annealing metaheuristics for the vehicle routing problem with time windows. *Annals of Operations Research*, 63(1), 3-27. https://doi.org/10.1007/BF02601637 -- SA design for VRPTW with relocate/swap neighborhoods and hierarchical objective.

5. Taillard, E.D., Badeau, P., Gendreau, M., Guertin, F. & Potvin, J.-Y. (1997). A tabu search heuristic for the vehicle routing problem with soft time windows. *Transportation Science*, 31(2), 170-186. https://doi.org/10.1287/trsc.31.2.170 -- Tabu search with adaptive memory; extends to soft time windows.

6. Gambardella, L.M., Taillard, E.D. & Agazzi, G. (1999). MACS-VRPTW: A multiple ant colony system for vehicle routing problems with time windows. In: Corne, D. et al. (eds) *New Ideas in Optimization*, McGraw-Hill, 63-76. -- Multiple ACS colonies: one minimizes vehicles, one minimizes distance.

7. Gehring, H. & Homberger, J. (1999). A parallel hybrid evolutionary metaheuristic for the vehicle routing problem with time windows. *Proceedings of EUROGEN*, 57-64. -- Extended Solomon benchmarks to 200-1000 customers.

8. Cordeau, J.-F., Laporte, G. & Mercier, A. (2001). A unified tabu search heuristic for vehicle routing problems with time windows. *Journal of the Operational Research Society*, 52(8), 928-936. https://doi.org/10.1057/palgrave.jors.2601163 -- Unified TS framework applicable to multiple VRP variants.

9. Berger, J. & Barkaoui, M. (2004). A parallel hybrid genetic algorithm for the vehicle routing problem with time windows. *Computers & Operations Research*, 31(12), 2037-2053. https://doi.org/10.1016/S0305-0548(03)00163-1 -- Parallel GA with route-based and sequence-based crossover operators.

10. Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455-472. https://doi.org/10.1287/trsc.40.4.455 -- Introduces ALNS with adaptive operator selection; widely adopted for VRPTW.

11. Vidal, T., Crainic, T.G., Gendreau, M., Lahrichi, N. & Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. *Operations Research*, 60(3), 611-624. https://doi.org/10.1287/opre.1120.1048 -- Hybrid Genetic Search framework with population diversity management; extended to VRPTW.

12. Vidal, T., Crainic, T.G., Gendreau, M. & Prins, C. (2013). Heuristics for multi-attribute vehicle routing problems: A survey and synthesis. *European Journal of Operational Research*, 231(1), 1-21. https://doi.org/10.1016/j.ejor.2013.02.053 -- Comprehensive survey of VRP metaheuristics including VRPTW-specific techniques.

13. Toth, P. & Vigo, D., eds. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). SIAM. -- Authoritative reference covering exact, heuristic, and metaheuristic approaches to all major VRP variants.

14. Baldacci, R., Mingozzi, A. & Roberti, R. (2012). Recent exact algorithms for solving the vehicle routing problem under capacity and time window constraints. *European Journal of Operational Research*, 218(3), 605-615. https://doi.org/10.1016/j.ejor.2011.11.037 -- Review of state-of-the-art exact methods including branch-and-cut-and-price.

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

### Formulation C: Set-Partitioning

Let $\mathcal{R}$ denote the set of all capacity-feasible routes, and for each route $r \in \mathcal{R}$ let $c_r$ be the route cost and $a_{ir} = 1$ if customer $i$ is in route $r$:

$$\min \sum_{r \in \mathcal{R}} c_r \, x_r \tag{11}$$

$$\text{s.t.} \quad \sum_{r \in \mathcal{R}} a_{ir}\, x_r = 1 \quad \forall\, i \in N \quad \text{(each customer in exactly one route)} \tag{12}$$

$$\sum_{r \in \mathcal{R}} x_r \leq K \quad \text{(vehicle limit)} \tag{13}$$

$$x_r \in \{0, 1\} \quad \forall\, r \in \mathcal{R} \tag{14}$$

**Strengths:** LP relaxation provides very tight lower bounds; forms the basis of Branch-Cut-and-Price.
**Weaknesses:** $|\mathcal{R}|$ is exponential in $n$ — requires column generation to enumerate routes implicitly.

### Capacity Cuts and Valid Inequalities

The two-index formulation uses **rounded capacity cuts** (constraint 4). For any customer subset $S \subseteq N$ with $|S| \geq 2$:

$$\sum_{i \in S}\sum_{j \notin S} x_{ij} \geq 2\left\lceil \frac{d(S)}{Q} \right\rceil \tag{15}$$

where $d(S) = \sum_{i \in S} q_i$ is the total demand of the subset. The left side counts the edges crossing the cut $\delta(S)$. Since each vehicle entering $S$ must also leave, and at least $\lceil d(S)/Q \rceil$ vehicles are needed to serve $S$, the right side gives a valid lower bound on the crossing edges.

These inequalities generalize subtour elimination: when $d(S) \leq Q$, the bound reduces to $\geq 2$, which simply forbids disconnected subtours. When $d(S) > Q$, the bound forces multiple vehicles to enter $S$, simultaneously enforcing both connectivity and capacity.

In practice, capacity cuts are separated on the fly during Branch-and-Cut. The **CVRPSEP** library (Lysgaard et al., 2004) provides efficient separation routines for rounded capacity inequalities, framed capacity inequalities, and strengthened comb inequalities.

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

### Best Known Solutions (CVRPLIB)

Selected instances from the classical sets with best-known solutions (BKS) from CVRPLIB:

| Instance | $n$ | $Q$ | BKS | Optimal? | Source |
|----------|-----|-----|-----|----------|--------|
| A-n32-k5 | 31 | 100 | 784 | Yes | Augerat et al. (1995) |
| A-n44-k6 | 43 | 100 | 937 | Yes | Augerat et al. (1995) |
| A-n60-k9 | 59 | 100 | 1354 | Yes | Augerat et al. (1995) |
| B-n50-k7 | 49 | 100 | 741 | Yes | Augerat et al. (1995) |
| E-n51-k5 | 50 | 160 | 521 | Yes | Christofides & Eilon (1969) |
| P-n101-k4 | 100 | 400 | 681 | [TODO: verify] | Augerat et al. (1995) |

**Note:** BKS values are taken from the CVRPLIB repository (http://vrp.atd-lab.inf.puc-rio.br/). Instances marked [TODO: verify] may have updated BKS values; consult the live CVRPLIB page for the latest figures.

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
  ── Phase 1: Initialization ──
  FOR each customer i ∈ {1, ..., n}:
    Create route R_i = [0, i, 0]        // individual depot-customer-depot route
    route_of[i] ← R_i
    route_demand[R_i] ← q_i

  ── Phase 2: Savings computation ──
  FOR each pair (i, j) with 1 ≤ i < j ≤ n:
    s(i,j) ← d(0,i) + d(0,j) - d(i,j)  // benefit of serving i,j consecutively
  Sort all s(i,j) in decreasing order → savings_list

  ── Phase 3: Parallel merge (consider all pairs at each step) ──
  FOR each (s, i, j) in savings_list:
    IF s ≤ 0: BREAK                    // no further benefit
    R_i ← route_of[i],  R_j ← route_of[j]
    IF R_i = R_j: CONTINUE             // already in same route
    IF i is NOT an endpoint of R_i: CONTINUE
    IF j is NOT an endpoint of R_j: CONTINUE
    IF route_demand[R_i] + route_demand[R_j] > Q: CONTINUE
    Merge R_i and R_j by connecting at endpoints i and j
    Update route_of[] for all customers in merged route

  RETURN {R : R is non-empty}
```

**Sequential variant:** Instead of considering all merges globally, build one route at a time by extending only the current route. Generally produces slightly worse solutions than parallel but is simpler to implement.

**Implementation note:** The `route_of[]` lookup and endpoint checks are $O(1)$ per merge if routes are stored as doubly-linked lists. This repository uses list-based routes with explicit endpoint checking (see `clarke_wright.py`).

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

#### 2-opt* Formal Definition

Given two routes $R_1 = (0, a_1, \ldots, a_i, a_{i+1}, \ldots, a_p, 0)$ and $R_2 = (0, b_1, \ldots, b_j, b_{j+1}, \ldots, b_q, 0)$, the 2-opt* move removes edges $(a_i, a_{i+1})$ and $(b_j, b_{j+1})$ and reconnects as:

- $R_1' = (0, a_1, \ldots, a_i, b_{j+1}, \ldots, b_q, 0)$
- $R_2' = (0, b_1, \ldots, b_j, a_{i+1}, \ldots, a_p, 0)$

The move is accepted only if both $R_1'$ and $R_2'$ satisfy capacity constraints. This exchanges the tails of two routes, enabling transfer of customer subsequences between vehicles.

#### Relocate (Inter-Route)

Remove customer $c$ from route $R_1$ and insert it at the best position in route $R_2$:

- **Cost delta:** $\Delta = [d(prev, next) - d(prev, c) - d(c, next)]$ + $[\text{best insertion cost in } R_2]$
- **Feasibility:** $\text{demand}(R_2) + q_c \leq Q$
- **Complexity:** $O(n)$ to evaluate all insertion positions in $R_2$

#### SWAP (Inter-Route)

Exchange customer $c_1 \in R_1$ with customer $c_2 \in R_2$:

- **Cost delta:** removal cost of $c_1$ from $R_1$ + removal cost of $c_2$ from $R_2$ + insertion cost of $c_2$ into $R_1$ + insertion cost of $c_1$ into $R_2$
- **Feasibility:** $\text{demand}(R_1) - q_{c_1} + q_{c_2} \leq Q$ and $\text{demand}(R_2) - q_{c_2} + q_{c_1} \leq Q$
- **Complexity:** $O(n^2)$ to evaluate all customer pairs across routes

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

#### Split Procedure (Prins, 2004)

The split algorithm converts a giant tour $\pi = (\pi_1, \pi_2, \ldots, \pi_n)$ into an optimal set of routes by solving a shortest-path problem on an auxiliary directed acyclic graph (DAG):

1. **Construct auxiliary graph $H$:** Create nodes $\{0, 1, \ldots, n\}$. For each arc $(i, j)$ with $0 \leq i < j \leq n$, the arc exists if route $(\pi_{i+1}, \ldots, \pi_j)$ is capacity-feasible, i.e., $\sum_{k=i+1}^{j} q_{\pi_k} \leq Q$. The arc cost equals the route cost $d(0, \pi_{i+1}) + \sum_{k=i+1}^{j-1} d(\pi_k, \pi_{k+1}) + d(\pi_j, 0)$.
2. **Shortest path:** Find the shortest path from node 0 to node $n$ in $H$. Each arc on this path corresponds to one route in the CVRP solution.
3. **Complexity:** $O(n)$ if the maximum route length is bounded (typical); $O(n^2)$ in the worst case.

```
ALGORITHM Split(π, d, q, Q)
  f[0] ← 0                           // DP cost to serve first 0 customers
  pred[0] ← -1
  FOR j = 1 TO n:
    f[j] ← ∞
    load ← 0;  cost ← 0
    FOR i = j DOWN TO 1:
      load ← load + q[π[i]]
      IF load > Q: BREAK
      IF i = j: cost ← d(0, π[j]) + d(π[j], 0)
      ELSE:     cost ← cost - d(π[i+1], 0) + d(π[i], π[i+1]) + d(0, π[i]) - d(0, π[i+1])
      // Simplified: recompute route cost for (π[i], ..., π[j])
      route_cost ← d(0, π[i]) + Σ d(π[k], π[k+1]) for k=i..j-1 + d(π[j], 0)
      IF f[i-1] + route_cost < f[j]:
        f[j] ← f[i-1] + route_cost
        pred[j] ← i-1
  Trace back from pred[n] to reconstruct routes
  RETURN routes
```

#### Parameter Tables

**Simulated Annealing** (this repository: `simulated_annealing.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{max}$ | 50,000 | Total neighborhood evaluations |
| Initial temperature | $T_0$ | auto | Calibrated to accept ~40% worsening moves initially |
| Cooling rate | $\alpha$ | 0.9995 | Geometric: $T_{k+1} = \alpha \cdot T_k$ |
| Neighborhoods | — | 3 | Relocate, swap, 2-opt* (uniform random selection) |

**Tabu Search** (this repository: `tabu_search.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{max}$ | 3,000 | Total neighborhood evaluations |
| Tabu tenure | $\theta$ | $\sqrt{n}$ | Iterations a customer remains tabu after relocation |
| Time limit | — | None | Optional wall-clock cutoff |
| Aspiration | — | global best | Override tabu if move yields new best |

**Genetic Algorithm** (this repository: `genetic_algorithm.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Population size | $P$ | 50 | Number of giant-tour individuals |
| Generations | $G$ | 300 | Evolutionary cycles |
| Mutation rate | $p_m$ | 0.15 | Probability of swap mutation per offspring |
| Tournament size | $k$ | 5 | Selection pressure |
| Crossover | OX | — | Order Crossover on giant tours |
| Local search | 2-opt | optional | Intra-route 2-opt on decoded routes |

**Ant Colony Optimization** (this repository: `ant_colony.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Number of ants | $m$ | $n$ | Ants per iteration |
| Max iterations | $I_{max}$ | 200 | Pheromone update cycles |
| Alpha | $\alpha$ | 1.0 | Pheromone importance exponent |
| Beta | $\beta$ | 3.0 | Heuristic ($1/d_{ij}$) importance exponent |
| Evaporation | $\rho$ | 0.1 | Pheromone decay rate |
| Pheromone bounds | MMAS | auto | MAX-MIN clamping from best-known solution |

### 5.5 Adaptive Large Neighborhood Search (ALNS)

ALNS (Ropke & Pisinger, 2006) generalizes LNS by maintaining a portfolio of destroy and repair operators, selected adaptively based on historical performance.

**Destroy operators** (remove $q$ customers from the current solution):

| Operator | Strategy |
|----------|----------|
| Random removal | Remove $q$ customers uniformly at random |
| Worst removal | Remove the $q$ customers whose removal saves the most distance |
| Shaw/related removal | Remove $q$ customers that are similar (close in distance, demand, or position) — encourages reassignment of natural clusters |
| Route removal | Remove all customers from one or more entire routes |

**Repair operators** (reinsert removed customers):

| Operator | Strategy |
|----------|----------|
| Greedy insertion | Insert each customer at the cheapest feasible position |
| Regret-2 | Insert the customer with highest regret (difference between best and 2nd-best insertion cost) — avoids greedy myopia |
| Regret-3 | Same idea with 3rd-best insertion — stronger diversification |

Operator weights are updated every $\sigma$ iterations using a roulette wheel: operators that found improving solutions receive a score bonus. This adaptive mechanism lets the search automatically favor effective operators for each instance.

### 5.6 Hybrid Genetic Search (HGS)

HGS (Vidal et al., 2012; updated in Vidal, 2022) is widely regarded as the current state-of-the-art metaheuristic for CVRP and many of its variants. Key design elements:

- **Infeasible solutions allowed:** The population can contain solutions that slightly violate capacity constraints. A penalty term is added to the fitness, and the penalty coefficient is dynamically adjusted to maintain a target ratio of feasible solutions.
- **Education (local search):** Each offspring undergoes an intensive local search phase using granular neighborhoods (relocate, swap, 2-opt*, CROSS) restricted to nearby customer pairs via a sparsified neighborhood graph.
- **Diversity management:** Fitness is a biased function of both solution cost and population diversity (measured by broken-pairs distance). This prevents premature convergence.
- **Population structure:** Separate feasible and infeasible sub-populations with survivor selection based on biased fitness.

HGS consistently achieves gaps below 0.5% on classical benchmarks and has found many new best-known solutions on the X-instances of Uchoa et al. (2017). The open-source implementation (HGS-CVRP) solves instances with 1000+ customers in practical time.

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

1. Dantzig, G.B. & Ramser, J.H. (1959). The truck dispatching problem. *Management Science*, 6(1), 80-91. https://doi.org/10.1287/mnsc.6.1.80
2. Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research*, 12(4), 568-581. https://doi.org/10.1287/opre.12.4.568
3. Gillett, B.E. & Miller, L.R. (1974). A heuristic algorithm for the vehicle-dispatch problem. *Operations Research*, 22(2), 340-349. https://doi.org/10.1287/opre.22.2.340
4. Potvin, J.-Y. & Rousseau, J.-M. (1995). An exchange heuristic for routeing problems with time windows. *Journal of the Operational Research Society*, 46(12), 1433-1446. https://doi.org/10.1057/jors.1995.198

### Exact Methods

5. Fukasawa, R., Longo, H., Lysgaard, J., Poggi de Aragao, M., Reis, M., Uchoa, E. & Werneck, R.F. (2006). Robust branch-and-cut-and-price for the capacitated vehicle routing problem. *Mathematical Programming*, 106(3), 491-511. https://doi.org/10.1007/s10107-005-0644-x
6. Lysgaard, J., Letchford, A.N. & Eglese, R.W. (2004). A new branch-and-cut algorithm for the capacitated vehicle routing problem. *Mathematical Programming*, 100(2), 423-445. https://doi.org/10.1007/s10107-003-0481-8

### Metaheuristics

7. Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985-2002. https://doi.org/10.1016/S0305-0548(03)00158-8
8. Ropke, S. & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455-472. https://doi.org/10.1287/trsc.1050.0135
9. Vidal, T., Crainic, T.G., Gendreau, M. & Prins, C. (2012). A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows. *Computers & Operations Research*, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018
10. Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. *Computers & Operations Research*, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643
11. Gendreau, M., Hertz, A. & Laporte, G. (1994). A tabu search heuristic for the vehicle routing problem. *Management Science*, 40(10), 1276-1290. https://doi.org/10.1287/mnsc.40.10.1276
12. Osman, I.H. (1993). Metastrategy simulated annealing and tabu search algorithms for the vehicle routing problem. *Annals of Operations Research*, 41(4), 421-451. https://doi.org/10.1007/BF02023004

### Books and Surveys

13. Toth, P. & Vigo, D., eds. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). MOS-SIAM Series on Optimization, SIAM. https://doi.org/10.1137/1.9781611973594
14. Laporte, G. (2009). Fifty years of vehicle routing. *Computers & Operations Research*, 36(11), 3068-3074. https://doi.org/10.1016/j.cor.2009.03.007
15. Laporte, G. (1992). The vehicle routing problem: An overview of exact and approximate algorithms. *European Journal of Operational Research*, 59(3), 345-358. https://doi.org/10.1016/0377-2217(92)90192-C

### Benchmark

16. Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T. & Subramanian, A. (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. *European Journal of Operational Research*, 257(3), 845-858. https://doi.org/10.1016/j.ejor.2016.08.012

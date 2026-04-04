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

### Formulation D: Single-Commodity Flow (SCF)

Gavish and Graves (1978) introduced a flow-based subtour elimination approach. A single commodity of $n$ units leaves node 1 (the depot), and each city consumes exactly 1 unit:

$$f_{ij} \leq (n - 1)\, x_{ij} \quad \forall\, (i, j) \in A \tag{8}$$

$$\sum_{j} f_{ji} - \sum_{j} f_{ij} = 1 \quad \forall\, i \in V \setminus \{1\} \quad \text{(consume 1 unit per city)} \tag{9}$$

$$f_{ij} \geq 0 \quad \forall\, (i, j) \in A \tag{10}$$

Combined with the degree constraints (2)-(3) and integrality (5), the flow conservation ensures connectivity. The LP relaxation is tighter than MTZ but weaker than DFJ.

**Variables:** $n^2$ binary $x_{ij}$ + $n^2$ continuous $f_{ij}$.
**Constraints:** $O(n^2)$ (compact).

### Formulation E: Multi-Commodity Flow (MCF)

Claus (1984) and Wong (1980) use $n - 1$ commodities, one for each city $k \in V \setminus \{1\}$. Each commodity $k$ sends one unit of flow from city 1 to city $k$:

$$\sum_{j} y^k_{1j} = 1 \quad \forall\, k \in V \setminus \{1\} \tag{11}$$

$$\sum_{j} y^k_{jk} = 1 \quad \forall\, k \in V \setminus \{1\} \tag{12}$$

$$\sum_{j} y^k_{ji} - \sum_{j} y^k_{ij} = 0 \quad \forall\, i \in V \setminus \{1, k\},\; \forall\, k \tag{13}$$

$$y^k_{ij} \leq x_{ij} \quad \forall\, (i, j) \in A,\; \forall\, k \tag{14}$$

$$y^k_{ij} \geq 0 \tag{15}$$

**Variables:** $n^2$ binary $x_{ij}$ + $O(n^3)$ continuous $y^k_{ij}$.
**Constraints:** $O(n^3)$.
**LP quality:** Equivalent to DFJ (the tightest compact-vs-exponential comparison), but at high computational cost due to the $O(n^3)$ variable count.

### Formulation Comparison

| Formulation | Binary vars | Continuous vars | Constraints | LP quality |
|-------------|-----------|----------------|-------------|------------|
| DFJ (subtour elim.) | $O(n^2)$ | 0 | $O(2^n)$ | Tightest (gold standard) |
| MTZ (compact) | $O(n^2)$ | $O(n)$ | $O(n^2)$ | Weakest |
| SCF (single-commodity flow) | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | Intermediate (tighter than MTZ) |
| MCF (multi-commodity flow) | $O(n^2)$ | $O(n^3)$ | $O(n^3)$ | Equivalent to DFJ |
| 1-Tree (Lagrangian) | n/a | n/a | n/a | Near-DFJ with subgradient tuning |

**Practical guidance:** DFJ with lazy constraint generation (via branch-and-cut) is the standard for exact solvers like Concorde. MTZ is convenient for small models in general-purpose MIP solvers. The 1-tree bound is preferred for custom B&B implementations.

### Christofides-Serdyukov 3/2-Approximation

The Christofides-Serdyukov algorithm (Christofides, 1976; Serdyukov, 1978) provides a 3/2-approximation for metric TSP, the best known polynomial-time guarantee for over four decades:

1. Compute a minimum spanning tree $T$ of the complete graph.
2. Let $O$ be the set of vertices with odd degree in $T$.
3. Find a minimum-weight perfect matching $M$ on the vertices in $O$.
4. Combine $T \cup M$ to form a multigraph $G'$ where every vertex has even degree.
5. Find an Eulerian tour in $G'$.
6. Shortcut the Eulerian tour to a Hamiltonian cycle by skipping already-visited cities.

**Guarantee:** The tour cost is at most $\frac{3}{2}$ times optimal. The key insight is that $\text{cost}(M) \leq \frac{1}{2}\,\text{OPT}$ (from the optimal tour restricted to $O$) and $\text{cost}(T) \leq \text{OPT}$. Note that in 2020 Karlin, Klein, and Oveis Gharan improved the bound slightly to $(3/2 - \epsilon)$ for some small $\epsilon > 10^{-36}$.

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

### TSPLIB Best Known Solutions

The following table lists commonly used TSPLIB benchmark instances with their best known solutions (BKS). All values below are established optima verified by the Concorde solver or exhaustive computation.

| Instance | $n$ | BKS | Optimal? | Notes |
|----------|-----|-----|----------|-------|
| eil51 | 51 | 426 | Yes | Eilon, 51-city Euclidean |
| berlin52 | 52 | 7542 | Yes | 52 locations in Berlin |
| st70 | 70 | 675 | Yes | 70-city instance |
| pr76 | 76 | 108159 | Yes | 76-city instance |
| kroA100 | 100 | 21282 | Yes | Krolak set A, 100 cities |
| ch150 | 150 | 6528 | Yes | Churritz, 150 cities |
| d198 | 198 | 15780 | Yes | Drilling problem |
| lin318 | 318 | 42029 | Yes | Lin, 318-city instance |
| pcb442 | 442 | 50778 | Yes | Printed circuit board |
| rat783 | 783 | 8806 | Yes | Rattled-grid, 783 cities |

**Source:** TSPLIB (Reinelt, 1991). Optimal solutions verified by Concorde (Applegate et al., 2006). Integer distances computed per TSPLIB EUC_2D convention (nint of Euclidean distance).

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

**High-level pseudocode:**

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

**Bitmask implementation detail:**

The key insight is representing subsets $S \subseteq V$ as bitmasks. City $j$ is in $S$ iff bit $j$ of the integer $S$ is set. The recurrence becomes:

```
ALGORITHM HeldKarpBitmask(d[0..n-1][0..n-1])
  // State: dp[S][j] = min-cost path from city 0 through all cities in S, ending at j
  // S is a bitmask; city 0 is always included (bit 0 set)
  dp[1 << 0][0] ← 0                          // base case: at city 0, visited {0}
  FOR S = 1 TO (1 << n) - 1:
    IF NOT (S & 1): CONTINUE                  // city 0 must be in S
    FOR j = 0 TO n-1:
      IF NOT (S & (1 << j)): CONTINUE         // j must be in S
      IF dp[S][j] = ∞: CONTINUE
      FOR k = 1 TO n-1:                       // extend to unvisited city k
        IF S & (1 << k): CONTINUE             // k must not be in S
        new_S ← S | (1 << k)
        dp[new_S][k] ← min(dp[new_S][k], dp[S][j] + d[j][k])
        parent[new_S][k] ← j                  // for tour reconstruction
  // Close the cycle: return to city 0
  full ← (1 << n) - 1
  opt ← min over j ∈ {1,...,n-1} of (dp[full][j] + d[j][0])
  // Reconstruct tour by backtracking through parent[]
  RETURN opt, reconstructed tour
```

**Memory note:** For $n = 23$, the DP table requires $2^{23} \times 23 \approx 193$ million entries. Using 64-bit floats, this is roughly 1.5 GB. Our implementation uses `numpy` arrays for efficient storage (see `exact/held_karp.py`).

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
    FOR i = 0 TO n-2:
      FOR j = i+2 TO n-1:
        IF i = 0 AND j = n-1: CONTINUE       // skip full reversal (equivalent tour)
        // O(1) delta evaluation — the critical optimization
        a ← tour[i],   b ← tour[i+1]
        c ← tour[j],   d ← tour[(j+1) mod n]
        Δ ← d(a,c) + d(b,d) - d(a,b) - d(c,d)
        IF Δ < -ε:                            // ε = 1e-10 for floating-point safety
          Reverse tour[i+1 .. j]
          improved ← TRUE
  RETURN tour
```

The $O(1)$ delta evaluation is essential: computing $\Delta$ from four edge lookups avoids recomputing the full $O(n)$ tour cost. For each candidate 2-opt move, only the two removed edges $\{(a, b), (c, d)\}$ and two inserted edges $\{(a, c), (b, d)\}$ change.

#### 3-opt: Eight Reconnection Cases

3-opt removes three edges from the tour, producing three segments. These three segments can be reconnected in $2^3 = 8$ ways (including the identity), of which 7 are non-trivial:

| Case | Reconnection | Includes |
|------|-------------|----------|
| 0 | A-B-C (identity) | No change |
| 1 | A-B'-C | Single 2-opt (reverse B) |
| 2 | A-B-C' | Single 2-opt (reverse C) |
| 3 | A'-B-C | Single 2-opt (reverse A) |
| 4 | A-B'-C' | Double 2-opt |
| 5 | A'-B-C' | Double 2-opt |
| 6 | A'-B'-C | Double 2-opt |
| 7 | A'-B'-C' | True 3-opt (not decomposable into 2-opt) |

Only case 7 represents a move not reachable by sequential 2-opt. The full 3-opt neighborhood has $O(n^3)$ size, making it expensive. **When to use 3-opt vs 2-opt:** 3-opt is useful for final polishing on small instances ($n < 200$) where 2-opt has converged to a local optimum. For larger instances, the cubic cost is prohibitive; use Or-opt or LK moves instead.

#### Or-opt: Segment Relocation

Or-opt (Or, 1976) relocates a segment of $\ell$ consecutive cities ($\ell \in \{1, 2, 3\}$) to another position in the tour:

```
ALGORITHM OrOpt(tour)
  improved ← TRUE
  WHILE improved:
    improved ← FALSE
    FOR seg_len ∈ {1, 2, 3}:                  // try segment lengths
      FOR i = 0 TO n-1:                       // segment start
        FOR j = 0 TO n-1:                     // insertion position
          IF j overlaps [i, i+seg_len): CONTINUE
          Compute Δ = (bridge cost + insert cost) - (remove cost + old edge)
          IF Δ < -ε:
            Remove segment tour[i..i+seg_len-1]
            Insert segment after position j
            improved ← TRUE
  RETURN tour
```

Or-opt is $O(n^2)$ per pass (like 2-opt) but explores a complementary neighborhood. The VND (Variable Neighborhood Descent) in this repository alternates 2-opt and Or-opt until neither improves.

#### Lin-Kernighan (LK) Moves

The Lin-Kernighan algorithm (Lin & Kernighan, 1973) performs variable-depth search: a sequence of edge swaps that generalizes $k$-opt for adaptive $k$. At each step, the algorithm:

1. Remove an edge $(t_1, t_2)$ from the tour.
2. Add a new edge $(t_2, t_3)$ not in the tour, with the gain criterion $G_i = g_1 + g_2 + \ldots + g_i > 0$ maintained greedily.
3. Continue extending the chain of swaps as long as the cumulative gain remains positive.
4. Close the tour and accept if the total improvement is positive.

**Why LK is powerful:** Unlike fixed-$k$ opt, LK adaptively determines the depth of the move. It effectively performs 3-opt, 4-opt, or higher moves when beneficial, while keeping the average cost closer to 2-opt. The Helsgaun extension (LKH) uses 5-opt sequential moves with candidate lists restricted to the $k$-nearest neighbors, achieving near-optimal results on instances with millions of cities.

**Note:** LK/LKH is not implemented in this repository but is the reference standard (see Section 5.5).

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

#### Parameter Tables

**Simulated Annealing (SA)** — as implemented in `metaheuristics/simulated_annealing.py`:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{\max}$ | 100,000 | Total number of 2-opt move attempts |
| Initial temperature | $T_0$ | Auto-calibrated | Set so 80% of uphill moves are accepted |
| Cooling rate | $\alpha$ | 0.9995 | Geometric: $T_{k+1} = \alpha \cdot T_k$ |
| Neighborhood | — | 2-opt | Random segment reversal |
| Warm start | — | Nearest neighbor | Initial tour from NN heuristic |

**Tabu Search (TS)** — as implemented in `metaheuristics/tabu_search.py`:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Max iterations | $I_{\max}$ | 5,000 | Full neighborhood scans |
| Tabu tenure | $\tau$ | $\lfloor\sqrt{n}\rfloor$ | Iterations a reversed segment stays tabu |
| Aspiration | — | Global best | Tabu overridden if move improves best known |
| Neighborhood | — | 2-opt | Full enumeration per iteration |

**Genetic Algorithm (GA)** — as implemented in `metaheuristics/genetic_algorithm.py`:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Population size | $P$ | 50 | Number of individuals |
| Generations | $G$ | 500 | Number of evolutionary cycles |
| Mutation rate | $p_m$ | 0.10 | Probability of swap mutation per offspring |
| Tournament size | $k$ | 5 | Selection pressure |
| Crossover | — | OX | Order Crossover preserving relative order |
| Elitism | — | 1 | Best individual always survives |
| Optional LS | — | 2-opt | Applied to each offspring if enabled |

**Ant Colony Optimization (ACO/MMAS)** — as implemented in `metaheuristics/ant_colony.py`:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Number of ants | $m$ | $n$ | One ant per city |
| Max iterations | $I_{\max}$ | 200 | Pheromone update cycles |
| Pheromone weight | $\alpha$ | 1.0 | Influence of pheromone trail |
| Visibility weight | $\beta$ | 3.0 | Influence of $1/d_{ij}$ heuristic |
| Evaporation rate | $\rho$ | 0.1 | Fraction of pheromone that evaporates |
| Pheromone bounds | — | MMAS | $[\tau_{\min}, \tau_{\max}]$ prevents stagnation |

#### Tour Encoding Strategies

| Encoding | Representation | Used by | Crossover |
|----------|---------------|---------|-----------|
| Permutation | Ordered list of city indices $[\pi_1, \ldots, \pi_n]$ | GA, SA, TS, IG, VNS | OX, PMX, CX, Edge recombination |
| Adjacency | $\text{adj}[i] = $ next city after $i$ | Edge recombination GA | Edge assembly |
| Random keys | Real vector $r \in [0,1]^n$; argsort gives permutation | PSO, DE (not implemented) | Arithmetic crossover |
| Binary matrix | $x_{ij} \in \{0,1\}$ edge variables | MIP formulations | n/a |

This repository uses **permutation encoding** throughout, which is the natural and most common representation for TSP. The random-key encoding is useful when adapting continuous optimization metaheuristics (PSO, Differential Evolution) to permutation problems, as standard arithmetic operators apply directly to the real-valued vector.

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

- Dantzig, G.B., Fulkerson, D.R. & Johnson, S.M. (1954). Solution of a large-scale traveling-salesman problem. *Journal of the Operations Research Society of America*, 2(4), 393-410. https://doi.org/10.1287/opre.2.4.393
- Held, M. & Karp, R.M. (1962). A dynamic programming approach to sequencing problems. *SIAM Journal on Applied Mathematics*, 10(1), 196-210. https://doi.org/10.1137/0110015
- Held, M. & Karp, R.M. (1970). The traveling-salesman problem and minimum spanning trees. *Operations Research*, 18(6), 1138-1162. https://doi.org/10.1287/opre.18.6.1138
- Karp, R.M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*, Plenum Press, 85-103. https://doi.org/10.1007/978-1-4684-2001-2_9
- Christofides, N. (1976). Worst-case analysis of a new heuristic for the travelling salesman problem. *Report 388, Graduate School of Industrial Administration, CMU*.
- Croes, G.A. (1958). A method for solving traveling salesman problems. *Operations Research*, 6(6), 791-812. https://doi.org/10.1287/opre.6.6.791
- Lin, S. & Kernighan, B.W. (1973). An effective heuristic algorithm for the traveling-salesman problem. *Operations Research*, 21(2), 498-516. https://doi.org/10.1287/opre.21.2.498
- Or, I. (1976). Traveling salesman-type combinatorial problems and their relation to the logistics of regional blood banking. *Ph.D. thesis*, Northwestern University.
- Miller, C.E., Tucker, A.W. & Zemlin, R.A. (1960). Integer programming formulation of traveling salesman problems. *Journal of the ACM*, 7(4), 326-329. https://doi.org/10.1145/321043.321046
- Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *Journal of the ACM*, 45(5), 753-782. https://doi.org/10.1145/290179.290180
- Dorigo, M. & Gambardella, L.M. (1997). Ant colony system: a cooperative learning approach to the traveling salesman problem. *IEEE Transactions on Evolutionary Computation*, 1(1), 53-66. https://doi.org/10.1109/4235.585892
- Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680. https://doi.org/10.1126/science.220.4598.671

### Books

- Applegate, D.L., Bixby, R.E., Chvatal, V. & Cook, W.J. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press. https://doi.org/10.1515/9781400841103
- Gutin, G. & Punnen, A.P., eds. (2007). *The Traveling Salesman Problem and Its Variations*. Springer. https://doi.org/10.1007/b101971

### Surveys

- Rosenkrantz, D.J., Stearns, R.E. & Lewis, P.M. (1977). An analysis of several heuristics for the traveling salesman problem. *SIAM Journal on Computing*, 6(3), 563-581. https://doi.org/10.1137/0206041
- Laporte, G. (1992). The traveling salesman problem: An overview of exact and approximate algorithms. *European Journal of Operational Research*, 59(2), 231-247. https://doi.org/10.1016/0377-2217(92)90138-Y
- Johnson, D.S. & McGeoch, L.A. (1997). The traveling salesman problem: A case study in local optimization. *Local Search in Combinatorial Optimization*, Wiley, 215-310.

### Solvers

- Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan traveling salesman heuristic. *European Journal of Operational Research*, 126(1), 106-130. https://doi.org/10.1016/S0377-2217(99)00284-2
- Helsgaun, K. (2009). General k-opt submoves for the Lin-Kernighan TSP heuristic. *Mathematical Programming Computation*, 1(2-3), 119-163. https://doi.org/10.1007/s12532-009-0004-6

### Benchmark

- Reinelt, G. (1991). TSPLIB — a traveling salesman problem library. *ORSA Journal on Computing*, 3(4), 376-384. https://doi.org/10.1287/ijoc.3.4.376

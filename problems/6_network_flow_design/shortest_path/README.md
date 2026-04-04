# Shortest Path Problem (SPP)

## 1. Problem Definition

- **Input:** Directed graph $G = (V, E)$ with edge weights $w(u,v)$, source $s$, target $t$
- **Decision:** Find a path from $s$ to $t$
- **Objective:** Minimize total path weight $\sum_{(u,v) \in P} w(u,v)$
- **Constraints:** Path must be simple (no cycles if negative weights exist)
- **Classification:** Polynomial — solvable in $O((V+E)\log V)$ for non-negative weights

### Complexity

| Variant | Algorithm | Complexity |
|---------|----------|-----------|
| Non-negative weights | Dijkstra | $O((V+E) \log V)$ |
| General weights | Bellman-Ford | $O(VE)$ |
| DAG | Topological sort + relax | $O(V+E)$ |
| All-pairs | Floyd-Warshall | $O(V^3)$ |
| Negative cycle detection | Bellman-Ford | $O(VE)$ |

---

## 2. Mathematical Formulation

### LP Formulation (shortest $s$-$t$ path)

$$\min \sum_{(u,v) \in E} w(u,v) \cdot x_{uv} \tag{1}$$

$$\sum_{v: (s,v) \in E} x_{sv} - \sum_{u: (u,s) \in E} x_{us} = 1 \quad \text{(flow out of source)} \tag{2}$$

$$\sum_{v: (t,v) \in E} x_{tv} - \sum_{u: (u,t) \in E} x_{ut} = -1 \quad \text{(flow into sink)} \tag{3}$$

$$\sum_{v} x_{uv} - \sum_{v} x_{vu} = 0 \quad \forall u \neq s, t \quad \text{(conservation)} \tag{4}$$

$$x_{uv} \geq 0 \tag{5}$$

The LP relaxation is always integral because the node-edge incidence matrix
of a directed graph is **totally unimodular** (TU). Every vertex of the LP
polyhedron has integer coordinates, so the LP optimum coincides with the
integer optimum. This is the fundamental reason that shortest path problems
are solvable in polynomial time despite being formulated as integer programs.

### Dual Formulation (Node Potentials)

The LP dual assigns a potential $\pi_v$ to each node:

$$\max\ \pi_t - \pi_s \tag{6}$$

$$\pi_v - \pi_u \leq w(u,v) \quad \forall (u,v) \in E \tag{7}$$

The dual constraints require that potentials respect edge weights: no
potential difference across an edge may exceed the edge weight. The optimal
dual potentials satisfy $\pi_v = d(s, v)$ (the shortest path distances from $s$).

The **reduced cost** of an edge $(u,v)$ is defined as:

$$\bar{w}(u,v) = w(u,v) - \pi_u + \pi_v \tag{8}$$

By complementary slackness, an edge $(u,v)$ lies on a shortest path if and
only if $\bar{w}(u,v) = 0$. Reduced costs are non-negative at optimality,
which is precisely the property that Johnson's algorithm exploits to
eliminate negative edge weights via potential reweighting.

### Strong Duality and Optimality Conditions

By LP strong duality, the minimum $s$-$t$ path cost equals the maximum
potential difference $\pi_t - \pi_s$. This connects to the max-flow min-cut
theorem: when edge weights represent capacities, the dual of the shortest
path LP yields the longest path (critical path) in a project network.

---

## 3. Variants

| Variant | Directory |
|---------|-----------|
| All-Pairs Shortest Path | `variants/all_pairs/` |

---

## 4. Solution Methods

### 4.1 Dijkstra's Algorithm (non-negative weights)

Dijkstra's algorithm (1959) is the standard single-source shortest path
algorithm for graphs with non-negative edge weights. It greedily extends
the shortest path tree by always processing the unvisited node with the
smallest tentative distance. The implementation in this repository uses
Python's `heapq` module (binary min-heap) with lazy deletion of stale
entries.

**Pseudocode (binary heap variant):**

```
ALGORITHM Dijkstra(G, w, s)
  dist[v] ← ∞ for all v; dist[s] ← 0
  prev[v] ← NIL for all v
  visited[v] ← false for all v
  Q ← min-priority-queue; INSERT(Q, (0, s))
  WHILE Q not empty:
    (d, u) ← EXTRACT-MIN(Q)
    IF visited[u]: CONTINUE        // lazy deletion of stale entries
    visited[u] ← true
    FOR each neighbor v of u with weight w(u,v):
      IF dist[u] + w(u,v) < dist[v]:
        dist[v] ← dist[u] + w(u,v)
        prev[v] ← u
        INSERT(Q, (dist[v], v))    // push new entry (old entry becomes stale)
  RETURN dist, prev
```

**Correctness argument.** When node $u$ is extracted from the priority
queue, $\text{dist}[u]$ is optimal. Proof by contradiction: if a shorter
path to $u$ existed, it would pass through some unvisited node $u'$ with
$\text{dist}[u'] \leq \text{dist}[u]$, but $u$ was the minimum in the
queue. Non-negative weights ensure no later relaxation can improve a
finalized distance.

**Priority queue variants:**

| Priority Queue | EXTRACT-MIN | DECREASE-KEY | Total Dijkstra |
|----------------|-------------|-------------|----------------|
| Binary heap | $O(\log V)$ | $O(\log V)$ | $O((V+E) \log V)$ |
| Fibonacci heap | $O(\log V)$ amortized | $O(1)$ amortized | $O(V \log V + E)$ |
| Array (unsorted) | $O(V)$ | $O(1)$ | $O(V^2)$ |

The Fibonacci heap variant achieves the theoretically optimal $O(V \log V + E)$
bound (Fredman and Tarjan, 1987), which is superior for dense graphs where
$E = \Theta(V^2)$. However, the constant factors make it slower than binary
heaps in practice for most graph sizes. The unsorted array variant with
$O(V^2)$ time is competitive for very dense graphs because it avoids heap
overhead entirely.

### 4.2 Bellman-Ford Algorithm (general weights)

The Bellman-Ford algorithm (Bellman, 1958; Ford, 1956) handles graphs with
negative edge weights and detects negative-weight cycles. It performs
$|V| - 1$ relaxation passes over all edges; if any further relaxation is
possible in a $(|V|)$-th pass, a negative cycle exists.

**Pseudocode:**

```
ALGORITHM BellmanFord(G, w, s)
  dist[v] ← ∞ for all v; dist[s] ← 0
  prev[v] ← NIL for all v
  FOR i = 1 TO |V| - 1:
    changed ← false
    FOR each edge (u, v, w(u,v)) in E:
      IF dist[u] + w(u,v) < dist[v]:
        dist[v] ← dist[u] + w(u,v)
        prev[v] ← u
        changed ← true
    IF NOT changed: BREAK             // early termination
  // Negative cycle detection (|V|-th pass):
  FOR each edge (u, v, w(u,v)) in E:
    IF dist[u] + w(u,v) < dist[v]:
      RETURN "Negative-weight cycle detected"
  RETURN dist, prev
```

**Correctness argument.** After $i$ relaxation passes, $\text{dist}[v]$
is optimal over all paths from $s$ to $v$ using at most $i$ edges. Since
any shortest path in a graph without negative cycles has at most $|V| - 1$
edges, $|V| - 1$ passes suffice. If the $|V|$-th pass still finds a
shorter path, a negative cycle must be reachable from $s$.

**Early termination.** The implementation includes early termination: if
no distance is updated in a pass, the algorithm stops. In the best case
(e.g., when edges are listed in topological order), this reduces the
number of passes to 1, though the worst case remains $O(VE)$.

### 4.3 A* Search Algorithm

A* (Hart, Nilsson, and Raphael, 1968) is a best-first search that uses a
heuristic function to guide exploration toward the target. It evaluates
each node $v$ using:

$$f(v) = g(v) + h(v)$$

where $g(v)$ is the known shortest distance from $s$ to $v$, and $h(v)$ is
a heuristic estimate of the distance from $v$ to the target $t$.

**Admissibility.** A heuristic $h$ is admissible if $h(v) \leq d(v, t)$
for all $v$ (it never overestimates the true distance). With an admissible
heuristic, A* is guaranteed to find an optimal shortest path.

**Consistency.** A heuristic is consistent (monotone) if
$h(u) \leq w(u,v) + h(v)$ for every edge $(u,v)$. Consistency implies
admissibility and ensures that each node is processed at most once (no
reopening), making A* with a consistent heuristic equivalent to Dijkstra
on a graph with modified edge weights $\bar{w}(u,v) = w(u,v) + h(v) - h(u)$.

**Common heuristics for geometric graphs:**
- Euclidean distance: $h(v) = \|v - t\|_2$ (admissible for straight-line distances)
- Manhattan distance: $h(v) = |x_v - x_t| + |y_v - y_t|$ (admissible for grid graphs)

**Complexity.** Worst case $O((V+E) \log V)$, same as Dijkstra, but with a
good heuristic it explores far fewer nodes, making it the preferred
point-to-point algorithm in practice (e.g., road network routing).

### 4.4 DAG Shortest Path (Topological Order Relaxation)

For directed acyclic graphs (DAGs), shortest paths can be computed in
$O(V + E)$ by relaxing edges in topological order. This handles negative
edge weights without the overhead of Bellman-Ford.

```
ALGORITHM DAG-ShortestPath(G, w, s)
  dist[v] ← ∞ for all v; dist[s] ← 0
  ORDER ← TopologicalSort(G)          // O(V + E)
  FOR each u in ORDER:
    FOR each neighbor v of u with weight w(u,v):
      IF dist[u] + w(u,v) < dist[v]:
        dist[v] ← dist[u] + w(u,v)
  RETURN dist
```

**Applications:** Critical path method (CPM) in project scheduling (RCPSP),
PERT analysis, longest path in DAGs (negate weights).

### 4.5 Floyd-Warshall Algorithm (All-Pairs)

Floyd-Warshall (Floyd, 1962; Warshall, 1962) solves the all-pairs shortest
path problem using dynamic programming. It considers each node $k$ as a
potential intermediate vertex and checks whether the path $i \to k \to j$
improves on the current best $i \to j$ path.

```
ALGORITHM Floyd-Warshall(W, n)
  dist ← copy of weight matrix W      // dist[i][j] = w(i,j) or ∞
  next[i][j] ← j if edge (i,j) exists, else NIL
  FOR k = 0 TO n-1:
    FOR i = 0 TO n-1:
      FOR j = 0 TO n-1:
        IF dist[i][k] + dist[k][j] < dist[i][j]:
          dist[i][j] ← dist[i][k] + dist[k][j]
          next[i][j] ← next[i][k]
  // Negative cycle detection: dist[i][i] < 0 for some i
  RETURN dist, next
```

**Complexity:** $O(V^3)$ time, $O(V^2)$ space. The triple loop structure
is simple and cache-friendly, making it competitive for dense graphs with
moderate $V$ (up to a few thousand nodes). Negative cycles are detected by
checking the diagonal: $\text{dist}[i][i] < 0$ implies node $i$ lies on a
negative cycle.

### 4.6 Johnson's Algorithm (All-Pairs, Sparse Graphs)

Johnson's algorithm (1977) computes all-pairs shortest paths efficiently on
sparse graphs by combining Bellman-Ford with repeated Dijkstra. The key
idea is **potential reweighting**: use Bellman-Ford to compute node
potentials that eliminate negative edge weights, then run Dijkstra from
each source on the reweighted graph.

```
ALGORITHM Johnson(G, w)
  // Step 1: Add virtual source q with zero-weight edges to all nodes
  Add node q to G
  FOR each v in V: add edge (q, v) with weight 0

  // Step 2: Bellman-Ford from q to get potentials h[v]
  h ← BellmanFord(G', w, q)
  IF negative cycle detected: RETURN "Negative cycle"
  Remove node q

  // Step 3: Reweight edges
  FOR each edge (u,v):
    w'(u,v) ← w(u,v) + h[u] - h[v]    // guaranteed non-negative

  // Step 4: Run Dijkstra from each source
  FOR each source s in V:
    d'[s] ← Dijkstra(G, w', s)
    FOR each v in V:
      dist[s][v] ← d'[s][v] - h[s] + h[v]  // un-reweight distances
  RETURN dist
```

**Complexity:** $O(VE + V^2 \log V)$ with Fibonacci heaps, or
$O(VE + V(V+E) \log V)$ with binary heaps. For sparse graphs where
$E = O(V)$, this is $O(V^2 \log V)$, significantly better than
Floyd-Warshall's $O(V^3)$.

### 4.7 BFS Shortest Path (Unweighted Graphs)

For unweighted graphs (or equivalently, unit-weight graphs), breadth-first
search computes shortest paths in $O(V + E)$ time. BFS explores nodes
in order of their distance from the source, so the first time a node is
reached is via a shortest path.

---

## 5. Algorithm Complexity Comparison

| Algorithm | Time | Space | Neg. Weights? | Neg. Cycles? | Notes |
|-----------|------|-------|---------------|--------------|-------|
| Dijkstra (binary heap) | $O((V+E) \log V)$ | $O(V)$ | No | No | Standard SSSP |
| Dijkstra (Fibonacci heap) | $O(V \log V + E)$ | $O(V)$ | No | No | Best for dense graphs |
| Dijkstra (unsorted array) | $O(V^2)$ | $O(V)$ | No | No | Simple, no heap overhead |
| Bellman-Ford | $O(VE)$ | $O(V)$ | Yes | Detects | Standard for negative weights |
| A* (binary heap) | $O((V+E) \log V)$ | $O(V)$ | No | No | Heuristic-guided, often sublinear in practice |
| DAG relaxation | $O(V+E)$ | $O(V)$ | Yes | N/A (acyclic) | Topological sort + single pass |
| BFS (unweighted) | $O(V+E)$ | $O(V)$ | N/A | N/A | Unit weights only |
| Floyd-Warshall | $O(V^3)$ | $O(V^2)$ | Yes | Detects | All-pairs, simple triple loop |
| Johnson's (Fibonacci heap) | $O(VE + V^2 \log V)$ | $O(V^2)$ | Yes | No (reweights) | All-pairs, best for sparse |
| Johnson's (binary heap) | $O(VE + V(V+E) \log V)$ | $O(V^2)$ | Yes | No (reweights) | Practical all-pairs |

**When to use which algorithm:**

- **Non-negative, single pair:** Dijkstra (or A* if a good heuristic is available)
- **Non-negative, single source:** Dijkstra
- **Negative weights, single source:** Bellman-Ford
- **DAG, any weights:** Topological order relaxation
- **All-pairs, dense graph:** Floyd-Warshall
- **All-pairs, sparse graph with negative weights:** Johnson's
- **Unweighted graph:** BFS

---

## 6. Implementations in This Repository

```
shortest_path/
├── instance.py                    # ShortestPathInstance, ShortestPathSolution
│                                  # from_edges(), from_matrix(), random()
│                                  # validate_solution(), benchmark instances
├── exact/
│   ├── dijkstra.py                # Dijkstra O((V+E) log V), binary heap
│   │                              # dijkstra() — single s-t path
│   │                              # dijkstra_all() — single-source all targets
│   └── bellman_ford.py            # Bellman-Ford O(VE), negative cycle detection
│                                  # bellman_ford() — single s-t path
├── variants/
│   └── all_pairs/                 # All-Pairs Shortest Path (APSP)
│       ├── instance.py            # APSPInstance, APSPSolution, validation
│       ├── heuristics.py          # Floyd-Warshall O(V^3), Repeated Dijkstra
│       └── tests/
│           └── test_apsp.py       # APSP test suite
└── tests/
    └── test_shortest_path.py      # 21 tests, 7 test classes
```

### Instance API

- `ShortestPathInstance.from_edges(n, edges, name)` — create from edge list
- `ShortestPathInstance.from_matrix(matrix, name)` — create from adjacency matrix (inf = no edge)
- `ShortestPathInstance.random(n, density, weight_range, seed)` — random directed graph
- `has_negative_weights()` — check for negative edge weights
- `validate_solution(instance, solution)` — verify path connectivity and distance

### Benchmark Instances

| Instance | Nodes | Edges | Negative Weights | Optimal 0-4 |
|----------|-------|-------|------------------|-------------|
| `simple_graph_5` | 5 | 7 | No | 7.0 (0-1-3-4) |
| `negative_weight_graph` | 5 | 6 | Yes | 1.0 (0-1-2-3-4) |

---

## 7. Applications

- **Road network routing:** Dijkstra / A* for GPS navigation systems
- **Internet routing:** Bellman-Ford in distance-vector protocols (RIP)
- **Link-state routing:** Dijkstra in OSPF and IS-IS protocols
- **Project scheduling:** Longest path in DAGs for critical path method (CPM/PERT)
- **Arbitrage detection:** Negative cycle detection on log-transformed exchange rates
- **Differential constraints:** Bellman-Ford for systems of difference constraints
- **Network flow:** Successive shortest path algorithm for min-cost flow
- **Social networks:** BFS for unweighted shortest paths and network diameter

---

## 8. Key References

1. Dijkstra, E.W. (1959). A note on two problems in connexion with graphs.
   *Numerische Mathematik*, 1(1), 269-271.
   [doi:10.1007/BF01386390](https://doi.org/10.1007/BF01386390)

2. Bellman, R. (1958). On a routing problem. *Quarterly of Applied
   Mathematics*, 16(1), 87-90.
   [doi:10.1090/qam/102435](https://doi.org/10.1090/qam/102435)

3. Ford, L.R. (1956). Network flow theory. Paper P-923, RAND Corporation,
   Santa Monica, CA.

4. Floyd, R.W. (1962). Algorithm 97: Shortest path. *Communications of
   the ACM*, 5(6), 345.
   [doi:10.1145/367766.368168](https://doi.org/10.1145/367766.368168)

5. Warshall, S. (1962). A theorem on boolean matrices. *Journal of the
   ACM*, 9(1), 11-12.
   [doi:10.1145/321105.321107](https://doi.org/10.1145/321105.321107)

6. Johnson, D.B. (1977). Efficient algorithms for shortest paths in sparse
   networks. *Journal of the ACM*, 24(1), 1-13.
   [doi:10.1145/321992.321993](https://doi.org/10.1145/321992.321993)

7. Hart, P.E., Nilsson, N.J. & Raphael, B. (1968). A formal basis for
   the heuristic determination of minimum cost paths. *IEEE Transactions
   on Systems Science and Cybernetics*, 4(2), 100-107.
   [doi:10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136)

8. Fredman, M.L. & Tarjan, R.E. (1987). Fibonacci heaps and their uses in
   improved network optimization algorithms. *Journal of the ACM*, 34(3),
   596-615.
   [doi:10.1145/28869.28874](https://doi.org/10.1145/28869.28874)

9. Ford, L.R. & Fulkerson, D.R. (1962). *Flows in Networks*. Princeton
   University Press.

10. Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2009).
    *Introduction to Algorithms* (3rd ed.). MIT Press. Chapters 22-25
    cover BFS, Dijkstra, Bellman-Ford, Floyd-Warshall, and Johnson's
    algorithm.

11. Schrijver, A. (2003). *Combinatorial Optimization: Polyhedra and
    Efficiency*. Springer. Total unimodularity and LP integrality of
    shortest path formulations.

12. Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). *Network Flows:
    Theory, Algorithms, and Applications*. Prentice Hall. Comprehensive
    treatment of shortest path algorithms in the network flow framework.

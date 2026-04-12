# Linear Assignment Problem (LAP)

## 1. Problem Definition

- **Input:** $n \times n$ cost matrix $C$ where $c_{ij}$ is the cost of assigning agent $i$ to task $j$
- **Decision:** Find a one-to-one assignment $\sigma: \{1,\ldots,n\} \to \{1,\ldots,n\}$
- **Objective:** Minimize total cost $\sum_{i=1}^{n} c_{i,\sigma(i)}$
- **Constraints:** Each agent assigned to exactly one task, each task assigned to exactly one agent
- **Classification:** **Polynomial** — $O(n^3)$ via Hungarian algorithm

---

## 2. Mathematical Formulation

$$\min \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}$$

$$\sum_{j=1}^{n} x_{ij} = 1 \quad \forall i \quad \text{(each agent assigned once)}$$

$$\sum_{i=1}^{n} x_{ij} = 1 \quad \forall j \quad \text{(each task assigned once)}$$

$$x_{ij} \in \{0,1\}$$

The constraint matrix is **totally unimodular**, so LP relaxation always yields integer solutions.

---

### Small Illustrative Instance

```
n = 3, cost matrix C:
     Task 1  Task 2  Task 3
A1 [  8       4       2  ]
A2 [  4       2       6  ]
A3 [  6       8       4  ]

Hungarian steps:
  Row reduction: subtract row mins → [[6,2,0],[2,0,4],[2,4,0]]
  Col reduction: subtract col mins → [[4,2,0],[0,0,4],[0,4,0]]
  Cover zeros with 2 lines → not enough, adjust
  Final optimal: A1→T3(2), A2→T2(2), A3→T1(6) = 10
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Hungarian (Kuhn-Munkres) | Exact | $O(n^3)$ | Shortest augmenting path, optimal |
| Auction algorithm (Bertsekas) | Exact | $O(n^3)$ worst | Competitive bidding, parallelizable |
| Greedy | Heuristic | $O(n^2)$ | Assign cheapest available, no guarantee |

### Hungarian Algorithm Overview

The Hungarian method works in two phases:

1. **Dual feasibility:** Initialize dual variables $u_i, v_j$ such that $u_i + v_j \leq c_{ij}$ for all $i, j$. Start with row and column reductions.

2. **Primal-dual iteration:** On the equality subgraph (edges where $u_i + v_j = c_{ij}$), find a maximum matching. If perfect, stop (optimal by complementary slackness). Otherwise, adjust duals: decrease $u_i$ for unmatched rows, increase $v_j$ for matched columns, to introduce new equality edges. Repeat.

The algorithm maintains the invariant that the matching size increases by at least one in each major iteration, giving at most $n$ phases of $O(n^2)$ work each.

### Total Unimodularity

The constraint matrix of the LAP formulation is totally unimodular (TU). This guarantees that every vertex of the LP relaxation polyhedron is integral, so the LP relaxation always yields an integer optimum. This is why LP solvers also find optimal LAP solutions.

### Rectangular and Bottleneck Variants

- **Rectangular LAP:** When the cost matrix is $m \times n$ with $m \neq n$, pad with dummy rows or columns of zero (or large) cost to make it square, then apply the standard Hungarian method.
- **Bottleneck assignment:** Minimize the maximum assignment cost $\min_\sigma \max_i c_{i,\sigma(i)}$. Solvable in $O(n^{2.5})$ via threshold-based binary search combined with maximum matching.
- **$k$-cardinality assignment:** Select exactly $k < n$ agent-task pairs to minimize cost. Solvable in $O(kn^2)$.

### Applications

- **Worker-task assignment** (minimize total cost of allocating employees to jobs)
- **Sensor-target assignment** (military and surveillance applications)
- **Object tracking** (frame-to-frame data association in computer vision)
- **Scheduling** (assigning jobs to time slots with processing-dependent costs)
- **Frequency allocation** (assigning channels to base stations at minimum interference cost)

### Relationship to Bipartite Matching

The LAP is equivalent to finding a minimum-weight perfect matching in a complete bipartite graph $K_{n,n}$ where left vertices represent agents, right vertices represent tasks, and edge weight $w(i,j) = c_{ij}$. The Hungarian method exploits this bipartite structure. For maximization variants (e.g., maximum weight assignment), negate the cost matrix or subtract from a large constant.

---

## 4. Implementation

> This problem is implemented in the parent `assignment/` folder.

See [`../assignment/`](../assignment/) — Hungarian algorithm (`exact/hungarian.py`), greedy heuristic (`heuristics/greedy_assignment.py`), 17-test suite.

---

## 5. Key References

- Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics*, 2(1-2), 83-97.
- Munkres, J. (1957). Algorithms for the assignment and transportation problems. *J. SIAM*, 5(1), 32-38.
- Bertsekas, D.P. (1988). The auction algorithm: a distributed relaxation method for the assignment problem. *Annals of Oper. Res.*, 14(1), 105-123.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). *Assignment Problems*. SIAM.
- Jonker, R. & Volgenant, A. (1987). A shortest augmenting path algorithm for dense and sparse linear assignment problems. *Computing*, 38(4), 325-340.

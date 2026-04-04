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

## 3. Solution Methods

| Method | Complexity | Description |
|--------|-----------|-------------|
| Hungarian (Kuhn-Munkres) | $O(n^3)$ | Shortest augmenting path, optimal |
| Auction algorithm (Bertsekas) | $O(n^3)$ worst | Competitive bidding, parallelizable |
| Greedy | $O(n^2)$ | Assign cheapest available, no guarantee |

---

## 4. Implementation

> This problem is implemented in the parent `assignment/` folder.

See [`../assignment/`](../assignment/) — Hungarian algorithm (`exact/hungarian.py`), greedy heuristic (`heuristics/greedy_assignment.py`), 17-test suite.

---

## 5. Key References

- Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics*, 2(1-2), 83-97.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). *Assignment Problems*. SIAM.

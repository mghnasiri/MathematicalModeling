# Fixed-Charge Network Design Problem (FCNDP)

## 1. Problem Definition

- **Input:** Potential edges with fixed opening costs and per-unit flow costs; nodes with supply/demand
- **Decision:** Which edges to open; flow routing
- **Objective:** Minimize total cost (fixed edge costs + variable flow costs)
- **Constraints:** Flow conservation; only open edges carry flow
- **Classification:** NP-hard

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy open | Heuristic | $O(|E|^2 \cdot |V|)$ | Greedily open edges with best cost-to-flow ratio |
| ILP formulation | Exact | Exponential worst | Binary edge variables + continuous flow via MILP |
| Lagrangian relaxation | Heuristic | $O(|E| \cdot |V| \cdot T)$ | Relax flow conservation, solve knapsack-like subproblems |

### Mathematical Formulation

$$\min \sum_{e \in E} f_e \cdot y_e + \sum_{e \in E} c_e \cdot x_e$$

where $y_e \in \{0,1\}$ indicates if edge $e$ is opened (fixed cost $f_e$) and
$x_e \geq 0$ is the flow on edge $e$ (variable cost $c_e$ per unit).

**Subject to:**
- Flow conservation: $\sum_{e \in \delta^+(v)} x_e - \sum_{e \in \delta^-(v)} x_e = b_v$ for all nodes $v$
- Capacity linking: $x_e \leq M \cdot y_e$ (flow only on open edges)
- $y_e \in \{0,1\}$, $x_e \geq 0$

### Greedy Edge Opening Pseudocode

```
GREEDY-NETWORK-DESIGN(nodes, edges, demands):
    open_edges <- {}
    total_cost <- 0

    // Compute benefit of each edge: flow savings minus fixed cost
    for each edge e = (u, v) with fixed cost f_e, variable cost c_e:
        benefit[e] <- estimate_flow_savings(e, demands) - f_e

    Sort edges by benefit / f_e descending

    for each edge e in sorted order:
        // Check if opening e reduces total cost
        open_edges_candidate <- open_edges + {e}

        // Route all demands on current open network
        routing_cost <- route_demands(open_edges_candidate, demands)

        if routing_cost + f_e + total_fixed < previous_total:
            open_edges <- open_edges_candidate
            total_cost <- routing_cost + sum of fixed costs

    return open_edges, total_cost
```

At each step, the algorithm evaluates whether opening an additional edge reduces
total cost (fixed + variable) enough to justify the fixed investment. Demand
routing uses shortest paths on the currently open subgraph.

---

## 3. Illustrative Instance

Consider 4 nodes with supply/demand: node 1 supplies 10 units, node 4 demands 10.

| Edge | Fixed Cost | Variable Cost/unit | Capacity |
|------|------------|--------------------|----------|
| 1->2 | 5 | 1 | 15 |
| 1->3 | 8 | 1 | 15 |
| 2->4 | 5 | 2 | 15 |
| 3->4 | 3 | 1 | 15 |
| 2->3 | 2 | 1 | 10 |

Path 1->2->4: fixed = 5+5 = 10, variable = (1+2)*10 = 30, total = 40.
Path 1->3->4: fixed = 8+3 = 11, variable = (1+1)*10 = 20, total = 31.
Path 1->2->3->4: fixed = 5+2+3 = 10, variable = (1+1+1)*10 = 30, total = 40.

Greedy would open 1->3 and 3->4 first (total 31), which is optimal here.

---

## 4. Implementations in This Repository

```
network_design/
├── instance.py                    # NetworkDesignInstance, NetworkDesignSolution
├── heuristics/
│   └── greedy_open.py             # Greedy edge opening
└── tests/
    └── test_network_design.py     # Network design test suite
```

---

## 5. Key References

- Magnanti, T.L. & Wong, R.T. (1984). Network design and transportation planning: Models and algorithms. *Transp. Sci.*, 18(1), 1-55.
- Balakrishnan, A., Magnanti, T.L. & Mirchandani, P. (1997). Network design. In *Annotated Bibliographies in Combinatorial Optimization*, Wiley, 311-334.
- Gendron, B., Crainic, T.G. & Frangioni, A. (1999). Multicommodity capacitated network design. In *Telecommunications Network Planning*, Springer, 1-19.

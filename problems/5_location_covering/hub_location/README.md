# Hub Location Problem (p-Hub Median)

## 1. Problem Definition

- **Input:** $n$ nodes, flow matrix $W_{ij}$, distance matrix $D_{ij}$, number of hubs $p$, inter-hub discount factor $\alpha$
- **Decision:** Select $p$ hub nodes; assign each non-hub to a hub (single allocation)
- **Objective:** Minimize total transportation cost (flows routed: origin hub → destination hub with discount $\alpha$)
- **Classification:** NP-hard (O'Kelly, 1987)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy hub | Heuristic | $O(n^2 \cdot p)$ | Greedily open hubs minimizing total cost |
| Enumeration | Exact | $O(\binom{n}{p} \cdot n^2)$ | Evaluate all $p$-subsets (feasible for small $n$) |
| SA hub | Metaheuristic | Problem-dependent | Swap hubs with Boltzmann acceptance |

### Single-Allocation Cost Model

For single allocation, every non-hub node $i$ is assigned to exactly one hub $h(i)$.
A flow of $W_{ij}$ units from node $i$ to node $j$ travels:
origin $i$ to hub $h(i)$, then hub $h(i)$ to hub $h(j)$ at discounted cost $\alpha$,
then hub $h(j)$ to destination $j$. The total cost is:

$$\sum_{i}\sum_{j} W_{ij} \bigl[ D_{i,h(i)} + \alpha \cdot D_{h(i),h(j)} + D_{h(j),j} \bigr]$$

### Enumeration Pseudocode

For small instances, enumerate all possible hub sets:

```
ENUMERATE-HUBS(nodes, p, W, D, alpha):
    best_cost <- infinity
    best_hubs <- None

    for each p-subset H of nodes:
        // Assign each node to its nearest hub
        cost <- 0
        for each node i:
            h_i <- argmin_{h in H} D[i][h]
            for each node j:
                h_j <- argmin_{h in H} D[j][h]
                cost <- cost + W[i][j] * (D[i][h_i] + alpha * D[h_i][h_j] + D[h_j][j])
        if cost < best_cost:
            best_cost <- cost
            best_hubs <- H

    return best_hubs, best_cost
```

Practical for $\binom{n}{p}$ small (e.g., $n \leq 20$, $p \leq 4$).

---

## 3. Illustrative Instance

Consider $n = 4$ nodes, $p = 2$ hubs, $\alpha = 0.5$:

Flow matrix $W$:
```
  1  2  3  4
1 [0  5  2  1]
2 [3  0  4  2]
3 [1  3  0  6]
4 [2  1  5  0]
```

Distance matrix $D$:
```
  1  2  3  4
1 [0  3  5  7]
2 [3  0  4  6]
3 [5  4  0  3]
4 [7  6  3  0]
```

With hubs {1, 3}: node 2 assigned to hub 1 (dist 3), node 4 assigned to hub 3 (dist 3).
Inter-hub distance $D[1][3] = 5$, discounted to $0.5 \times 5 = 2.5$.

---

## 4. Implementations in This Repository

```
hub_location/
├── instance.py                    # HubLocationInstance, HubLocationSolution
├── heuristics/
│   └── greedy_hub.py              # Greedy hub selection
└── tests/
    └── test_hub_location.py       # Hub location test suite
```

---

## 5. Key References

- O'Kelly, M.E. (1987). A quadratic integer program for the location of interacting hub facilities. *European J. Oper. Res.*, 32(3), 393-404.
- Campbell, J.F. (1994). Integer programming formulations of discrete hub location problems. *European J. Oper. Res.*, 72(2), 387-405.
- Ernst, A.T. & Krishnamoorthy, M. (1996). Efficient algorithms for the uncapacitated single allocation p-hub median problem. *Location Science*, 4(3), 139-154.

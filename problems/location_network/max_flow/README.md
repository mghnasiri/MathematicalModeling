# Maximum Flow Problem (Max-Flow)

## Problem Definition

Given a directed graph $G = (V, E)$ with edge capacities $c(u,v)$, a source $s$ and sink $t$, find the maximum flow from $s$ to $t$ respecting capacity constraints and flow conservation.

The **Max-Flow Min-Cut Theorem** (Ford & Fulkerson, 1956) states that the maximum flow equals the minimum $s$-$t$ cut capacity.

## Complexity

| Algorithm | Complexity | Description |
|-----------|-----------|-------------|
| Edmonds-Karp | $O(VE^2)$ | BFS augmenting paths |
| Push-Relabel | $O(V^2 E)$ | Preflow with relabeling |
| Dinic's | $O(V^2 E)$ | Blocking flows in layered graph |

## Key References

- Ford, L.R. & Fulkerson, D.R. (1956). Maximal flow through a network. *Canadian J. Math.*, 8, 399-404. https://doi.org/10.4153/CJM-1956-045-5
- Edmonds, J. & Karp, R.M. (1972). Theoretical improvements in network flow algorithms. *JACM*, 19(2), 248-264. https://doi.org/10.1145/321694.321699
- Goldberg, A.V. & Tarjan, R.E. (1988). A new approach to maximum flow. *JACM*, 35(4), 921-940. https://doi.org/10.1145/48014.61051

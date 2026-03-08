# Shortest Path Problem (SPP)

## Problem Definition

Given a directed graph $G = (V, E)$ with edge weights $w(u,v)$, find the path from source $s$ to target $t$ with minimum total weight.

## Complexity

| Variant | Complexity | Algorithm |
|---------|-----------|-----------|
| Non-negative weights | $O((V+E) \log V)$ | Dijkstra (1959) |
| Arbitrary weights | $O(VE)$ | Bellman-Ford (1958) |
| DAG | $O(V+E)$ | Topological sort + relaxation |
| All-pairs | $O(V^3)$ | Floyd-Warshall |

## Implementations

| Algorithm | Handles Negative | Detects Neg. Cycles | Description |
|-----------|:---:|:---:|-------------|
| Dijkstra | No | — | Binary heap priority queue |
| Bellman-Ford | Yes | Yes | V-1 edge relaxation rounds |

## Key References

- Dijkstra, E.W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271. https://doi.org/10.1007/BF01386390
- Bellman, R. (1958). On a routing problem. *Q. Appl. Math.*, 16(1), 87-90. https://doi.org/10.1090/qam/102435
- Fredman, M.L. & Tarjan, R.E. (1987). Fibonacci heaps and network optimization. *JACM*, 34(3), 596-615. https://doi.org/10.1145/28869.28874

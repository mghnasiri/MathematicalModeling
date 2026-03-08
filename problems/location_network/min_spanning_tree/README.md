# Minimum Spanning Tree (MST)

## Problem Definition

Given an undirected, weighted, connected graph $G = (V, E)$, find the spanning tree $T$ of minimum total edge weight — a connected acyclic subgraph spanning all $V$ vertices with exactly $|V| - 1$ edges.

## Complexity

| Algorithm | Complexity | Approach |
|-----------|-----------|----------|
| Kruskal's | $O(E \log E)$ | Sort edges, union-find cycle detection |
| Prim's | $O(E \log V)$ | Grow tree from root, binary heap |
| Borůvka's | $O(E \log V)$ | Parallel edge contraction |

## Key References

- Kruskal, J.B. (1956). On the shortest spanning subtree. *Proc. AMS*, 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
- Prim, R.C. (1957). Shortest connection networks. *Bell System Tech. J.*, 36(6), 1389-1401. https://doi.org/10.1002/j.1538-7305.1957.tb01515.x
- Cormen, T.H. et al. (2009). *Introduction to Algorithms*, 3rd ed. MIT Press, Ch. 23.

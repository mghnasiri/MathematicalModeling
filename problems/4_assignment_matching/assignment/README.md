# Linear Assignment Problem (LAP)

## Problem Definition

Given an $n \times n$ cost matrix $C$, find a one-to-one assignment of agents to tasks minimizing total cost:

$$\min_{\sigma \in S_n} \sum_{i=1}^{n} c_{i, \sigma(i)}$$

where $\sigma$ is a permutation (bijection from agents to tasks).

## Complexity

| Algorithm | Complexity | Description |
|-----------|-----------|-------------|
| Hungarian (Kuhn-Munkres) | $O(n^3)$ | Dual potentials + augmenting paths |
| Auction | $O(n^3)$ avg | Parallel-friendly, Bertsekas (1979) |
| Greedy | $O(n^2)$ | Not optimal, fast upper bound |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Generalized Assignment (GAP)](variants/generalized/) | `variants/generalized/` | Multiple items per agent with resource constraints |
| [Quadratic Assignment (QAP)](variants/quadratic/) | `variants/quadratic/` | Minimize pairwise interaction costs based on facility-location assignment |
| [Maximum Weight Matching](variants/max_weight_matching/) | `variants/max_weight_matching/` | Maximize total weight of matched pairs in a bipartite graph |

## Key References

- Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Res. Logist.*, 2(1-2), 83-97. https://doi.org/10.1002/nav.3800020109
- Munkres, J. (1957). Algorithms for the assignment and transportation problems. *SIAM J.*, 5(1), 32-38. https://doi.org/10.1137/0105003
- Jonker, R. & Volgenant, A. (1987). A shortest augmenting path algorithm for LAP. *Computing*, 38(4), 325-340. https://doi.org/10.1007/BF02278710

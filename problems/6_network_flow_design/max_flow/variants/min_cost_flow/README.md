# Minimum Cost Flow Problem (Max Flow Variant)

## What Changes

In the **minimum cost flow** variant, each edge has both a capacity u(e) and a
per-unit transportation cost a(e). Vertices have supply/demand values b(v):
positive for sources (supply), negative for sinks (demand), zero for
transshipment nodes. The objective is to find a flow satisfying all
supply/demand balance constraints that minimizes total cost. The base max
flow problem maximizes throughput from s to t without cost considerations;
min-cost flow optimizes the cost of routing a specified amount of flow.

This is one of the most versatile network optimization models, generalizing
shortest path (unit supply at s, unit demand at t), max flow (add cost edge
from t to s), assignment (bipartite unit-capacity network), and transportation
problems. Applications include supply chain logistics, telecommunications
routing, and production planning.

## Mathematical Formulation

The base max flow formulation adds per-edge costs and generalizes to
multiple sources/sinks:

```
min  sum_{(i,j) in E}  a_ij * f_ij
s.t. sum_j  f_ij  -  sum_j  f_ji  =  b_i    for all i in V  (flow balance)
     0 <= f_ij <= u_ij                        for all (i,j) in E  (capacity)
```

where b_i > 0 for supply nodes, b_i < 0 for demand nodes, and
sum_i b_i = 0 (total supply equals total demand).

**Relationship to max flow:** Set b_s = F (total flow), b_t = -F, all other
b_i = 0, and minimize cost of pushing F units from s to t.

## Complexity

| Algorithm | Time | Reference |
|-----------|------|-----------|
| Successive Shortest Paths | O(V^2 * E) | Ahuja et al. (1993) |
| Cycle-Canceling | O(V * E^2 * C * U) | Klein (1967) |
| Cost Scaling | O(V^3 * log(VC)) | Goldberg & Tarjan (1990) |

Polynomial. The successive shortest paths algorithm repeatedly finds
minimum-cost augmenting paths using Bellman-Ford on the residual graph,
sending flow along each path until all supply is routed.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Edmonds-Karp (base max flow) | No | Maximizes flow, ignores costs |
| Successive Shortest Paths (variant) | Yes | Exact, O(V^2 * E) |
| Cycle-Canceling | Possible | Find negative-cost cycles in residual graph |

## Implementations

Python files in this directory:
- `instance.py` -- MinCostFlowInstance, edge costs, supply/demand vectors
- `heuristics.py` -- Successive Shortest Paths (exact, Bellman-Ford based)
- `tests/test_mcf.py` -- 15 tests

## Applications

- Supply chain logistics (route goods from factories to warehouses cheaply)
- Telecommunications routing (route traffic at minimum cost)
- Transportation planning (assign shipments to routes)
- Production planning (allocate resources across time periods)

## Key References

- Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). Network Flows: Theory, Algorithms, and Applications. Prentice Hall.
- Goldberg, A.V. & Tarjan, R.E. (1990). "Finding minimum-cost circulations by successive approximation." Mathematics of Operations Research 15(3), 430-466.
- Klein, M. (1967). "A primal method for minimal cost flows." Management Science 14(3), 205-220.

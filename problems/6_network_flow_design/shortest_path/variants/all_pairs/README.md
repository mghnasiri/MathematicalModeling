# All-Pairs Shortest Path (Shortest Path Variant)

## What Changes

In the **all-pairs shortest path** variant (APSP), the goal is to compute the
shortest path distance between every pair of vertices, not just from a single
source to a single target. The output is an n x n distance matrix D where
D[i][j] is the shortest path from vertex i to vertex j, plus optionally a
next-hop matrix for path reconstruction. This models scenarios requiring
complete distance information -- routing tables in network protocols, distance
matrices for location-based optimization, and transitive closure computation
in graph analysis.

The key structural difference from the single-source problem is that
running Dijkstra or Bellman-Ford from each vertex independently is
O(V * single-source), but Floyd-Warshall exploits the all-pairs structure
with a simpler O(V^3) dynamic programming approach that also handles
negative weights.

## Mathematical Formulation

The base shortest path formulation is replicated for all (s, t) pairs:

**Distance matrix:**
```
D[i][j] = min  sum_{(u,v) in P}  w(u, v)
           over all paths P from i to j
```

**Floyd-Warshall recurrence:**
```
D_k[i][j] = min(D_{k-1}[i][j],  D_{k-1}[i][k] + D_{k-1}[k][j])
```
where D_k[i][j] is the shortest i-to-j path using only intermediate vertices
{1, ..., k}. Initialize D_0[i][j] = w(i,j) if edge exists, infinity otherwise.

**Negative cycle detection:** If D[i][i] < 0 for any i after completion,
the graph contains a negative cycle.

## Complexity

| Algorithm | Time | Space | Negative Weights? | Reference |
|-----------|------|-------|-------------------|-----------|
| Floyd-Warshall | O(V^3) | O(V^2) | Yes (no neg cycles) | Floyd (1962) |
| Repeated Dijkstra | O(V(V+E) log V) | O(V^2) | No | Dijkstra (1959) |
| Johnson's Algorithm | O(V^2 log V + VE) | O(V^2) | Yes (reweighting) | Johnson (1977) |

Floyd-Warshall is preferred for dense graphs (E close to V^2). For sparse
graphs with non-negative weights, repeated Dijkstra is faster. Johnson's
algorithm handles negative weights on sparse graphs by reweighting edges
via Bellman-Ford, then running Dijkstra from each vertex.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Dijkstra (base single-source) | Partially | Must run V times; non-negative weights only |
| Bellman-Ford (base single-source) | Partially | Must run V times; O(V^2 * E) total |
| Floyd-Warshall (variant) | Yes | O(V^3) DP, handles negative weights |
| Repeated Dijkstra (variant) | Yes | O(V(V+E) log V), faster for sparse non-negative |

## Implementations

Python files in this directory:
- `instance.py` -- APSPInstance, adjacency matrix, distance matrix output
- `heuristics.py` -- Floyd-Warshall, Repeated Dijkstra (both exact)
- `tests/test_apsp.py` -- 19 tests

## Applications

- Network routing tables (all-to-all shortest paths)
- Distance matrix computation for TSP/VRP input
- Graph diameter and centrality measures
- Transitive closure and reachability analysis

## Key References

- Floyd, R.W. (1962). "Algorithm 97: Shortest path." Communications of the ACM 5(6), 345.
- Warshall, S. (1962). "A theorem on boolean matrices." JACM 9(1), 11-12.
- Johnson, D.B. (1977). "Efficient algorithms for shortest paths in sparse networks." JACM 24(1), 1-13.

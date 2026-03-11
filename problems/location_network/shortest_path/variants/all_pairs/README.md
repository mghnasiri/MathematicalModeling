# All-Pairs Shortest Path (APSP)

## Problem Definition

Given a weighted directed graph G = (V, E), compute the shortest path distance between every pair of vertices (u, v). Supports path reconstruction via next-hop matrix.

**Objective:** Compute n x n distance matrix D where D[i][j] = shortest path from i to j.

## Complexity

- Floyd-Warshall: O(V^3), handles negative weights (no negative cycles)
- Repeated Dijkstra: O(V * (V + E) log V), non-negative weights only

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Floyd-Warshall | Exact | Floyd (1962), Warshall (1962) |
| Repeated Dijkstra | Exact | Dijkstra (1959) |

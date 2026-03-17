# Maximum Weight Matching

## Problem Definition

Given a weighted bipartite graph with n workers and m tasks, find a matching (subset of edges with no shared endpoints) that maximizes total weight. Unlike the standard LAP, not all entities need to be matched and the sets can have different sizes.

## Complexity

Polynomial — O(n^3) via Hungarian method on augmented matrix.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy Max-Weight | Heuristic | — |
| Hungarian (Kuhn-Munkres) | Exact | Kuhn (1955) |

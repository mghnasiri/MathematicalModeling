# Steiner Tree Problem in Graphs

## Problem Definition

Given an undirected weighted graph and a subset of terminal vertices, find the minimum-weight tree spanning all terminals. Non-terminal (Steiner) nodes may be included if they reduce total weight.

## Complexity

NP-hard (Karp, 1972). Polynomial for |S|=2 (shortest path) or |S|=|V| (MST).

## Algorithms

| Algorithm | Type | Approx Ratio | Reference |
|-----------|------|-------------|-----------|
| KMB | Heuristic | 2(1-1/l) | Kou, Markowsky & Berman (1981) |
| Shortest Path | Heuristic | — | — |

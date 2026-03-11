# Minimum Cost Flow Problem

## Problem Definition

Given a directed graph with edge capacities and per-unit costs, find the flow satisfying supply/demand constraints that minimizes total transportation cost. Generalizes shortest path, max flow, and assignment problems.

## Complexity

Polynomial — O(V^2 * E) via successive shortest paths.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Successive Shortest Paths | Exact | Ahuja, Magnanti & Orlin (1993) |

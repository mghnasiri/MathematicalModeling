"""
Greedy Heuristics for Graph Coloring

Two approaches:
    1. Sequential greedy: process vertices in order, assign smallest
       available color.
    2. DSatur (Brélaz, 1979): process vertex with highest saturation
       degree (number of distinct colors among neighbors).

DSatur is exact for bipartite and cycle graphs.

Complexity: O(V + E) for greedy, O(V^2) for DSatur.

References:
    - Brélaz, D. (1979). New methods to color the vertices of a graph.
      Comm. ACM, 22(4), 251-256.
    - Welsh, D.J.A. & Powell, M.B. (1967). An upper bound for the
      chromatic number. Computer J., 10(1), 85-86.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("gc_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
GraphColoringInstance = _inst.GraphColoringInstance
GraphColoringSolution = _inst.GraphColoringSolution


def greedy_sequential(instance: GraphColoringInstance,
                      order: list[int] | None = None) -> GraphColoringSolution:
    """Sequential greedy coloring.

    Args:
        instance: GraphColoringInstance.
        order: Vertex processing order. None for natural order.

    Returns:
        GraphColoringSolution.
    """
    adj = instance.adjacency_list()
    n = instance.n_vertices
    colors = [-1] * n

    if order is None:
        order = list(range(n))

    for v in order:
        neighbor_colors = {colors[u] for u in adj[v] if colors[u] >= 0}
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[v] = c

    return GraphColoringSolution(
        colors=colors,
        n_colors=len(set(colors)),
        is_valid=instance.is_valid_coloring(colors),
    )


def greedy_largest_first(instance: GraphColoringInstance) -> GraphColoringSolution:
    """Welsh-Powell: sort vertices by degree descending."""
    adj = instance.adjacency_list()
    order = sorted(range(instance.n_vertices), key=lambda v: len(adj[v]), reverse=True)
    return greedy_sequential(instance, order)


def dsatur(instance: GraphColoringInstance) -> GraphColoringSolution:
    """DSatur (Brélaz): pick vertex with highest saturation degree.

    Saturation = number of distinct colors among colored neighbors.
    Ties broken by highest uncolored degree.

    Args:
        instance: GraphColoringInstance.

    Returns:
        GraphColoringSolution.
    """
    adj = instance.adjacency_list()
    n = instance.n_vertices
    colors = [-1] * n
    saturation = [0] * n
    neighbor_colors: list[set[int]] = [set() for _ in range(n)]

    for _ in range(n):
        # Pick uncolored vertex with max saturation, break ties by degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in range(n):
            if colors[v] >= 0:
                continue
            s = saturation[v]
            d = len(adj[v])
            if s > best_sat or (s == best_sat and d > best_deg):
                best_sat = s
                best_deg = d
                best_v = v

        if best_v == -1:
            break

        # Assign smallest available color
        c = 0
        while c in neighbor_colors[best_v]:
            c += 1
        colors[best_v] = c

        # Update saturation of neighbors
        for u in adj[best_v]:
            if colors[u] < 0 and c not in neighbor_colors[u]:
                neighbor_colors[u].add(c)
                saturation[u] += 1

    return GraphColoringSolution(
        colors=colors,
        n_colors=len(set(colors)),
        is_valid=instance.is_valid_coloring(colors),
    )

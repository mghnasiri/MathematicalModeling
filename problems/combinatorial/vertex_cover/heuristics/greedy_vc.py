"""
Greedy Heuristics for Minimum Vertex Cover.

Problem: MVC (Minimum Vertex Cover)
Complexity: O(V + E) for edge-based 2-approximation

Algorithms:
1. greedy_edge_cover: Pick an uncovered edge, add both endpoints.
   Guaranteed 2-approximation (Bar-Yehuda & Even, 1981).
2. greedy_degree_cover: Iteratively add highest-degree vertex.
   No constant-factor guarantee but often better in practice.

References:
    Bar-Yehuda, R. & Even, S. (1981). A linear-time approximation
    algorithm for the weighted vertex cover problem.
    https://doi.org/10.1016/0196-6774(81)90021-7
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np


def _load_parent(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "vc_instance_greedy",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
VertexCoverInstance = _inst.VertexCoverInstance
VertexCoverSolution = _inst.VertexCoverSolution


def greedy_edge_cover(instance: VertexCoverInstance) -> VertexCoverSolution:
    """2-approximation: for each uncovered edge, add both endpoints.

    Args:
        instance: A VertexCoverInstance.

    Returns:
        VertexCoverSolution with at most 2*OPT vertices.
    """
    cover = set()
    for u, v in instance.edges:
        if u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)

    cover_list = sorted(cover)
    return VertexCoverSolution(cover=cover_list, size=len(cover_list))


def greedy_degree_cover(instance: VertexCoverInstance) -> VertexCoverSolution:
    """Greedy by degree: iteratively add vertex covering most uncovered edges.

    Args:
        instance: A VertexCoverInstance.

    Returns:
        VertexCoverSolution.
    """
    cover = set()
    uncovered = set(range(len(instance.edges)))

    while uncovered:
        # Count uncovered edges per vertex
        degree = [0] * instance.n_vertices
        for idx in uncovered:
            u, v = instance.edges[idx]
            if u not in cover:
                degree[u] += 1
            if v not in cover:
                degree[v] += 1

        # Add vertex with max uncovered-degree
        best_v = max(range(instance.n_vertices), key=lambda v: degree[v])
        cover.add(best_v)

        # Remove newly covered edges
        newly_covered = set()
        for idx in uncovered:
            u, v = instance.edges[idx]
            if u in cover or v in cover:
                newly_covered.add(idx)
        uncovered -= newly_covered

    cover_list = sorted(cover)
    return VertexCoverSolution(cover=cover_list, size=len(cover_list))


if __name__ == "__main__":
    _inst_mod = _load_parent(
        "vc_inst_main",
        os.path.join(os.path.dirname(__file__), "..", "instance.py"),
    )
    inst = _inst_mod.VertexCoverInstance.cycle(8)
    sol_e = greedy_edge_cover(inst)
    sol_d = greedy_degree_cover(inst)
    print(f"Edge-based: {sol_e}")
    print(f"Degree-based: {sol_d}")

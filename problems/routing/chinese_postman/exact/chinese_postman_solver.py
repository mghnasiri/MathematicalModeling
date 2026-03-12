"""
Chinese Postman Solver — exact algorithm for undirected CPP.

Problem: CPP (Chinese Postman Problem, undirected)
Complexity: O(V^3) — dominated by shortest paths + matching

Algorithm:
1. If the graph is Eulerian (all even-degree vertices), find an Euler tour
   with total weight = sum of all edge weights.
2. Otherwise, find the minimum weight perfect matching on odd-degree vertices
   (using shortest paths), duplicate matched edges, then find an Euler tour.

For the matching step, with k odd-degree vertices, we enumerate all perfect
matchings (feasible for small k) or use a greedy matching heuristic.

References:
    Edmonds, J. & Johnson, E.L. (1973). Matching, Euler tours and the
    Chinese postman. Mathematical Programming, 5(1), 88-124.
    https://doi.org/10.1007/BF01580113
"""

from __future__ import annotations

import os
import numpy as np
from itertools import combinations

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module(
    "cpp_instance_solver", os.path.join(_parent_dir, "instance.py")
)
ChinesePostmanInstance = _inst.ChinesePostmanInstance
ChinesePostmanSolution = _inst.ChinesePostmanSolution


def _find_euler_tour(
    n: int,
    edge_list: list[tuple[int, int, float]],
) -> list[int]:
    """Find an Euler tour in a multigraph using Hierholzer's algorithm.

    Args:
        n: Number of vertices.
        edge_list: List of (u, v, weight) tuples (may have duplicates).

    Returns:
        List of vertex indices forming an Euler circuit.
    """
    # Build adjacency list with edge indices
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for idx, (u, v, _) in enumerate(edge_list):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    used = [False] * len(edge_list)
    stack = [0]
    tour = []

    while stack:
        v = stack[-1]
        found = False
        while adj[v]:
            neighbor, edge_idx = adj[v][-1]
            adj[v].pop()
            if used[edge_idx]:
                continue
            used[edge_idx] = True
            # Also remove from neighbor's list (lazily handled by used check)
            stack.append(neighbor)
            found = True
            break
        if not found:
            tour.append(stack.pop())

    return tour


def _min_weight_perfect_matching(
    odd_vertices: list[int],
    dist_matrix: np.ndarray,
) -> list[tuple[int, int]]:
    """Find minimum weight perfect matching on odd-degree vertices.

    For small numbers of odd vertices, enumerate all perfect matchings.
    For larger sets, use a greedy heuristic.

    Args:
        odd_vertices: List of odd-degree vertex indices.
        dist_matrix: All-pairs shortest path distances.

    Returns:
        List of (u, v) pairs in the matching.
    """
    k = len(odd_vertices)
    if k == 0:
        return []
    if k == 2:
        return [(odd_vertices[0], odd_vertices[1])]

    # For small k, enumerate all perfect matchings
    if k <= 12:
        return _exact_matching(odd_vertices, dist_matrix)

    # Greedy matching for larger instances
    return _greedy_matching(odd_vertices, dist_matrix)


def _exact_matching(
    vertices: list[int],
    dist_matrix: np.ndarray,
) -> list[tuple[int, int]]:
    """Enumerate all perfect matchings to find minimum weight one."""
    if len(vertices) == 0:
        return []
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]

    best_cost = float("inf")
    best_matching: list[tuple[int, int]] = []

    # Fix first vertex, try pairing with each other
    first = vertices[0]
    rest = vertices[1:]

    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i + 1:]
        sub_matching = _exact_matching(remaining, dist_matrix)
        cost = dist_matrix[first][partner] + sum(
            dist_matrix[u][v] for u, v in sub_matching
        )
        if cost < best_cost:
            best_cost = cost
            best_matching = [(first, partner)] + sub_matching

    return best_matching


def _greedy_matching(
    vertices: list[int],
    dist_matrix: np.ndarray,
) -> list[tuple[int, int]]:
    """Greedy minimum weight matching: iteratively pair closest vertices."""
    remaining = list(vertices)
    matching = []

    while len(remaining) >= 2:
        best_cost = float("inf")
        best_i = -1
        best_j = -1
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                cost = dist_matrix[remaining[i]][remaining[j]]
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_j = j
        matching.append((remaining[best_i], remaining[best_j]))
        # Remove in reverse order to preserve indices
        remaining.pop(best_j)
        remaining.pop(best_i)

    return matching


def _shortest_path_edges(
    u: int, v: int, dist_matrix: np.ndarray, adj_matrix: np.ndarray
) -> list[tuple[int, int, float]]:
    """Reconstruct shortest path edges between u and v.

    Uses the adjacency matrix and shortest path distances to reconstruct
    the path by greedy forwarding.

    Args:
        u: Start vertex.
        v: End vertex.
        dist_matrix: All-pairs shortest path distances.
        adj_matrix: Original adjacency/weight matrix.

    Returns:
        List of (u, v, weight) edges along the shortest path.
    """
    n = len(dist_matrix)
    path_edges = []
    current = u

    while current != v:
        best_next = -1
        best_cost = float("inf")
        for neighbor in range(n):
            if adj_matrix[current][neighbor] > 0:
                cost = adj_matrix[current][neighbor] + dist_matrix[neighbor][v]
                if cost < best_cost:
                    best_cost = cost
                    best_next = neighbor
        if best_next < 0:
            break
        w = adj_matrix[current][best_next]
        path_edges.append((current, best_next, w))
        current = best_next

    return path_edges


def chinese_postman(
    instance: ChinesePostmanInstance,
) -> ChinesePostmanSolution:
    """Solve the Chinese Postman Problem.

    If the graph is Eulerian, directly find an Euler tour.
    Otherwise, find minimum weight matching on odd-degree vertices,
    duplicate matched shortest-path edges, then find Euler tour.

    Args:
        instance: A ChinesePostmanInstance.

    Returns:
        ChinesePostmanSolution with optimal (or near-optimal) tour.
    """
    if not instance.is_connected():
        raise ValueError("Graph must be connected for Chinese Postman")

    odd_verts = instance.odd_degree_vertices()

    if not odd_verts:
        # Eulerian graph — find Euler tour directly
        tour = _find_euler_tour(instance.n_vertices, instance.edges)
        return ChinesePostmanSolution(
            tour=tour,
            total_weight=instance.total_edge_weight(),
            duplicated_edges=[],
        )

    # Non-Eulerian: find minimum weight matching on odd-degree vertices
    dist_matrix = instance.shortest_paths()
    matching = _min_weight_perfect_matching(odd_verts, dist_matrix)

    # Duplicate edges along shortest paths for matched pairs
    duplicated_edges = []
    for u, v in matching:
        path_edges = _shortest_path_edges(
            u, v, dist_matrix, instance.adj_matrix
        )
        duplicated_edges.extend(path_edges)

    # Build augmented edge list
    all_edges = list(instance.edges) + duplicated_edges

    # Find Euler tour on augmented graph
    tour = _find_euler_tour(instance.n_vertices, all_edges)

    total_weight = instance.total_edge_weight() + sum(
        w for _, _, w in duplicated_edges
    )

    return ChinesePostmanSolution(
        tour=tour,
        total_weight=total_weight,
        duplicated_edges=duplicated_edges,
    )


if __name__ == "__main__":
    from instance import eulerian_square, bridge_graph, non_eulerian_triangle

    print("=== Chinese Postman Solver ===\n")

    for name, inst_fn in [
        ("eulerian_square", eulerian_square),
        ("triangle_k3", non_eulerian_triangle),
        ("bridge_graph", bridge_graph),
    ]:
        inst = inst_fn()
        sol = chinese_postman(inst)
        print(f"{name}: {sol}")
        print(f"  Tour: {sol.tour}")

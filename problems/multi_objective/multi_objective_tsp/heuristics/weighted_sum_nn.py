"""Weighted-Sum Nearest Neighbor for Multi-Objective TSP.

Combines multiple distance matrices using weighted sums and applies
nearest-neighbor heuristic. Varies weights to approximate the Pareto front.

Also provides a Pareto-filtering utility to extract non-dominated solutions.

Complexity: O(W * n^2) where W = number of weight vectors.

References:
    Jaszkiewicz, A. (2002). Genetic local search for multi-objective
    combinatorial optimization. European Journal of Operational Research,
    137(1), 50-71.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np
from itertools import product


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "motsp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MultiObjectiveTSPInstance = _inst.MultiObjectiveTSPInstance
MultiObjectiveTSPSolution = _inst.MultiObjectiveTSPSolution


def _nearest_neighbor(dist_matrix: np.ndarray, start: int = 0) -> list[int]:
    """Nearest-neighbor TSP heuristic on a single distance matrix.

    Args:
        dist_matrix: n x n distance matrix.
        start: Starting city.

    Returns:
        Tour as a list of city indices.
    """
    n = dist_matrix.shape[0]
    visited = {start}
    tour = [start]
    current = start

    for _ in range(n - 1):
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if j not in visited and dist_matrix[current, j] < best_dist:
                best_dist = dist_matrix[current, j]
                best_next = j
        tour.append(best_next)
        visited.add(best_next)
        current = best_next

    return tour


def _filter_pareto(points: list[tuple[float, ...]],
                   tours: list[list[int]]
                   ) -> tuple[list[tuple[float, ...]], list[list[int]]]:
    """Filter to non-dominated solutions, removing duplicates."""
    filtered_pts = []
    filtered_tours = []

    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i != j:
                if all(p2[k] <= p1[k] for k in range(len(p1))) and \
                   any(p2[k] < p1[k] for k in range(len(p1))):
                    dominated = True
                    break
        if not dominated:
            # Check duplicate
            is_dup = False
            for fp in filtered_pts:
                if all(abs(fp[k] - p1[k]) < 1e-9 for k in range(len(p1))):
                    is_dup = True
                    break
            if not is_dup:
                filtered_pts.append(p1)
                filtered_tours.append(tours[i])

    return filtered_pts, filtered_tours


def weighted_sum_nn(instance: MultiObjectiveTSPInstance,
                    n_weights: int = 11) -> MultiObjectiveTSPSolution:
    """Weighted-sum nearest neighbor for multi-objective TSP.

    For each weight vector, combines the distance matrices and applies
    nearest-neighbor heuristic. Filters non-dominated solutions.

    Args:
        instance: A MultiObjectiveTSPInstance.
        n_weights: Number of weight grid points per objective pair.

    Returns:
        A MultiObjectiveTSPSolution with approximate Pareto front.
    """
    all_points = []
    all_tours = []

    if instance.n_objectives == 2:
        weights_list = [(a, 1 - a) for a in np.linspace(0, 1, n_weights)]
    else:
        # Generate weight vectors on simplex
        steps = np.linspace(0, 1, n_weights)
        weights_list = []
        for combo in product(steps, repeat=instance.n_objectives - 1):
            if sum(combo) <= 1.0:
                w = list(combo) + [1.0 - sum(combo)]
                weights_list.append(tuple(w))

    for weights in weights_list:
        # Combine distance matrices
        combined = sum(w * D for w, D in zip(weights,
                       instance.distance_matrices))

        # Multi-start NN
        for start in range(min(instance.n, 5)):
            tour = _nearest_neighbor(combined, start=start)
            costs = instance.evaluate(tour)
            all_points.append(costs)
            all_tours.append(tour)

    front, tours = _filter_pareto(all_points, all_tours)

    # Sort by first objective
    paired = sorted(zip(front, tours), key=lambda x: x[0][0])
    front = [p[0] for p in paired]
    tours = [p[1] for p in paired]

    return MultiObjectiveTSPSolution(
        pareto_front=front, pareto_tours=tours, n_solutions=len(front)
    )


if __name__ == "__main__":
    inst = MultiObjectiveTSPInstance.random(n=8, n_objectives=2, seed=42)
    print(f"Instance: {inst.n} cities, {inst.n_objectives} objectives")

    sol = weighted_sum_nn(inst, n_weights=11)
    print(f"\nWeighted-sum NN: {sol.n_solutions} Pareto solutions")
    for pt in sol.pareto_front:
        print(f"  costs={tuple(f'{c:.1f}' for c in pt)}")

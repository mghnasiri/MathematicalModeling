"""
Variable Neighborhood Search for TSP.

Problem: TSP (Traveling Salesman Problem)

VNS uses multiple neighborhood structures:
    N1: 2-opt — reverse a random segment
    N2: Or-opt — relocate a random segment of 1-3 cities
    N3: Double-bridge — 4-edge perturbation (strong diversification)

Local search uses best-improvement 2-opt.
Warm-started with nearest neighbor multi-start.

Complexity: O(iterations * k_max * n^2) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P., Mladenović, N. & Pérez, J.A.M. (2010). Variable
    neighbourhood search: methods and applications. Annals of Operations
    Research, 175(1), 367-407.
    https://doi.org/10.1007/s10479-009-0657-6
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("tsp_instance_vns", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def vns(
    instance: TSPInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using Variable Neighborhood Search.

    Args:
        instance: A TSPInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix
    start_time = time.time()

    if n <= 3:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Warm-start with nearest neighbor multi-start
    _nn = _load_mod(
        "tsp_nn_vns",
        os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
    )
    best_tour = None
    best_dist = float("inf")
    for start in range(min(n, 5)):
        sol = _nn.nearest_neighbor(instance, start=start)
        if sol.distance < best_dist:
            best_dist = sol.distance
            best_tour = sol.tour[:]

    tour = best_tour[:]
    current_dist = best_dist

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = tour[:]
            if k == 1:
                _shake_2opt(shaken, rng)
            elif k == 2:
                _shake_or_opt(shaken, rng)
            else:
                _shake_double_bridge(shaken, rng)

            # Local search (2-opt)
            shaken_dist = _two_opt_ls(instance, shaken, dist)

            if shaken_dist < current_dist - 1e-10:
                tour = shaken
                current_dist = shaken_dist
                k = 1

                if current_dist < best_dist - 1e-10:
                    best_dist = current_dist
                    best_tour = tour[:]
            else:
                k += 1

    return TSPSolution(
        tour=best_tour,
        distance=instance.tour_distance(best_tour),
    )


def _shake_2opt(tour: list[int], rng: np.random.Generator) -> None:
    """Random 2-opt move."""
    n = len(tour)
    i = rng.integers(0, n)
    j = rng.integers(0, n)
    if i > j:
        i, j = j, i
    if j - i > 1 and not (i == 0 and j == n - 1):
        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]


def _shake_or_opt(tour: list[int], rng: np.random.Generator) -> None:
    """Random Or-opt move (relocate 1-3 cities)."""
    n = len(tour)
    seg_len = min(rng.integers(1, 4), n - 1)
    pos = rng.integers(0, n)
    # Extract segment
    segment = []
    for k in range(seg_len):
        segment.append(tour[(pos + k) % n])
    # Remove segment
    for c in segment:
        tour.remove(c)
    # Insert at random position
    ins = rng.integers(0, len(tour) + 1)
    for k, c in enumerate(segment):
        tour.insert(ins + k, c)


def _shake_double_bridge(tour: list[int], rng: np.random.Generator) -> None:
    """Double-bridge perturbation (4-edge)."""
    n = len(tour)
    if n < 8:
        _shake_2opt(tour, rng)
        return
    cuts = sorted(rng.choice(range(1, n), size=3, replace=False))
    a, b, c = cuts
    new_tour = tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
    tour[:] = new_tour


def _two_opt_ls(
    instance: TSPInstance,
    tour: list[int],
    dist: np.ndarray,
) -> float:
    """Best-improvement 2-opt local search."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return instance.tour_distance(tour)


if __name__ == "__main__":
    inst = TSPInstance.random(n=20, seed=42)
    print(f"TSP: {inst.n} cities")

    _nn = _load_mod(
        "tsp_nn_vns_main",
        os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"),
    )
    nn_sol = _nn.nearest_neighbor(inst)
    print(f"NN: distance={nn_sol.distance:.2f}")

    vns_sol = vns(inst, seed=42)
    print(f"VNS: distance={vns_sol.distance:.2f}")

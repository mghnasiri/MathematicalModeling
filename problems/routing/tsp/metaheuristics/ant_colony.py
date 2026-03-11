"""
Ant Colony Optimization for TSP.

Problem: TSP (Traveling Salesman Problem)
Algorithm: Ant System (AS) with MAX-MIN Ant System (MMAS) pheromone bounds

Each ant probabilistically constructs a tour using pheromone trails and
heuristic distance information. Pheromones are updated using the best
solution found, with bounds to prevent stagnation (MMAS strategy).

Warm-started with nearest neighbor heuristic to initialize pheromone bounds.

Complexity: O(iterations * n_ants * n^2) per run.

References:
    Dorigo, M. & Gambardella, L.M. (1997). Ant colony system: a cooperative
    learning approach to the traveling salesman problem. IEEE Transactions
    on Evolutionary Computation, 1(1), 53-66.
    https://doi.org/10.1109/4235.585892

    Stützle, T. & Hoos, H.H. (2000). MAX-MIN Ant System. Future Generation
    Computer Systems, 16(8), 889-914.
    https://doi.org/10.1016/S0167-739X(00)00043-1
"""

from __future__ import annotations

import os
import time
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module("tsp_instance_aco", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def ant_colony(
    instance: TSPInstance,
    n_ants: int | None = None,
    max_iterations: int = 200,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using Ant Colony Optimization (MMAS variant).

    Args:
        instance: A TSPInstance.
        n_ants: Number of ants per iteration. Default: n.
        max_iterations: Maximum number of iterations.
        alpha: Pheromone importance exponent.
        beta: Heuristic (1/distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix

    if n <= 3:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    if n_ants is None:
        n_ants = min(n, 30)

    start_time = time.time()

    # Heuristic information: eta[i][j] = 1 / d(i,j)
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] > 0:
                eta[i][j] = 1.0 / dist[i][j]

    # Initialize with nearest neighbor to get pheromone bounds
    _nn_mod = _load_module(
        "tsp_nn_aco", os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py")
    )
    nn_sol = _nn_mod.nearest_neighbor_multistart(instance)
    best_tour = nn_sol.tour[:]
    best_dist = nn_sol.distance

    # MMAS pheromone bounds
    tau_max = 1.0 / (rho * best_dist)
    tau_min = tau_max / (2.0 * n)

    # Initialize pheromone matrix
    tau = np.full((n, n), tau_max)
    np.fill_diagonal(tau, 0.0)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        iteration_best_tour = None
        iteration_best_dist = float("inf")

        for _ in range(n_ants):
            tour = _construct_tour(n, tau, eta, alpha, beta, rng)
            d = instance.tour_distance(tour)

            if d < iteration_best_dist:
                iteration_best_dist = d
                iteration_best_tour = tour

        # Update global best
        if iteration_best_dist < best_dist:
            best_dist = iteration_best_dist
            best_tour = iteration_best_tour[:]
            # Update bounds
            tau_max = 1.0 / (rho * best_dist)
            tau_min = tau_max / (2.0 * n)

        # Pheromone evaporation
        tau *= (1.0 - rho)

        # Deposit pheromones using global best (MMAS strategy)
        deposit = 1.0 / best_dist
        for i in range(n):
            j = best_tour[(i + 1) % n]
            ci = best_tour[i]
            tau[ci][j] += deposit
            tau[j][ci] += deposit

        # Clamp pheromones to [tau_min, tau_max]
        np.clip(tau, tau_min, tau_max, out=tau)

    return TSPSolution(tour=best_tour, distance=instance.tour_distance(best_tour))


def _construct_tour(
    n: int,
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[int]:
    """Construct a tour probabilistically using pheromone and heuristic info."""
    visited = [False] * n
    start = rng.integers(0, n)
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        # Compute selection probabilities
        probs = np.zeros(n)
        for j in range(n):
            if not visited[j]:
                probs[j] = (tau[current][j] ** alpha) * (eta[current][j] ** beta)

        total = probs.sum()
        if total <= 0:
            # Fallback: pick random unvisited
            unvisited = [j for j in range(n) if not visited[j]]
            next_city = rng.choice(unvisited)
        else:
            probs /= total
            next_city = rng.choice(n, p=probs)

        tour.append(next_city)
        visited[next_city] = True

    return tour


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Ant Colony Optimization for TSP ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol = ant_colony(inst, seed=42)
        print(f"{name}: distance={sol.distance:.1f}, tour={sol.tour}")

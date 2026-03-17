"""
Ant Colony Optimization for CVRP.

Problem: CVRP (Capacitated Vehicle Routing Problem)
Algorithm: Ant System with MAX-MIN pheromone bounds (MMAS)

Each ant constructs a set of routes by probabilistically selecting the
next customer based on pheromone trails and distance heuristic information.
Routes are built sequentially: when a vehicle reaches capacity, it returns
to the depot and starts a new route.

Warm-started with Clarke-Wright savings to initialize pheromone bounds.

Complexity: O(iterations * n_ants * n^2) per run.

References:
    Bullnheimer, B., Hartl, R.F. & Strauss, C. (1999). An improved ant
    system algorithm for the vehicle routing problem. Annals of Operations
    Research, 89, 319-328.
    https://doi.org/10.1023/A:1018940026670

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


_inst = _load_module("cvrp_instance_aco", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def ant_colony(
    instance: CVRPInstance,
    n_ants: int | None = None,
    max_iterations: int = 200,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using Ant Colony Optimization (MMAS variant).

    Args:
        instance: A CVRPInstance.
        n_ants: Number of ants per iteration. Default: n.
        max_iterations: Maximum number of iterations.
        alpha: Pheromone importance exponent.
        beta: Heuristic (1/distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix
    Q = instance.capacity
    start_time = time.time()

    if n_ants is None:
        n_ants = min(n, 20)

    # Heuristic information: eta[i][j] = 1 / d(i,j) for i,j in 0..n
    size = n + 1
    eta = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j and dist[i][j] > 0:
                eta[i][j] = 1.0 / dist[i][j]

    # Warm-start with Clarke-Wright
    _cw_mod = _load_module(
        "cvrp_cw_aco", os.path.join(_parent_dir, "heuristics", "clarke_wright.py")
    )
    cw_sol = _cw_mod.clarke_wright_savings(instance)
    best_routes = [r[:] for r in cw_sol.routes]
    best_dist = cw_sol.distance

    # MMAS pheromone bounds
    tau_max = 1.0 / (rho * best_dist)
    tau_min = tau_max / (2.0 * n)

    # Initialize pheromone matrix (nodes 0..n)
    tau = np.full((size, size), tau_max)
    np.fill_diagonal(tau, 0.0)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        iter_best_routes = None
        iter_best_dist = float("inf")

        for _ in range(n_ants):
            routes = _construct_solution(n, dist, Q, instance.demands,
                                         tau, eta, alpha, beta, rng)
            d = instance.total_distance(routes)
            if d < iter_best_dist:
                iter_best_dist = d
                iter_best_routes = routes

        # Update global best
        if iter_best_dist < best_dist:
            best_dist = iter_best_dist
            best_routes = [r[:] for r in iter_best_routes]
            tau_max = 1.0 / (rho * best_dist)
            tau_min = tau_max / (2.0 * n)

        # Pheromone evaporation
        tau *= (1.0 - rho)

        # Deposit pheromones using global best
        deposit = 1.0 / best_dist
        for route in best_routes:
            prev = 0
            for cust in route:
                tau[prev][cust] += deposit
                tau[cust][prev] += deposit
                prev = cust
            tau[prev][0] += deposit
            tau[0][prev] += deposit

        # Clamp pheromones
        np.clip(tau, tau_min, tau_max, out=tau)

    return CVRPSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


def _construct_solution(
    n: int,
    dist: np.ndarray,
    Q: float,
    demands: np.ndarray,
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Construct a CVRP solution with one ant."""
    unvisited = set(range(1, n + 1))
    routes = []

    while unvisited:
        route = []
        current = 0
        remaining_cap = Q

        while unvisited:
            # Feasible customers that fit in remaining capacity
            feasible = [c for c in unvisited if demands[c - 1] <= remaining_cap + 1e-10]
            if not feasible:
                break

            # Compute probabilities
            probs = np.zeros(len(feasible))
            for idx, c in enumerate(feasible):
                probs[idx] = (tau[current][c] ** alpha) * (eta[current][c] ** beta)

            total = probs.sum()
            if total <= 0:
                next_cust = rng.choice(feasible)
            else:
                probs /= total
                next_cust = feasible[rng.choice(len(feasible), p=probs)]

            route.append(next_cust)
            unvisited.remove(next_cust)
            remaining_cap -= demands[next_cust - 1]
            current = next_cust

        if route:
            routes.append(route)

    return routes


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Ant Colony Optimization for CVRP ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = ant_colony(inst, seed=42)
        print(f"{name}: {sol}")

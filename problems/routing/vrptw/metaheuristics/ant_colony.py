"""
Ant Colony Optimization for VRPTW.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)
Algorithm: MAX-MIN Ant System with time window feasibility

Each ant constructs routes by probabilistically selecting next customers
based on pheromone trails and heuristic information (distance + urgency).
Routes respect both capacity and time window constraints. New routes are
started when no feasible customer can be added.

Warm-started with Solomon's insertion heuristic.

Complexity: O(iterations * n_ants * n^2) per run.

References:
    Gambardella, L.M., Taillard, É.D. & Agazzi, G. (1999). MACS-VRPTW:
    a multiple ant colony system for vehicle routing problems with time
    windows. In: Corne, D. et al. (eds) New Ideas in Optimization,
    McGraw-Hill, 63-76.

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


_inst = _load_module("vrptw_instance_aco", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def ant_colony(
    instance: VRPTWInstance,
    n_ants: int | None = None,
    max_iterations: int = 200,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using Ant Colony Optimization.

    Args:
        instance: A VRPTWInstance.
        n_ants: Number of ants per iteration. Default: min(n, 15).
        max_iterations: Maximum number of iterations.
        alpha: Pheromone importance exponent.
        beta: Heuristic importance exponent.
        rho: Pheromone evaporation rate.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        VRPTWSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    dist = instance.distance_matrix
    start_time_clock = time.time()

    if n_ants is None:
        n_ants = min(n, 15)

    size = n + 1  # nodes 0..n

    # Heuristic information
    eta = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j and dist[i][j] > 0:
                eta[i][j] = 1.0 / dist[i][j]

    # Warm-start with Solomon insertion
    _si_mod = _load_module(
        "vrptw_si_aco",
        os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"),
    )
    init_sol = _si_mod.solomon_insertion(instance)
    best_routes = [r[:] for r in init_sol.routes]
    best_dist = init_sol.distance

    # MMAS bounds
    tau_max = 1.0 / (rho * max(best_dist, 1e-6))
    tau_min = tau_max / (2.0 * n)

    tau = np.full((size, size), tau_max)
    np.fill_diagonal(tau, 0.0)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time_clock >= time_limit:
            break

        iter_best_routes = None
        iter_best_dist = float("inf")

        for _ in range(n_ants):
            routes = _construct_solution(instance, tau, eta, alpha, beta, rng)
            d = instance.total_distance(routes)
            if d < iter_best_dist:
                iter_best_dist = d
                iter_best_routes = routes

        if iter_best_dist < best_dist:
            best_dist = iter_best_dist
            best_routes = [r[:] for r in iter_best_routes]
            tau_max = 1.0 / (rho * max(best_dist, 1e-6))
            tau_min = tau_max / (2.0 * n)

        # Evaporation
        tau *= (1.0 - rho)

        # Deposit on global best
        deposit = 1.0 / max(best_dist, 1e-6)
        for route in best_routes:
            prev = 0
            for cust in route:
                tau[prev][cust] += deposit
                tau[cust][prev] += deposit
                prev = cust
            tau[prev][0] += deposit
            tau[0][prev] += deposit

        np.clip(tau, tau_min, tau_max, out=tau)

    return VRPTWSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


def _construct_solution(
    instance: VRPTWInstance,
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Construct a feasible VRPTW solution with one ant."""
    n = instance.n
    Q = instance.capacity
    unvisited = set(range(1, n + 1))
    routes = []

    while unvisited:
        route = []
        current = 0
        current_time = instance.time_windows[0][0]
        remaining_cap = Q

        while unvisited:
            # Find feasible customers
            feasible = []
            for c in unvisited:
                if instance.demands[c - 1] > remaining_cap + 1e-10:
                    continue
                arrival = current_time + instance.travel_time(current, c)
                if arrival > instance.time_windows[c][1] + 1e-10:
                    continue
                # Check if we can return to depot after serving c
                start = max(arrival, instance.time_windows[c][0])
                depart = start + instance.service_times[c]
                return_time = depart + instance.travel_time(c, 0)
                if return_time > instance.time_windows[0][1] + 1e-10:
                    continue
                feasible.append(c)

            if not feasible:
                break

            # Probabilistic selection
            probs = np.zeros(len(feasible))
            for idx, c in enumerate(feasible):
                probs[idx] = (tau[current][c] ** alpha) * (eta[current][c] ** beta)

            total = probs.sum()
            if total <= 0:
                next_cust = rng.choice(feasible)
            else:
                probs /= total
                next_cust = feasible[rng.choice(len(feasible), p=probs)]

            # Add to route
            route.append(next_cust)
            unvisited.remove(next_cust)
            arrival = current_time + instance.travel_time(current, next_cust)
            start = max(arrival, instance.time_windows[next_cust][0])
            current_time = start + instance.service_times[next_cust]
            remaining_cap -= instance.demands[next_cust - 1]
            current = next_cust

        if route:
            routes.append(route)

    return routes


if __name__ == "__main__":
    from instance import solomon_c101_mini, tight_tw5

    print("=== Ant Colony Optimization for VRPTW ===\n")

    for name, inst_fn in [
        ("solomon_c101_mini", solomon_c101_mini),
        ("tight_tw5", tight_tw5),
    ]:
        inst = inst_fn()
        sol = ant_colony(inst, seed=42)
        print(f"{name}: {sol}")

"""
Genetic Algorithm for CVRP — Route-based encoding with crossover repair.

Problem: CVRP (Capacitated Vehicle Routing Problem)

Encoding: Giant tour (single permutation of all customers), decoded into
routes by splitting at capacity limits (route-first, cluster-second).

Crossover: Order Crossover (OX) on the giant tour.
Mutation: Swap mutation on the giant tour.
Local search: Optional intra-route 2-opt.

References:
    Prins, C. (2004). A simple and effective evolutionary algorithm
    for the vehicle routing problem. Computers & Operations Research,
    31(12), 1985-2002.
    https://doi.org/10.1016/S0305-0548(03)00158-8

    Baker, B.M. & Ayechew, M.A. (2003). A genetic algorithm for the
    vehicle routing problem. Computers & Operations Research, 30(5),
    787-800.
    https://doi.org/10.1016/S0305-0548(02)00051-5
"""

from __future__ import annotations

import os
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_module("cvrp_instance_ga", os.path.join(_parent_dir, "instance.py"))
CVRPInstance = _inst.CVRPInstance
CVRPSolution = _inst.CVRPSolution


def _split_into_routes(
    instance: CVRPInstance, giant_tour: list[int]
) -> list[list[int]]:
    """Split a giant tour into capacity-feasible routes.

    Args:
        instance: CVRP instance.
        giant_tour: Permutation of customers (1-indexed).

    Returns:
        List of routes.
    """
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_demand = 0.0

    for customer in giant_tour:
        demand = instance.demands[customer - 1]
        if current_demand + demand > instance.capacity + 1e-10:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = demand
        else:
            current_route.append(customer)
            current_demand += demand

    if current_route:
        routes.append(current_route)

    return routes


def _order_crossover(
    parent1: list[int], parent2: list[int], rng: np.random.Generator
) -> list[int]:
    """Order Crossover (OX) on giant tours."""
    n = len(parent1)
    i, j = sorted(rng.choice(n, size=2, replace=False))

    child = [-1] * n
    child[i:j + 1] = parent1[i:j + 1]

    in_child = set(child[i:j + 1])
    pos = (j + 1) % n
    for city in parent2[j + 1:] + parent2[:j + 1]:
        if city not in in_child:
            child[pos] = city
            pos = (pos + 1) % n

    return child


def _intra_route_2opt(instance: CVRPInstance, route: list[int]) -> list[int]:
    """Apply 2-opt within a single route."""
    if len(route) <= 3:
        return route

    dist = instance.distance_matrix
    improved = True
    route = route[:]

    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Consider edges: (prev, route[i]) - (route[j], next)
                prev_node = 0 if i == 0 else route[i - 1]
                next_node = 0 if j == len(route) - 1 else route[j + 1]

                old = dist[prev_node][route[i]] + dist[route[j]][next_node]
                new = dist[prev_node][route[j]] + dist[route[i]][next_node]
                if new < old - 1e-10:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True

    return route


def genetic_algorithm(
    instance: CVRPInstance,
    pop_size: int = 50,
    generations: int = 300,
    mutation_rate: float = 0.15,
    tournament_size: int = 5,
    use_local_search: bool = False,
    seed: int | None = None,
) -> CVRPSolution:
    """Solve CVRP using a genetic algorithm with giant-tour encoding.

    Args:
        instance: A CVRPInstance.
        pop_size: Population size.
        generations: Number of generations.
        mutation_rate: Probability of swap mutation.
        tournament_size: Tournament selection size.
        use_local_search: If True, apply intra-route 2-opt to offspring.
        seed: Random seed for reproducibility.

    Returns:
        CVRPSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if n == 0:
        return CVRPSolution(routes=[], distance=0.0)

    # Initialize population with random giant tours
    population = []
    for _ in range(pop_size):
        perm = list(rng.permutation(n) + 1)  # 1-indexed customers
        population.append(perm)

    # Seed with Clarke-Wright solution
    _cw_mod = _load_module(
        "cvrp_cw_ga", os.path.join(_parent_dir, "heuristics", "clarke_wright.py"))
    cw_sol = _cw_mod.clarke_wright_savings(instance)
    cw_giant = [c for route in cw_sol.routes for c in route]
    if len(cw_giant) == n:
        population[0] = cw_giant

    def evaluate(giant_tour: list[int]) -> float:
        routes = _split_into_routes(instance, giant_tour)
        return instance.total_distance(routes)

    fitness = [evaluate(ind) for ind in population]
    best_idx = int(np.argmin(fitness))
    best_giant = population[best_idx][:]
    best_cost = fitness[best_idx]

    for gen in range(generations):
        new_population = []
        new_fitness = []

        # Elitism
        new_population.append(best_giant[:])
        new_fitness.append(best_cost)

        while len(new_population) < pop_size:
            # Tournament selection
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p1_idx = candidates[int(np.argmin([fitness[c] for c in candidates]))]

            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p2_idx = candidates[int(np.argmin([fitness[c] for c in candidates]))]

            # Crossover
            child = _order_crossover(population[p1_idx], population[p2_idx], rng)

            # Mutation
            if rng.random() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]

            # Optional local search
            if use_local_search:
                routes = _split_into_routes(instance, child)
                routes = [_intra_route_2opt(instance, r) for r in routes]
                child = [c for r in routes for c in r]

            cost = evaluate(child)
            new_population.append(child)
            new_fitness.append(cost)

        population = new_population
        fitness = new_fitness

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_cost:
            best_cost = fitness[gen_best_idx]
            best_giant = population[gen_best_idx][:]

    best_routes = _split_into_routes(instance, best_giant)
    return CVRPSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


if __name__ == "__main__":
    from instance import small6, christofides1, medium12

    print("=== Genetic Algorithm for CVRP ===\n")

    for name, inst_fn in [
        ("small6", small6),
        ("christofides1", christofides1),
        ("medium12", medium12),
    ]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name}: {sol}")

"""
Genetic Algorithm for VRPTW — Giant-tour encoding with TW-aware split.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)

Encoding: Giant tour (permutation of all customers), split into feasible
routes respecting both capacity and time window constraints.

Crossover: Order Crossover (OX) on the giant tour.
Mutation: Swap mutation.
Warm-started with Solomon's insertion heuristic.

References:
    Potvin, J.-Y. & Bengio, S. (1996). The vehicle routing problem
    with time windows part II: Genetic search. INFORMS Journal on
    Computing, 8(2), 165-172.
    https://doi.org/10.1287/ijoc.8.2.165

    Berger, J. & Barkaoui, M. (2004). A parallel hybrid genetic
    algorithm for the vehicle routing problem with time windows.
    Computers & Operations Research, 31(12), 2037-2053.
    https://doi.org/10.1016/S0305-0548(03)00163-1
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


_inst = _load_module("vrptw_instance_ga", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def _split_into_routes(
    instance: VRPTWInstance, giant_tour: list[int]
) -> list[list[int]]:
    """Split a giant tour into capacity- and time-window-feasible routes.

    Args:
        instance: VRPTW instance.
        giant_tour: Permutation of customers (1-indexed).

    Returns:
        List of feasible routes.
    """
    routes: list[list[int]] = []
    current_route: list[int] = []
    current_demand = 0.0
    current_time = instance.time_windows[0][0]
    prev = 0

    for customer in giant_tour:
        demand = instance.demands[customer - 1]

        # Check capacity
        if current_demand + demand > instance.capacity + 1e-10:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = demand
            current_time = instance.time_windows[0][0]
            prev = 0
            # Recompute time for single customer
            arrival = current_time + instance.travel_time(0, customer)
            if arrival > instance.time_windows[customer][1] + 1e-10:
                # Can't even serve this customer from depot — keep anyway
                routes.append(current_route)
                current_route = []
                current_demand = 0.0
                current_time = instance.time_windows[0][0]
                prev = 0
                continue
            start = max(arrival, instance.time_windows[customer][0])
            current_time = start + instance.service_times[customer]
            prev = customer
            continue

        # Check time window
        arrival = current_time + instance.travel_time(prev, customer)
        if arrival > instance.time_windows[customer][1] + 1e-10:
            # Start new route
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = demand
            current_time = instance.time_windows[0][0]
            arrival = current_time + instance.travel_time(0, customer)
            if arrival > instance.time_windows[customer][1] + 1e-10:
                routes.append(current_route)
                current_route = []
                current_demand = 0.0
                current_time = instance.time_windows[0][0]
                prev = 0
                continue
            start = max(arrival, instance.time_windows[customer][0])
            current_time = start + instance.service_times[customer]
            prev = customer
            continue

        # Check return to depot feasibility
        start = max(arrival, instance.time_windows[customer][0])
        depart = start + instance.service_times[customer]
        return_time = depart + instance.travel_time(customer, 0)
        if return_time > instance.time_windows[0][1] + 1e-10:
            if current_route:
                routes.append(current_route)
            current_route = [customer]
            current_demand = demand
            current_time = instance.time_windows[0][0]
            arrival = current_time + instance.travel_time(0, customer)
            start = max(arrival, instance.time_windows[customer][0])
            current_time = start + instance.service_times[customer]
            prev = customer
            continue

        # Feasible — add to current route
        current_route.append(customer)
        current_demand += demand
        current_time = depart
        prev = customer

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


def genetic_algorithm(
    instance: VRPTWInstance,
    pop_size: int = 50,
    generations: int = 300,
    mutation_rate: float = 0.15,
    tournament_size: int = 5,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using a genetic algorithm with giant-tour encoding.

    Args:
        instance: A VRPTWInstance.
        pop_size: Population size.
        generations: Number of generations.
        mutation_rate: Probability of swap mutation.
        tournament_size: Tournament selection size.
        seed: Random seed for reproducibility.

    Returns:
        VRPTWSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if n == 0:
        return VRPTWSolution(routes=[], distance=0.0)

    # Initialize population
    population = []
    for _ in range(pop_size):
        perm = list(rng.permutation(n) + 1)
        population.append(perm)

    # Seed with Solomon solution
    _si_mod = _load_module(
        "vrptw_si_ga",
        os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"))
    si_sol = _si_mod.solomon_insertion(instance)
    si_giant = [c for route in si_sol.routes for c in route]
    if len(si_giant) == n:
        population[0] = si_giant

    def evaluate(giant_tour: list[int]) -> float:
        routes = _split_into_routes(instance, giant_tour)
        return instance.total_distance(routes)

    fitness = [evaluate(ind) for ind in population]
    best_idx = int(np.argmin(fitness))
    best_giant = population[best_idx][:]
    best_cost = fitness[best_idx]

    for gen in range(generations):
        new_population = [best_giant[:]]
        new_fitness = [best_cost]

        while len(new_population) < pop_size:
            # Tournament selection
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p1_idx = candidates[int(np.argmin([fitness[c] for c in candidates]))]
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p2_idx = candidates[int(np.argmin([fitness[c] for c in candidates]))]

            child = _order_crossover(population[p1_idx], population[p2_idx], rng)

            if rng.random() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]

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
    return VRPTWSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


if __name__ == "__main__":
    from instance import solomon_c101_mini, tight_tw5

    print("=== Genetic Algorithm for VRPTW ===\n")

    for name, inst_fn in [
        ("solomon_c101_mini", solomon_c101_mini),
        ("tight_tw5", tight_tw5),
    ]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name}: {sol}")

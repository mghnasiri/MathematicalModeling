"""
Genetic Algorithm for TSP — Order Crossover (OX) with 2-opt local search.

Problem: TSP (Traveling Salesman Problem)

Encoding: Permutation of city indices.
Crossover: Order Crossover (OX) — preserves relative order.
Mutation: Swap mutation — swap two random cities.
Selection: Tournament selection.
Local search: Optional 2-opt applied to offspring.

References:
    Davis, L. (1985). Applying adaptive algorithms to epistatic domains.
    Proceedings of IJCAI, 162-164.

    Goldberg, D.E. & Lingle, R. (1985). Alleles, loci, and the
    traveling salesman problem. Proceedings of ICGA, 154-159.
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

_inst = _load_module("tsp_instance_ga", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def _order_crossover(
    parent1: list[int], parent2: list[int], rng: np.random.Generator
) -> list[int]:
    """Order Crossover (OX): copy a segment from parent1, fill rest from parent2.

    Args:
        parent1: First parent tour.
        parent2: Second parent tour.
        rng: Random number generator.

    Returns:
        Child tour.
    """
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


def _swap_mutation(tour: list[int], rng: np.random.Generator) -> list[int]:
    """Swap two random cities in the tour."""
    n = len(tour)
    i, j = rng.choice(n, size=2, replace=False)
    tour = tour[:]
    tour[i], tour[j] = tour[j], tour[i]
    return tour


def genetic_algorithm(
    instance: TSPInstance,
    pop_size: int = 50,
    generations: int = 500,
    mutation_rate: float = 0.1,
    tournament_size: int = 5,
    use_local_search: bool = False,
    seed: int | None = None,
) -> TSPSolution:
    """Solve TSP using a genetic algorithm with OX crossover.

    Args:
        instance: A TSPInstance.
        pop_size: Population size.
        generations: Number of generations.
        mutation_rate: Probability of mutation per offspring.
        tournament_size: Tournament selection size.
        use_local_search: If True, apply 2-opt to each offspring.
        seed: Random seed for reproducibility.

    Returns:
        TSPSolution with the best tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if n <= 3:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Initialize population with random permutations
    population = []
    for _ in range(pop_size):
        perm = list(rng.permutation(n))
        population.append(perm)

    # Add nearest neighbor solution
    _nn_mod = _load_module(
        "tsp_nn_ga", os.path.join(_parent_dir, "heuristics", "nearest_neighbor.py"))
    nn_sol = _nn_mod.nearest_neighbor(instance)
    population[0] = nn_sol.tour[:]

    fitness = [instance.tour_distance(ind) for ind in population]
    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_cost = fitness[best_idx]

    for gen in range(generations):
        new_population = []
        new_fitness = []

        # Elitism: keep best individual
        new_population.append(best_tour[:])
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
                child = _swap_mutation(child, rng)

            # Optional local search
            if use_local_search:
                _ls_mod = _load_module(
                    "tsp_ls_ga", os.path.join(
                        _parent_dir, "metaheuristics", "local_search.py"))
                sol = _ls_mod.two_opt(instance, child)
                child = sol.tour

            child_cost = instance.tour_distance(child)
            new_population.append(child)
            new_fitness.append(child_cost)

        population = new_population
        fitness = new_fitness

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_cost:
            best_cost = fitness[gen_best_idx]
            best_tour = population[gen_best_idx][:]

    return TSPSolution(tour=best_tour, distance=instance.tour_distance(best_tour))


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Genetic Algorithm ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name}: distance={sol.distance:.1f}")

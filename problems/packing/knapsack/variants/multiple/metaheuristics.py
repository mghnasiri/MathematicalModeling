"""
Genetic Algorithm for Multiple Knapsack Problem (mKP).

Problem: mKP

Integer-vector encoding: gene[j] ∈ {-1, 0, ..., k-1} gives the knapsack
assignment for item j. Repair operator removes lowest-density items from
overloaded knapsacks.

Warm-started with greedy value-density heuristic.

Complexity: O(generations * pop_size * n * k) per run.

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. Wiley.

    Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the
    multidimensional knapsack problem. Journal of Heuristics, 4, 63-86.
    https://doi.org/10.1023/A:1009642405419
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mkp_multi_instance_meta", os.path.join(_this_dir, "instance.py"))
MultipleKnapsackInstance = _inst.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst.MultipleKnapsackSolution

_heur = _load_mod("mkp_multi_heur_meta", os.path.join(_this_dir, "heuristics.py"))
greedy_value_density = _heur.greedy_value_density


def _repair(
    instance: MultipleKnapsackInstance,
    assignments: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Repair infeasible solution by removing lowest-density items."""
    n, k = instance.n, instance.k
    density = instance.values / np.maximum(instance.weights, 1e-10)

    load = np.zeros(k)
    for j in range(n):
        if assignments[j] >= 0:
            load[assignments[j]] += instance.weights[j]

    for i in range(k):
        if load[i] > instance.capacities[i] + 1e-10:
            items_in = [(j, density[j]) for j in range(n) if assignments[j] == i]
            items_in.sort(key=lambda x: x[1])
            for j, _ in items_in:
                if load[i] <= instance.capacities[i] + 1e-10:
                    break
                load[i] -= instance.weights[j]
                assignments[j] = -1

    # Try to add unassigned items
    unassigned = [j for j in range(n) if assignments[j] < 0]
    unassigned.sort(key=lambda j: density[j], reverse=True)
    for j in unassigned:
        for i in range(k):
            if load[i] + instance.weights[j] <= instance.capacities[i] + 1e-10:
                assignments[j] = i
                load[i] += instance.weights[j]
                break

    return assignments


def _fitness(instance: MultipleKnapsackInstance, assignments: list[int]) -> float:
    """Compute fitness (total value of assigned items)."""
    return sum(instance.values[j] for j in range(instance.n) if assignments[j] >= 0)


def genetic_algorithm(
    instance: MultipleKnapsackInstance,
    pop_size: int = 50,
    max_generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> MultipleKnapsackSolution:
    """Solve mKP using a Genetic Algorithm.

    Args:
        instance: A MultipleKnapsackInstance.
        pop_size: Population size.
        max_generations: Maximum generations.
        crossover_rate: Crossover probability.
        mutation_rate: Per-gene mutation probability.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        MultipleKnapsackSolution.
    """
    rng = np.random.default_rng(seed)
    n, k = instance.n, instance.k
    start_time = time.time()

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # Initialize population
    population: list[list[int]] = []

    # Seed with greedy
    greedy_sol = greedy_value_density(instance)
    population.append(greedy_sol.assignments[:])

    # Random individuals
    for _ in range(pop_size - 1):
        ind = list(rng.integers(-1, k, size=n))
        ind = _repair(instance, ind, rng)
        population.append(ind)

    fitness = [_fitness(instance, ind) for ind in population]
    best_idx = int(np.argmax(fitness))
    best_ind = population[best_idx][:]
    best_val = fitness[best_idx]

    for gen in range(max_generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_pop = [best_ind[:]]  # Elitism

        while len(new_pop) < pop_size:
            # Tournament selection
            i1, i2 = rng.integers(0, pop_size, size=2)
            p1 = population[i1] if fitness[i1] >= fitness[i2] else population[i2]
            i3, i4 = rng.integers(0, pop_size, size=2)
            p2 = population[i3] if fitness[i3] >= fitness[i4] else population[i4]

            # Uniform crossover
            if rng.random() < crossover_rate:
                child = [p1[j] if rng.random() < 0.5 else p2[j] for j in range(n)]
            else:
                child = p1[:]

            # Mutation
            for j in range(n):
                if rng.random() < mutation_rate:
                    child[j] = int(rng.integers(-1, k))

            child = _repair(instance, child, rng)
            new_pop.append(child)

        population = new_pop[:pop_size]
        fitness = [_fitness(instance, ind) for ind in population]

        gen_best = int(np.argmax(fitness))
        if fitness[gen_best] > best_val + 1e-10:
            best_val = fitness[gen_best]
            best_ind = population[gen_best][:]

    return MultipleKnapsackSolution(assignments=best_ind, value=best_val)


if __name__ == "__main__":
    inst = MultipleKnapsackInstance.random(n=20, k=3, seed=42)
    print(f"mKP: {inst.n} items, {inst.k} knapsacks")

    gr_sol = greedy_value_density(inst)
    print(f"Greedy: value={gr_sol.value:.0f}")

    ga_sol = genetic_algorithm(inst, seed=42)
    print(f"GA: value={ga_sol.value:.0f}")

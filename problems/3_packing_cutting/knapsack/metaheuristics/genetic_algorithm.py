"""
Genetic Algorithm for the 0-1 Knapsack Problem.

Problem: 0-1 Knapsack (KP01)

Encoding: Binary vector x ∈ {0,1}^n where x_i = 1 if item i is selected.
Crossover: Uniform crossover — each bit from either parent with equal probability.
Mutation: Bit flip — flip each bit with probability 1/n.
Repair: If infeasible, remove items with lowest value/weight ratio until feasible.
Fitness: Total value (infeasible solutions are repaired).

References:
    Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the
    multidimensional knapsack problem. Journal of Heuristics,
    4(1), 63-86.
    https://doi.org/10.1023/A:1009642405419

    Khuri, S., Bäck, T. & Heitkötter, J. (1994). The zero/one
    multiple knapsack problem and genetic algorithms. Proceedings
    of the 1994 ACM Symposium on Applied Computing, 188-193.
    https://doi.org/10.1145/326619.326694
"""

from __future__ import annotations

import os
import importlib.util
import sys

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("kp_instance_ga", os.path.join(_parent_dir, "instance.py"))
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def _repair(
    instance: KnapsackInstance,
    chromosome: np.ndarray,
) -> np.ndarray:
    """Repair infeasible solution by removing worst-ratio items.

    Args:
        instance: Knapsack instance.
        chromosome: Binary vector.

    Returns:
        Repaired binary vector.
    """
    chrom = chromosome.copy()
    total_weight = float(np.dot(chrom, instance.weights))

    if total_weight <= instance.capacity + 1e-10:
        return chrom

    # Sort selected items by value/weight ratio (ascending) for removal
    selected = np.where(chrom == 1)[0]
    ratios = [
        (instance.values[i] / instance.weights[i] if instance.weights[i] > 0
         else float("inf"))
        for i in selected
    ]
    order = sorted(range(len(selected)), key=lambda k: ratios[k])

    for k in order:
        i = selected[k]
        chrom[i] = 0
        total_weight -= instance.weights[i]
        if total_weight <= instance.capacity + 1e-10:
            break

    return chrom


def genetic_algorithm(
    instance: KnapsackInstance,
    pop_size: int = 50,
    generations: int = 200,
    mutation_rate: float | None = None,
    tournament_size: int = 3,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using a genetic algorithm.

    Args:
        instance: A KnapsackInstance.
        pop_size: Population size.
        generations: Number of generations.
        mutation_rate: Bit-flip probability. If None, uses 1/n.
        tournament_size: Tournament selection size.
        seed: Random seed for reproducibility.

    Returns:
        KnapsackSolution with the best solution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if n == 0:
        return KnapsackSolution(items=[], value=0.0, weight=0.0)

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # Initialize population
    population = rng.integers(0, 2, size=(pop_size, n)).astype(float)
    for i in range(pop_size):
        population[i] = _repair(instance, population[i])

    def fitness(chrom: np.ndarray) -> float:
        return float(np.dot(chrom, instance.values))

    fitnesses = np.array([fitness(ind) for ind in population])
    best_idx = int(np.argmax(fitnesses))
    best_chrom = population[best_idx].copy()
    best_fit = fitnesses[best_idx]

    for gen in range(generations):
        new_pop = [best_chrom.copy()]  # Elitism

        while len(new_pop) < pop_size:
            # Tournament selection
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p1_idx = candidates[int(np.argmax(fitnesses[candidates]))]
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p2_idx = candidates[int(np.argmax(fitnesses[candidates]))]

            # Uniform crossover
            mask = rng.integers(0, 2, size=n)
            child = np.where(mask, population[p1_idx], population[p2_idx])

            # Mutation
            flip_mask = rng.random(n) < mutation_rate
            child = np.where(flip_mask, 1.0 - child, child)

            # Repair
            child = _repair(instance, child)
            new_pop.append(child)

        population = np.array(new_pop[:pop_size])
        fitnesses = np.array([fitness(ind) for ind in population])

        gen_best = int(np.argmax(fitnesses))
        if fitnesses[gen_best] > best_fit:
            best_fit = fitnesses[gen_best]
            best_chrom = population[gen_best].copy()

    items = sorted(int(i) for i in np.where(best_chrom == 1)[0])
    return KnapsackSolution(
        items=items,
        value=instance.total_value(items),
        weight=instance.total_weight(items),
    )


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== Genetic Algorithm for 0-1 Knapsack ===\n")

    for name, inst_fn in [
        ("small4", small_knapsack_4),
        ("medium8", medium_knapsack_8),
    ]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name}: {sol}")

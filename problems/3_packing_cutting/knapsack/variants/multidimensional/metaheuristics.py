"""
Genetic Algorithm for the Multi-dimensional Knapsack Problem (MKP).

Problem: MKP (d-KP)

Encoding: Binary vector x ∈ {0,1}^n where x_i = 1 if item i is selected.
Crossover: Uniform crossover — each bit from either parent with p=0.5.
Mutation: Bit flip with probability 1/n per bit.
Repair: If infeasible, remove items with worst pseudo-utility ratio
    until all capacity constraints are satisfied.
Fitness: Total value after repair.

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


_inst = _load_mod("mkp_instance_meta", os.path.join(_this_dir, "instance.py"))
MKPInstance = _inst.MKPInstance
MKPSolution = _inst.MKPSolution

_heur = _load_mod("mkp_heuristics", os.path.join(_this_dir, "heuristics.py"))
greedy_pseudo_utility = _heur.greedy_pseudo_utility


def _repair(
    instance: MKPInstance,
    chrom: np.ndarray,
    pseudo_utility: np.ndarray,
) -> np.ndarray:
    """Repair infeasible chromosome by removing worst-ratio items."""
    c = chrom.copy()
    weights = instance.weights @ c  # shape (d,)

    violations = weights - instance.capacities
    if np.all(violations <= 1e-10):
        return c

    # Remove items with lowest pseudo-utility first
    selected = np.where(c == 1)[0]
    order = sorted(selected, key=lambda i: pseudo_utility[i])

    for i in order:
        c[i] = 0
        weights -= instance.weights[:, i]
        if np.all(weights <= instance.capacities + 1e-10):
            break

    return c


def genetic_algorithm(
    instance: MKPInstance,
    pop_size: int = 50,
    max_generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> MKPSolution:
    """Solve MKP using a Genetic Algorithm.

    Args:
        instance: An MKPInstance.
        pop_size: Population size.
        max_generations: Maximum number of generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Per-bit mutation probability. Defaults to 1/n.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        MKPSolution with the best selection found.
    """
    rng = np.random.default_rng(seed)
    n, d = instance.n, instance.d
    start_time = time.time()

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # Precompute pseudo-utility
    pseudo_utility = np.zeros(n)
    for i in range(n):
        norm_w = sum(
            instance.weights[j][i] / max(instance.capacities[j], 1e-10)
            for j in range(d)
        )
        pseudo_utility[i] = instance.values[i] / max(norm_w, 1e-10)

    # Initialize population from greedy + random
    population = np.zeros((pop_size, n), dtype=int)

    # Seed with greedy solution
    greedy_sol = greedy_pseudo_utility(instance)
    population[0][greedy_sol.items] = 1

    # Random initialization for rest
    for k in range(1, pop_size):
        density = rng.uniform(0.2, 0.8)
        population[k] = (rng.random(n) < density).astype(int)
        population[k] = _repair(instance, population[k], pseudo_utility)

    # Compute fitness
    fitness = np.array([
        float(instance.values @ population[k])
        for k in range(pop_size)
    ])

    best_idx = np.argmax(fitness)
    best_chrom = population[best_idx].copy()
    best_val = fitness[best_idx]

    for gen in range(max_generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_pop = np.zeros_like(population)
        new_fitness = np.zeros(pop_size)

        # Elitism: keep best
        new_pop[0] = best_chrom.copy()
        new_fitness[0] = best_val

        for k in range(1, pop_size):
            # Tournament selection (size 3)
            t1, t2, t3 = rng.choice(pop_size, 3, replace=False)
            p1 = max(t1, t2, t3, key=lambda i: fitness[i])
            t1, t2, t3 = rng.choice(pop_size, 3, replace=False)
            p2 = max(t1, t2, t3, key=lambda i: fitness[i])

            # Crossover
            if rng.random() < crossover_rate:
                mask = rng.integers(0, 2, size=n)
                child = np.where(mask, population[p1], population[p2])
            else:
                child = population[p1].copy()

            # Mutation
            flip = rng.random(n) < mutation_rate
            child = np.where(flip, 1 - child, child)

            # Repair
            child = _repair(instance, child, pseudo_utility)
            new_pop[k] = child
            new_fitness[k] = float(instance.values @ child)

        population = new_pop
        fitness = new_fitness

        gen_best = np.argmax(fitness)
        if fitness[gen_best] > best_val + 1e-10:
            best_val = fitness[gen_best]
            best_chrom = population[gen_best].copy()

    items = sorted(np.where(best_chrom == 1)[0].tolist())
    return MKPSolution(
        items=items,
        value=instance.total_value(items),
        weights=instance.total_weights(items),
    )


if __name__ == "__main__":
    inst = MKPInstance.random(n=20, d=3, seed=42)
    print(f"MKP: {inst.n} items, {inst.d} dims")

    gr_sol = greedy_pseudo_utility(inst)
    print(f"Greedy: value={gr_sol.value:.1f}")

    ga_sol = genetic_algorithm(inst, seed=42)
    print(f"GA: value={ga_sol.value:.1f}")

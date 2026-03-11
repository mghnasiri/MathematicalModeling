"""
Genetic Algorithm for 1D Cutting Stock Problem (CSP).

Problem: CSP1D (1D Cutting Stock)

Encoding: Permutation of individual items (expanded from demands).
Decoded into rolls using First Fit: process items in permutation order,
placing each into the first roll where it fits.

Operators:
- Order Crossover (OX): preserve relative order from both parents
- Swap mutation: exchange two random genes

Warm-started with FFD ordering.

Complexity: O(generations * pop_size * N) per run, where N = total items.

References:
    Falkenauer, E. (1996). A hybrid grouping genetic algorithm for bin
    packing. Journal of Heuristics, 2(1), 5-30.
    https://doi.org/10.1007/BF00226291

    Reeves, C.R. (1996). Hybrid genetic algorithms for bin-packing and
    related problems. Annals of Operations Research, 63(3), 371-396.
    https://doi.org/10.1007/BF02125404
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("csp_instance_ga", os.path.join(_parent_dir, "instance.py"))
CuttingStockInstance = _inst.CuttingStockInstance
CuttingStockSolution = _inst.CuttingStockSolution


def _decode_ff(
    lengths: np.ndarray,
    stock_length: float,
    perm: list[int],
) -> list[list[int]]:
    """Decode a permutation of items into rolls using First Fit."""
    rolls: list[list[int]] = []
    remaining: list[float] = []

    for item_type in perm:
        size = lengths[item_type]
        placed = False
        for r in range(len(rolls)):
            if remaining[r] >= size - 1e-10:
                rolls[r].append(item_type)
                remaining[r] -= size
                placed = True
                break
        if not placed:
            rolls.append([item_type])
            remaining.append(stock_length - size)

    return rolls


def _rolls_to_solution(
    instance: CuttingStockInstance,
    rolls: list[list[int]],
) -> CuttingStockSolution:
    """Convert decoded rolls to a CuttingStockSolution."""
    pattern_dict: dict[tuple, int] = {}
    for roll in rolls:
        counts = np.zeros(instance.m, dtype=int)
        for item_type in roll:
            counts[item_type] += 1
        key = tuple(counts)
        pattern_dict[key] = pattern_dict.get(key, 0) + 1

    patterns = [
        (np.array(key, dtype=int), freq)
        for key, freq in pattern_dict.items()
    ]
    num_rolls = sum(freq for _, freq in patterns)
    return CuttingStockSolution(patterns=patterns, num_rolls=num_rolls)


def _ox_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Order Crossover (OX) for permutations with duplicates.

    Copies a segment from parent1, fills remaining positions from parent2
    preserving relative order.
    """
    n = len(parent1)
    i, j = sorted(rng.choice(n, size=2, replace=False))

    child = [None] * n
    # Copy segment from parent1
    child[i:j + 1] = parent1[i:j + 1]

    # Track how many of each type we've placed
    placed: dict[int, int] = {}
    for gene in child[i:j + 1]:
        placed[gene] = placed.get(gene, 0) + 1

    # Count how many of each type we need total (from parent1)
    total_needed: dict[int, int] = {}
    for gene in parent1:
        total_needed[gene] = total_needed.get(gene, 0) + 1

    # Fill from parent2, skipping items we've already placed enough of
    p2_filtered = []
    remaining_needed = {k: v - placed.get(k, 0) for k, v in total_needed.items()}
    for gene in parent2:
        if remaining_needed.get(gene, 0) > 0:
            p2_filtered.append(gene)
            remaining_needed[gene] -= 1

    idx = 0
    for pos in range(n):
        if child[pos] is None:
            child[pos] = p2_filtered[idx]
            idx += 1

    return child


def genetic_algorithm(
    instance: CuttingStockInstance,
    population_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    time_limit: float | None = None,
    seed: int | None = None,
) -> CuttingStockSolution:
    """Solve CSP using a Genetic Algorithm with permutation encoding.

    Args:
        instance: Cutting stock instance.
        population_size: Number of individuals.
        generations: Maximum generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best CuttingStockSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    pop_size = population_size

    # Expand demands into individual items
    items: list[int] = []
    for i in range(instance.m):
        items.extend([i] * instance.demands[i])

    total_items = len(items)
    if total_items == 0:
        return CuttingStockSolution(patterns=[], num_rolls=0)

    # ── Initialize population ────────────────────────────────────────────
    population: list[list[int]] = []
    fitnesses: list[int] = []

    # Seed with FFD ordering
    ffd_perm = sorted(items, key=lambda i: -instance.lengths[i])
    ffd_rolls = _decode_ff(instance.lengths, instance.stock_length, ffd_perm)
    population.append(ffd_perm)
    fitnesses.append(len(ffd_rolls))

    # Fill with random permutations
    while len(population) < pop_size:
        perm = list(items)
        rng.shuffle(perm)
        rolls = _decode_ff(instance.lengths, instance.stock_length, perm)
        population.append(perm)
        fitnesses.append(len(rolls))

    gbest_idx = int(np.argmin(fitnesses))
    gbest_perm = list(population[gbest_idx])
    gbest_rolls = fitnesses[gbest_idx]

    for gen in range(generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Binary tournament selection
        def tournament():
            a, b = rng.choice(pop_size, size=2, replace=False)
            return a if fitnesses[a] <= fitnesses[b] else b

        p1_idx = tournament()
        p2_idx = tournament()

        # Crossover
        if rng.random() < crossover_rate and total_items >= 2:
            child = _ox_crossover(
                population[p1_idx], population[p2_idx], rng,
            )
        else:
            child = list(population[p1_idx])

        # Mutation: swap two genes
        if rng.random() < mutation_rate and total_items >= 2:
            i, j = rng.choice(total_items, size=2, replace=False)
            child[i], child[j] = child[j], child[i]

        # Evaluate
        child_rolls = _decode_ff(instance.lengths, instance.stock_length, child)
        child_fitness = len(child_rolls)

        # Replace worst
        worst_idx = int(np.argmax(fitnesses))
        if child_fitness < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = child_fitness

            if child_fitness < gbest_rolls:
                gbest_rolls = child_fitness
                gbest_perm = list(child)

    best_rolls = _decode_ff(instance.lengths, instance.stock_length, gbest_perm)
    return _rolls_to_solution(instance, best_rolls)


if __name__ == "__main__":
    from instance import simple_csp_3, classic_csp_4

    print("=== GA for Cutting Stock ===\n")

    for name, inst_fn in [
        ("simple3", simple_csp_3),
        ("classic4", classic_csp_4),
    ]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name} (LB={inst.lower_bound()}): GA={sol.num_rolls} rolls")

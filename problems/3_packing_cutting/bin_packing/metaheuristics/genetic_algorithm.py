"""
Genetic Algorithm for the 1D Bin Packing Problem.

Problem: 1D Bin Packing (BPP)

Encoding: Permutation of items — decoded using First Fit Decreasing on
the permutation order (items placed in the given order using First Fit).

Crossover: Order Crossover (OX).
Mutation: Swap two random items in the permutation.
Fitness: Number of bins used (minimize).

References:
    Falkenauer, E. (1996). A hybrid grouping genetic algorithm for
    bin packing. Journal of Heuristics, 2(1), 5-30.
    https://doi.org/10.1007/BF00226291

    Quiroz-Castellanos, M., Cruz-Reyes, L., Torres-Jimenez, J.,
    Gómez-Santillán, C., Fraire-Huacuja, H.J. & Alvim, A.C.F.
    (2015). A grouping genetic algorithm with controlled gene
    transmission for the bin packing problem. Computers &
    Operations Research, 55, 52-64.
    https://doi.org/10.1016/j.cor.2014.10.010
"""

from __future__ import annotations

import os
import sys

import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("bpp_instance_ga", os.path.join(_parent_dir, "instance.py"))
BinPackingInstance = _inst.BinPackingInstance
BinPackingSolution = _inst.BinPackingSolution


def _decode_permutation(
    instance: BinPackingInstance, perm: list[int]
) -> list[list[int]]:
    """Decode a permutation into bins using First Fit.

    Args:
        instance: BPP instance.
        perm: Permutation of item indices.

    Returns:
        List of bins (each a list of item indices).
    """
    bins: list[list[int]] = []
    remaining: list[float] = []

    for idx in perm:
        size = instance.sizes[idx]
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= size - 1e-10:
                bins[b].append(idx)
                remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining.append(instance.capacity - size)

    return bins


def _order_crossover(
    parent1: list[int], parent2: list[int], rng: np.random.Generator
) -> list[int]:
    """Order Crossover (OX)."""
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
    instance: BinPackingInstance,
    pop_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.15,
    tournament_size: int = 3,
    seed: int | None = None,
) -> BinPackingSolution:
    """Solve 1D Bin Packing using a genetic algorithm.

    Args:
        instance: A BinPackingInstance.
        pop_size: Population size.
        generations: Number of generations.
        mutation_rate: Probability of swap mutation.
        tournament_size: Tournament selection size.
        seed: Random seed for reproducibility.

    Returns:
        BinPackingSolution with the best packing found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if n == 0:
        return BinPackingSolution(bins=[], num_bins=0)

    # Initialize population: sorted permutation + random ones
    population = []
    # Seed with sorted-decreasing order (like FFD)
    sorted_perm = sorted(range(n), key=lambda i: instance.sizes[i], reverse=True)
    population.append(sorted_perm)

    for _ in range(pop_size - 1):
        perm = list(rng.permutation(n))
        population.append(perm)

    def fitness(perm: list[int]) -> int:
        return len(_decode_permutation(instance, perm))

    fitnesses = [fitness(ind) for ind in population]
    best_idx = int(np.argmin(fitnesses))
    best_perm = population[best_idx][:]
    best_fit = fitnesses[best_idx]

    for gen in range(generations):
        new_pop = [best_perm[:]]  # Elitism

        while len(new_pop) < pop_size:
            # Tournament selection (minimize bins)
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p1_idx = candidates[int(np.argmin([fitnesses[c] for c in candidates]))]
            candidates = rng.choice(pop_size, size=tournament_size, replace=False)
            p2_idx = candidates[int(np.argmin([fitnesses[c] for c in candidates]))]

            child = _order_crossover(population[p1_idx], population[p2_idx], rng)

            if rng.random() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]

            new_pop.append(child)

        population = new_pop[:pop_size]
        fitnesses = [fitness(ind) for ind in population]

        gen_best = int(np.argmin(fitnesses))
        if fitnesses[gen_best] < best_fit:
            best_fit = fitnesses[gen_best]
            best_perm = population[gen_best][:]

    best_bins = _decode_permutation(instance, best_perm)
    return BinPackingSolution(bins=best_bins, num_bins=len(best_bins))


if __name__ == "__main__":
    from instance import easy_bpp_6, tight_bpp_8

    print("=== Genetic Algorithm for Bin Packing ===\n")

    for name, inst_fn in [("easy6", easy_bpp_6), ("tight8", tight_bpp_8)]:
        inst = inst_fn()
        sol = genetic_algorithm(inst, seed=42)
        print(f"{name}: {sol}")

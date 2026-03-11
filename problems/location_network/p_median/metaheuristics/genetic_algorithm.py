"""
Genetic Algorithm for the p-Median Problem.

Problem: PMP (p-Median Problem)

Encoding: Binary vector of length m where exactly p genes are 1 (open
facilities). Customers are assigned to their nearest open facility.

Operators:
- Uniform crossover with repair to ensure exactly p facilities
- Bit-flip mutation: close one facility and open another
- Binary tournament selection

Warm-started with greedy and Teitz-Bart solutions.

Complexity: O(generations * pop_size * m * n) per run.

References:
    Alp, O., Erkut, E. & Drezner, Z. (2003). An efficient genetic
    algorithm for the p-median problem. Annals of Operations Research,
    122(1), 21-42.
    https://doi.org/10.1023/A:1026130003508

    Resende, M.G.C. & Werneck, R.F. (2004). A hybrid heuristic for
    the p-median problem. Journal of Heuristics, 10(1), 59-88.
    https://doi.org/10.1023/B:HEUR.0000019986.96257.50
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


_inst = _load_mod("pm_instance_ga", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution

_greedy = _load_mod(
    "pm_greedy_ga",
    os.path.join(_parent_dir, "heuristics", "greedy_pmedian.py"),
)
greedy_pmedian = _greedy.greedy_pmedian
interchange = _greedy.interchange


def _evaluate(
    instance: PMedianInstance,
    chromosome: np.ndarray,
) -> tuple[float, list[int]]:
    """Evaluate a binary chromosome (exactly p ones).

    Returns:
        (total_cost, assignments)
    """
    open_facs = [i for i in range(instance.m) if chromosome[i]]
    if not open_facs:
        return float("inf"), [0] * instance.n

    assignments = []
    total = 0.0
    for j in range(instance.n):
        best_fac = min(open_facs, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best_fac)
        total += instance.weights[j] * instance.distance_matrix[best_fac][j]
    return total, assignments


def _repair(
    chromosome: np.ndarray,
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Repair chromosome to have exactly p open facilities."""
    result = chromosome.copy()
    open_count = int(np.sum(result))

    while open_count > p:
        open_indices = np.where(result == 1)[0]
        close_idx = rng.choice(open_indices)
        result[close_idx] = 0
        open_count -= 1

    while open_count < p:
        closed_indices = np.where(result == 0)[0]
        open_idx = rng.choice(closed_indices)
        result[open_idx] = 1
        open_count += 1

    return result


def genetic_algorithm(
    instance: PMedianInstance,
    population_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMedianSolution:
    """Solve p-Median using a Genetic Algorithm.

    Args:
        instance: p-Median instance.
        population_size: Number of individuals.
        generations: Maximum generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation (swap open/closed).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best PMedianSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    m, p = instance.m, instance.p
    pop_size = population_size

    # ── Initialize population ────────────────────────────────────────────
    population: list[np.ndarray] = []
    fitnesses: list[float] = []

    # Seed from heuristics
    for heuristic in [greedy_pmedian, interchange]:
        sol = heuristic(instance)
        chrom = np.zeros(m, dtype=int)
        for i in sol.open_facilities:
            chrom[i] = 1
        cost, _ = _evaluate(instance, chrom)
        population.append(chrom)
        fitnesses.append(cost)

    # Fill with random chromosomes (exactly p ones)
    while len(population) < pop_size:
        chrom = np.zeros(m, dtype=int)
        indices = rng.choice(m, size=p, replace=False)
        chrom[indices] = 1
        cost, _ = _evaluate(instance, chrom)
        population.append(chrom)
        fitnesses.append(cost)

    gbest_idx = int(np.argmin(fitnesses))
    gbest_chrom = population[gbest_idx].copy()
    gbest_cost = fitnesses[gbest_idx]

    for gen in range(generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Binary tournament selection
        def tournament():
            a, b = rng.choice(pop_size, size=2, replace=False)
            return a if fitnesses[a] <= fitnesses[b] else b

        p1_idx = tournament()
        p2_idx = tournament()

        # Uniform crossover
        if rng.random() < crossover_rate:
            mask = rng.integers(0, 2, size=m)
            child = np.where(mask, population[p1_idx], population[p2_idx])
            child = _repair(child, p, rng)
        else:
            child = population[p1_idx].copy()

        # Mutation: swap one open/closed pair
        if rng.random() < mutation_rate:
            open_indices = np.where(child == 1)[0]
            closed_indices = np.where(child == 0)[0]
            if len(open_indices) > 0 and len(closed_indices) > 0:
                close_fac = rng.choice(open_indices)
                open_fac = rng.choice(closed_indices)
                child[close_fac] = 0
                child[open_fac] = 1

        child_cost, _ = _evaluate(instance, child)

        # Replace worst
        worst_idx = int(np.argmax(fitnesses))
        if child_cost < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = child_cost

            if child_cost < gbest_cost:
                gbest_cost = child_cost
                gbest_chrom = child.copy()

    # Build solution
    _, best_assignments = _evaluate(instance, gbest_chrom)
    open_facilities = [i for i in range(m) if gbest_chrom[i]]

    return PMedianSolution(
        open_facilities=open_facilities,
        assignments=best_assignments,
        cost=gbest_cost,
    )


if __name__ == "__main__":
    from instance import small_pmedian_6_2

    inst = small_pmedian_6_2()
    sol = genetic_algorithm(inst, seed=42)
    print(f"GA: cost={sol.cost:.1f}, open={sol.open_facilities}")

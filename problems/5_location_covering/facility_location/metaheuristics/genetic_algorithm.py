"""
Genetic Algorithm for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)

Encoding: Binary vector of length m, where gene[i] = 1 if facility i
is open. Customers are assigned to their nearest open facility.

Operators:
- Uniform crossover: each gene inherited from a random parent
- Bit-flip mutation: toggle each facility with probability 1/m
- Repair: ensure at least one facility is open

Warm-started with greedy add and greedy drop solutions.

Complexity: O(generations * pop_size * m * n) per run.

References:
    Kratica, J., Tošić, D., Filipović, V. & Ljubić, I. (2001).
    Solving the simple plant location problem by genetic algorithm.
    RAIRO - Operations Research, 35(1), 127-142.
    https://doi.org/10.1051/ro:2001107

    Cornuéjols, G., Nemhauser, G.L. & Wolsey, L.A. (1990). The
    uncapacitated facility location problem. In: Mirchandani, P.B.
    & Francis, R.L. (eds) Discrete Location Theory, Wiley, 119-171.
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


_inst = _load_mod("fl_instance_ga", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution

_greedy = _load_mod(
    "fl_greedy_ga",
    os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
)
greedy_add = _greedy.greedy_add
greedy_drop = _greedy.greedy_drop


def _evaluate(
    instance: FacilityLocationInstance,
    chromosome: np.ndarray,
) -> tuple[float, list[int]]:
    """Evaluate a binary chromosome.

    Returns:
        (total_cost, assignments)
    """
    open_set = [i for i in range(instance.m) if chromosome[i]]
    if not open_set:
        # Penalty: open cheapest facility
        open_set = [int(np.argmin(instance.fixed_costs))]

    assignments = []
    assign_cost = 0.0
    for j in range(instance.n):
        best_fac = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best_fac)
        assign_cost += instance.assignment_costs[best_fac][j]

    fixed_cost = sum(instance.fixed_costs[i] for i in open_set)
    return fixed_cost + assign_cost, assignments


def genetic_algorithm(
    instance: FacilityLocationInstance,
    population_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using a Genetic Algorithm.

    Args:
        instance: Facility location instance.
        population_size: Number of individuals.
        generations: Maximum generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Per-gene flip probability. Default: 1/m.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FacilityLocationSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    m = instance.m
    pop_size = population_size

    if mutation_rate is None:
        mutation_rate = 1.0 / m

    # ── Initialize population ────────────────────────────────────────────
    population: list[np.ndarray] = []
    fitnesses: list[float] = []

    # Seed from greedy heuristics
    for heuristic in [greedy_add, greedy_drop]:
        sol = heuristic(instance)
        chrom = np.zeros(m, dtype=int)
        for i in sol.open_facilities:
            chrom[i] = 1
        cost, _ = _evaluate(instance, chrom)
        population.append(chrom)
        fitnesses.append(cost)

    # Fill with random
    while len(population) < pop_size:
        chrom = rng.integers(0, 2, size=m)
        if np.sum(chrom) == 0:
            chrom[rng.integers(0, m)] = 1
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
        else:
            child = population[p1_idx].copy()

        # Mutation: bit-flip
        flip = rng.random(size=m) < mutation_rate
        child = np.where(flip, 1 - child, child)

        # Repair: ensure at least one facility is open
        if np.sum(child) == 0:
            child[rng.integers(0, m)] = 1

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

    return FacilityLocationSolution(
        open_facilities=open_facilities,
        assignments=best_assignments,
        cost=gbest_cost,
    )


if __name__ == "__main__":
    from instance import small_uflp_3_5, medium_uflp_5_10

    print("=== GA for Facility Location ===\n")

    inst = small_uflp_3_5()
    sol = genetic_algorithm(inst, seed=42)
    print(f"small_3_5: cost={sol.cost:.1f}, open={sol.open_facilities}")

    inst2 = medium_uflp_5_10()
    sol2 = genetic_algorithm(inst2, seed=42)
    print(f"medium_5_10: cost={sol2.cost:.1f}, open={sol2.open_facilities}")

"""
Genetic Algorithm for Single Machine Weighted Tardiness — 1 || ΣwjTj

Implements a GA using permutation encoding (job sequence). Each
chromosome is a permutation of jobs decoded by computing completion
times and evaluating the objective.

Algorithm:
    1. Initialize population from ATC/EDD + random permutations.
    2. Binary tournament selection.
    3. Order Crossover (OX).
    4. Swap mutation.
    5. Replace worst if offspring is better.

Notation: 1 || ΣwjTj (or 1 || ΣTj)
Complexity: O(generations * pop_size * n) per run.

References:
    Rubin, P.A. & Ragatz, G.L. (1995). Scheduling in a sequence
    dependent setup environment with genetic search. Computers &
    Operations Research, 22(1), 85-99.
    https://doi.org/10.1016/0305-0548(93)E0016-M

    Bean, J.C. (1994). Genetic algorithms and random keys for
    sequencing and optimization. ORSA Journal on Computing, 6(2),
    154-160.
    https://doi.org/10.1287/ijoc.6.2.154
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("sm_instance_ga", os.path.join(_sm_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_rules = _load_mod(
    "sm_dispatching_ga",
    os.path.join(_sm_dir, "heuristics", "dispatching_rules.py"),
)
edd = _rules.edd

_atc_mod = _load_mod(
    "sm_atc_ga",
    os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py"),
)
atc = _atc_mod.atc


def _ox_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Order Crossover (OX) for permutations."""
    n = len(parent1)
    i, j = sorted(rng.choice(n, size=2, replace=False))

    child = [None] * n
    child[i:j + 1] = parent1[i:j + 1]

    segment = set(child[i:j + 1])
    p2_filtered = [g for g in parent2 if g not in segment]

    idx = 0
    for pos in range(n):
        if child[pos] is None:
            child[pos] = p2_filtered[idx]
            idx += 1

    return child


def genetic_algorithm(
    instance: SingleMachineInstance,
    objective: str = "weighted_tardiness",
    population_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    time_limit: float | None = None,
    seed: int | None = None,
) -> SingleMachineSolution:
    """Solve single machine scheduling using a Genetic Algorithm.

    Args:
        instance: Single machine instance (must have due_dates).
        objective: "weighted_tardiness" (ΣwjTj) or "total_tardiness" (ΣTj).
        population_size: Number of individuals.
        generations: Maximum generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best SingleMachineSolution found.
    """
    rng = np.random.default_rng(seed)
    assert instance.due_dates is not None, "Due dates required"
    n = instance.n
    start_time = time.time()
    pop_size = population_size

    eval_fn = (
        compute_weighted_tardiness if objective == "weighted_tardiness"
        else compute_total_tardiness
    )
    obj_name = "ΣwjTj" if objective == "weighted_tardiness" else "ΣTj"

    # ── Initialize population ────────────────────────────────────────────
    population: list[list[int]] = []
    fitnesses: list[float] = []

    # Seed from dispatching rules
    if objective == "weighted_tardiness" and instance.weights is not None:
        try:
            init_sol = atc(instance)
            population.append(list(init_sol.sequence))
            fitnesses.append(eval_fn(instance, init_sol.sequence))
        except Exception:
            pass

    edd_sol = edd(instance)
    population.append(list(edd_sol.sequence))
    fitnesses.append(eval_fn(instance, edd_sol.sequence))

    # Fill with random permutations
    while len(population) < pop_size:
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(perm)
        fitnesses.append(eval_fn(instance, perm))

    gbest_idx = int(np.argmin(fitnesses))
    gbest_seq = list(population[gbest_idx])
    gbest_obj = fitnesses[gbest_idx]

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
        if rng.random() < crossover_rate and n >= 2:
            child = _ox_crossover(
                population[p1_idx], population[p2_idx], rng,
            )
        else:
            child = list(population[p1_idx])

        # Mutation: swap two jobs
        if rng.random() < mutation_rate and n >= 2:
            i, j = rng.choice(n, size=2, replace=False)
            child[i], child[j] = child[j], child[i]

        child_obj = eval_fn(instance, child)

        # Replace worst
        worst_idx = int(np.argmax(fitnesses))
        if child_obj < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = child_obj

            if child_obj < gbest_obj:
                gbest_obj = child_obj
                gbest_seq = list(child)

    return SingleMachineSolution(
        sequence=gbest_seq,
        objective_value=gbest_obj,
        objective_name=obj_name,
    )


if __name__ == "__main__":
    inst = SingleMachineInstance.random(n=15, seed=42)
    print(f"Jobs: {inst.n}")

    sol_atc = atc(inst)
    print(f"ATC: ΣwjTj = {sol_atc.objective_value}")

    sol_ga = genetic_algorithm(inst, objective="weighted_tardiness", seed=42)
    print(f"GA:  ΣwjTj = {sol_ga.objective_value}")

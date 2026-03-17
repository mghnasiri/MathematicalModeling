"""
Genetic Algorithm for RCPSP (PS | prec | Cmax)

Implements a GA with activity-list encoding. Each chromosome is a
precedence-feasible permutation of activities. The Serial SGS decodes
each chromosome into a feasible schedule.

Uses tournament selection, two-point order crossover, and swap mutation
with precedence repair.

Complexity: O(generations * pop_size * n^2 * K).

Reference:
    Hartmann, S. (1998).
    "A competitive genetic algorithm for resource-constrained project scheduling."
    Naval Research Logistics, 45(7), 733-750.
    https://doi.org/10.1002/(SICI)1520-6750(199810)45:7<733::AID-NAV5>3.0.CO;2-C
"""

from __future__ import annotations
import sys
import os
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

_inst = _load_mod("rcpsp_instance_mod", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution

_sgs = _load_mod("rcpsp_serial_sgs_mod", os.path.join(_parent_dir, "heuristics", "serial_sgs.py"))
serial_sgs = _sgs.serial_sgs


def genetic_algorithm(
    instance: RCPSPInstance,
    pop_size: int = 50,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
    seed: int | None = None,
) -> RCPSPSolution:
    """
    Genetic Algorithm for RCPSP.

    Args:
        instance: RCPSP instance.
        pop_size: Population size.
        generations: Number of generations.
        crossover_rate: Crossover probability.
        mutation_rate: Mutation probability.
        seed: Random seed.

    Returns:
        Best RCPSPSolution found.
    """
    rng = np.random.default_rng(seed)
    total = instance.n + 2

    # Initialize population with random precedence-feasible permutations
    population = [
        _random_activity_list(instance, rng) for _ in range(pop_size)
    ]

    # Evaluate
    fitness = [_evaluate(instance, chromo) for chromo in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_chromo = list(population[best_idx])
    best_fitness = fitness[best_idx]

    for gen in range(generations):
        new_pop = [list(best_chromo)]  # Elitism

        while len(new_pop) < pop_size:
            p1 = _tournament(population, fitness, rng)
            p2 = _tournament(population, fitness, rng)

            if rng.random() < crossover_rate:
                c1, c2 = _crossover(instance, p1, p2, rng)
            else:
                c1, c2 = list(p1), list(p2)

            if rng.random() < mutation_rate:
                _mutate(instance, c1, rng)
            if rng.random() < mutation_rate:
                _mutate(instance, c2, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitness = [_evaluate(instance, chromo) for chromo in population]

        gen_best = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[gen_best] < best_fitness:
            best_fitness = fitness[gen_best]
            best_chromo = list(population[gen_best])

    # Decode best
    sol = serial_sgs(instance, priority_list=best_chromo)
    return sol


def _random_activity_list(
    instance: RCPSPInstance,
    rng: np.random.Generator,
) -> list[int]:
    """Generate a random precedence-feasible activity list."""
    total = instance.n + 2
    in_degree = {i: len(instance.predecessors.get(i, [])) for i in range(total)}
    eligible = [i for i in range(total) if in_degree[i] == 0]
    result = []

    while eligible:
        idx = rng.integers(0, len(eligible))
        act = eligible[idx]
        eligible.remove(act)
        result.append(act)
        for succ in instance.successors.get(act, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.append(succ)

    return result


def _evaluate(instance: RCPSPInstance, activity_list: list[int]) -> int:
    """Evaluate an activity list using Serial SGS."""
    sol = serial_sgs(instance, priority_list=activity_list)
    return sol.makespan


def _tournament(
    population: list[list[int]],
    fitness: list[int],
    rng: np.random.Generator,
    size: int = 3,
) -> list[int]:
    """Tournament selection."""
    indices = rng.choice(len(population), size=min(size, len(population)),
                         replace=False)
    best = min(indices, key=lambda i: fitness[i])
    return list(population[best])


def _crossover(
    instance: RCPSPInstance,
    p1: list[int],
    p2: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """
    Two-point order crossover with precedence repair.

    Takes a segment from p1, fills remaining from p2, then repairs
    any precedence violations.
    """
    n = len(p1)
    pts = sorted(rng.choice(n, size=2, replace=False))
    pt1, pt2 = int(pts[0]), int(pts[1])

    c1 = _ox_crossover(p1, p2, pt1, pt2)
    c2 = _ox_crossover(p2, p1, pt1, pt2)

    # Repair precedence feasibility
    c1 = _repair_precedence(instance, c1)
    c2 = _repair_precedence(instance, c2)

    return c1, c2


def _ox_crossover(p1: list[int], p2: list[int], pt1: int, pt2: int) -> list[int]:
    """Order crossover."""
    n = len(p1)
    child = [-1] * n
    child[pt1:pt2] = p1[pt1:pt2]
    used = set(child[pt1:pt2])

    pos = pt2
    for val in p2[pt2:] + p2[:pt2]:
        if val not in used:
            child[pos % n] = val
            used.add(val)
            pos += 1

    return child


def _repair_precedence(
    instance: RCPSPInstance,
    activity_list: list[int],
) -> list[int]:
    """Repair a permutation to be precedence-feasible."""
    total = instance.n + 2
    position = {act: i for i, act in enumerate(activity_list)}

    # Repeatedly fix violations
    changed = True
    while changed:
        changed = False
        for act in range(total):
            for pred in instance.predecessors.get(act, []):
                if position.get(pred, 0) > position.get(act, 0):
                    # Swap pred before act
                    i, j = position[pred], position[act]
                    activity_list[i], activity_list[j] = (
                        activity_list[j], activity_list[i]
                    )
                    position[activity_list[i]] = i
                    position[activity_list[j]] = j
                    changed = True

    return activity_list


def _mutate(
    instance: RCPSPInstance,
    activity_list: list[int],
    rng: np.random.Generator,
):
    """Swap two activities if it maintains precedence feasibility."""
    n = len(activity_list)
    for _ in range(3):  # Try a few times
        i, j = sorted(rng.choice(n, size=2, replace=False))
        # Swap and check
        activity_list[i], activity_list[j] = activity_list[j], activity_list[i]
        if _is_precedence_feasible(instance, activity_list):
            return
        # Undo
        activity_list[i], activity_list[j] = activity_list[j], activity_list[i]


def _is_precedence_feasible(
    instance: RCPSPInstance,
    activity_list: list[int],
) -> bool:
    """Check if an activity list respects precedence."""
    position = {act: i for i, act in enumerate(activity_list)}
    total = instance.n + 2
    for act in range(total):
        for pred in instance.predecessors.get(act, []):
            if position.get(pred, 0) > position.get(act, 0):
                return False
    return True


if __name__ == "__main__":
    print("=== GA for RCPSP ===\n")

    inst = RCPSPInstance.random(n=15, num_resources=2, seed=42)
    print(f"Critical path LB: {inst.critical_path_length()}")

    sol = genetic_algorithm(inst, pop_size=30, generations=100, seed=42)
    print(f"GA makespan: {sol.makespan}")

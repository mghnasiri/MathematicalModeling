"""
Generalized Assignment Problem (GAP) — Metaheuristics.

Algorithms:
    - Simulated Annealing with reassign and swap moves.

References:
    Osman, I.H. (1995). Heuristics for the generalized assignment problem:
    simulated annealing and tabu search approaches. OR Spektrum, 17(4),
    211-225. https://doi.org/10.1007/BF01720977
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("gap_instance_m", os.path.join(_this_dir, "instance.py"))
GAPInstance = _inst.GAPInstance
GAPSolution = _inst.GAPSolution

_heur = _load_mod("gap_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_ratio = _heur.greedy_ratio


def simulated_annealing(
    instance: GAPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> GAPSolution:
    """Simulated Annealing for GAP.

    Moves: reassign a job to a different agent, or swap two jobs
    between agents.

    Args:
        instance: GAP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        GAPSolution.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m

    # Initialize from greedy
    init = greedy_ratio(instance)
    assignment = list(init.assignment)
    cost = init.total_cost

    # Compute agent loads
    loads = np.zeros(m)
    for j in range(n):
        loads[assignment[j]] += instance.resource[assignment[j]][j]

    best_assignment = list(assignment)
    best_cost = cost

    # Penalty for infeasibility
    penalty_weight = best_cost * 0.5

    def objective(assign, ld):
        c = sum(instance.cost[assign[j]][j] for j in range(n))
        pen = sum(max(0.0, ld[i] - instance.capacity[i])
                  for i in range(m))
        return c + penalty_weight * pen

    obj = objective(assignment, loads)
    best_obj = obj

    temp = best_cost * 0.15
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        move = rng.integers(0, 2)

        if move == 0:
            # Reassign a random job to a different agent
            j = int(rng.integers(0, n))
            old_agent = assignment[j]
            new_agent = int(rng.integers(0, m - 1))
            if new_agent >= old_agent:
                new_agent += 1

            new_loads = loads.copy()
            new_loads[old_agent] -= instance.resource[old_agent][j]
            new_loads[new_agent] += instance.resource[new_agent][j]

            new_assign = list(assignment)
            new_assign[j] = new_agent
            new_obj = objective(new_assign, new_loads)
            delta = new_obj - obj

            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
                assignment = new_assign
                loads = new_loads
                obj = new_obj
                cost = sum(instance.cost[assignment[jj]][jj] for jj in range(n))

        else:
            # Swap two jobs between different agents
            j1 = int(rng.integers(0, n))
            j2 = int(rng.integers(0, n - 1))
            if j2 >= j1:
                j2 += 1
            if assignment[j1] == assignment[j2]:
                temp *= cooling_rate
                continue

            a1, a2 = assignment[j1], assignment[j2]
            new_loads = loads.copy()
            new_loads[a1] = loads[a1] - instance.resource[a1][j1] + instance.resource[a1][j2]
            new_loads[a2] = loads[a2] - instance.resource[a2][j2] + instance.resource[a2][j1]

            new_assign = list(assignment)
            new_assign[j1] = a2
            new_assign[j2] = a1
            new_obj = objective(new_assign, new_loads)
            delta = new_obj - obj

            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
                assignment = new_assign
                loads = new_loads
                obj = new_obj
                cost = sum(instance.cost[assignment[jj]][jj] for jj in range(n))

        # Update best (only if feasible)
        feasible = all(loads[i] <= instance.capacity[i] + 1e-6
                       for i in range(m))
        if feasible and cost < best_cost:
            best_cost = cost
            best_assignment = list(assignment)
            best_obj = obj

        temp *= cooling_rate

    return GAPSolution(assignment=best_assignment, total_cost=best_cost)


if __name__ == "__main__":
    from instance import small_gap_6x3

    inst = small_gap_6x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

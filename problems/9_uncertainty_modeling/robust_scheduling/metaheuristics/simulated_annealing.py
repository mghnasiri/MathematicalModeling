"""
Simulated Annealing for Min-Max Regret Single Machine Scheduling

Uses swap/insertion neighborhood to minimize maximum regret of ΣwjCj.

References:
    - Kouvelis, P. & Yu, G. (1997). Robust Discrete Optimization.
      Springer. https://doi.org/10.1007/978-1-4757-2620-6
"""
from __future__ import annotations

import sys
import os
import importlib.util
import math

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("rsched_instance", os.path.join(_base, "instance.py"))
_heur = _load_parent("rsched_heur", os.path.join(_base, "heuristics", "minmax_regret_heuristics.py"))

RobustSchedulingInstance = _inst.RobustSchedulingInstance
RobustSchedulingSolution = _inst.RobustSchedulingSolution


def simulated_annealing(
    instance: RobustSchedulingInstance,
    max_iterations: int = 5000,
    initial_temp: float = 50.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
) -> RobustSchedulingSolution:
    """SA for min-max regret ΣwjCj.

    Args:
        instance: RobustSchedulingInstance.
        max_iterations: Number of iterations.
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.

    Returns:
        Best RobustSchedulingSolution found.
    """
    rng = np.random.default_rng(seed)

    # Initialize with midpoint WSPT
    init_sol = _heur.midpoint_wspt(instance)
    current = list(init_sol.permutation)
    optimal_vals = instance._compute_optimal_twc()

    current_regret = instance.max_regret_twc(current, optimal_vals)
    best = list(current)
    best_regret = current_regret
    temp = initial_temp

    for _ in range(max_iterations):
        neighbor = list(current)

        if rng.random() < 0.5:
            # Swap two adjacent jobs
            i = rng.integers(len(neighbor) - 1)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        else:
            # Insertion: remove job, reinsert at random position
            i = rng.integers(len(neighbor))
            job = neighbor.pop(i)
            j = rng.integers(len(neighbor) + 1)
            neighbor.insert(j, job)

        n_regret = instance.max_regret_twc(neighbor, optimal_vals)
        delta = n_regret - current_regret

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current = neighbor
            current_regret = n_regret

            if n_regret < best_regret:
                best = list(neighbor)
                best_regret = n_regret

        temp *= cooling_rate

    scenario_vals = [
        instance.total_weighted_completion(best, s)
        for s in range(instance.n_scenarios)
    ]
    expected = float(np.dot(scenario_vals, instance.probabilities))

    return RobustSchedulingSolution(
        permutation=best,
        max_regret=best_regret,
        expected_twc=expected,
        scenario_values=scenario_vals,
    )

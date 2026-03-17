"""
Simulated Annealing for the Stochastic Knapsack Problem

Uses flip-bit neighborhood with penalty for infeasibility.
The objective combines value maximization with a penalty term
proportional to the probability of constraint violation.

Complexity: O(max_iter * S) per run.

References:
    - Dean, B.C., Goemans, M.X. & Vondrák, J. (2008). Approximating the
      stochastic knapsack problem. Math. Oper. Res., 33(1), 1-14.
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

_inst = _load_parent("sk_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
StochasticKnapsackInstance = _inst.StochasticKnapsackInstance
StochasticKnapsackSolution = _inst.StochasticKnapsackSolution


def simulated_annealing(
    instance: StochasticKnapsackInstance,
    alpha: float = 0.1,
    max_iterations: int = 5000,
    initial_temp: float = 50.0,
    cooling_rate: float = 0.995,
    penalty_weight: float = 100.0,
    seed: int = 42,
) -> StochasticKnapsackSolution:
    """SA for stochastic knapsack with chance constraint penalty.

    Objective: maximize value - penalty * max(0, alpha - P(infeasible))

    Args:
        instance: StochasticKnapsackInstance.
        alpha: Maximum violation probability.
        max_iterations: Number of SA iterations.
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor.
        penalty_weight: Penalty multiplier for infeasibility.
        seed: Random seed.

    Returns:
        Best StochasticKnapsackSolution found.
    """
    rng = np.random.default_rng(seed)

    # Initialize with greedy
    _greedy = _load_parent("sk_greedy", os.path.join(
        os.path.dirname(__file__), "..", "heuristics", "greedy_stochastic.py"))
    init_sol = _greedy.greedy_mean_weight(instance)

    current = init_sol.selection.copy()

    def evaluate(sel):
        val = instance.solution_value(sel)
        prob = instance.feasibility_probability(sel)
        violation = max(0, alpha - (1.0 - prob))
        return val - penalty_weight * violation * val

    current_obj = evaluate(current)
    best = current.copy()
    best_obj = current_obj
    temp = initial_temp

    for _ in range(max_iterations):
        # Flip a random bit
        i = rng.integers(instance.n)
        neighbor = current.copy()
        neighbor[i] = 1.0 - neighbor[i]

        neighbor_obj = evaluate(neighbor)
        delta = neighbor_obj - current_obj

        if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-10)):
            current = neighbor
            current_obj = neighbor_obj

            if current_obj > best_obj:
                feas = instance.feasibility_probability(current)
                if feas >= 1.0 - alpha - 1e-9:
                    best = current.copy()
                    best_obj = current_obj

        temp *= cooling_rate

    return StochasticKnapsackSolution(
        selection=best,
        total_value=instance.solution_value(best),
        feasibility_prob=instance.feasibility_probability(best),
        expected_weight=instance.expected_weight(best),
    )

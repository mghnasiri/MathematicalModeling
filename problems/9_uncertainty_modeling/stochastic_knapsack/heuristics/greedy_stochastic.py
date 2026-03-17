"""
Greedy Heuristics for the Stochastic Knapsack Problem

Two approaches:
    1. Mean-weight greedy: Use expected weights, apply standard value-density
       greedy from deterministic knapsack.
    2. Chance-constrained greedy: Only add items if P(feasible) >= 1-alpha
       after inclusion.

Complexity: O(n log n + n * S) — sorting + feasibility evaluation.

References:
    - Dean, B.C., Goemans, M.X. & Vondrák, J. (2008). Approximating the
      stochastic knapsack problem. Math. Oper. Res., 33(1), 1-14.
      https://doi.org/10.1287/moor.1070.0285
"""
from __future__ import annotations

import sys
import os
import importlib.util

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


def greedy_mean_weight(instance: StochasticKnapsackInstance) -> StochasticKnapsackSolution:
    """Greedy using expected weights and value-density ranking.

    Sort items by value / E[weight] descending, add items as long as
    expected total weight fits.

    Args:
        instance: StochasticKnapsackInstance.

    Returns:
        StochasticKnapsackSolution.
    """
    mean_w = instance.mean_weights
    density = instance.values / np.maximum(mean_w, 1e-9)
    order = np.argsort(-density)

    selection = np.zeros(instance.n, dtype=float)
    current_weight = 0.0

    for i in order:
        if current_weight + mean_w[i] <= instance.capacity + 1e-9:
            selection[i] = 1.0
            current_weight += mean_w[i]

    return StochasticKnapsackSolution(
        selection=selection,
        total_value=instance.solution_value(selection),
        feasibility_prob=instance.feasibility_probability(selection),
        expected_weight=instance.expected_weight(selection),
    )


def greedy_chance_constrained(
    instance: StochasticKnapsackInstance,
    alpha: float = 0.1,
) -> StochasticKnapsackSolution:
    """Chance-constrained greedy: maintain P(feasible) >= 1 - alpha.

    Sort items by value-density, add item only if inclusion keeps
    the feasibility probability above the threshold.

    Args:
        instance: StochasticKnapsackInstance.
        alpha: Maximum allowed infeasibility probability.

    Returns:
        StochasticKnapsackSolution with P(feasible) >= 1 - alpha.
    """
    mean_w = instance.mean_weights
    density = instance.values / np.maximum(mean_w, 1e-9)
    order = np.argsort(-density)

    selection = np.zeros(instance.n, dtype=float)

    for i in order:
        selection[i] = 1.0
        prob = instance.feasibility_probability(selection)
        if prob < 1.0 - alpha - 1e-9:
            selection[i] = 0.0  # undo

    return StochasticKnapsackSolution(
        selection=selection,
        total_value=instance.solution_value(selection),
        feasibility_prob=instance.feasibility_probability(selection),
        expected_weight=instance.expected_weight(selection),
    )

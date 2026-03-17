"""
Greedy Heuristics — Constructive heuristics for Multi-dimensional Knapsack.

Problem: Multi-dimensional Knapsack (MdKP)
Complexity: O(n log n + n * d) for greedy, O(n * d) for LP rounding

Greedy Aggregate Efficiency: Sort items by value divided by aggregate
resource consumption (sum of normalized weights). Greedily add items
that fit across all dimensions.

LP Relaxation Rounding: Solve the LP relaxation (fractional), then
round down and greedily add remaining items.

References:
    Fréville, A. (2004). The multidimensional 0-1 knapsack problem:
    An overview. European Journal of Operational Research, 155(1), 1-21.
    https://doi.org/10.1016/S0377-2217(03)00274-1

    Pirkul, H. (1987). A heuristic solution procedure for the multiconstraint
    zero-one knapsack problem. Naval Research Logistics, 34(2), 161-172.
    https://doi.org/10.1002/1520-6750(198704)34:2<161::AID-NAV3220340203>3.0.CO;2-A
"""

from __future__ import annotations

import os
import importlib.util
import sys

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mdkp_instance_gr", os.path.join(_parent_dir, "instance.py"))
MultidimKnapsackInstance = _inst.MultidimKnapsackInstance
MultidimKnapsackSolution = _inst.MultidimKnapsackSolution


def greedy_aggregate_efficiency(
    instance: MultidimKnapsackInstance,
) -> MultidimKnapsackSolution:
    """Solve MdKP using greedy aggregate efficiency.

    Compute aggregate efficiency for each item as:
        e_i = v_i / sum_k (w_{ik} / C_k)

    Sort by efficiency (descending) and greedily add items that
    remain feasible across all dimensions.

    Args:
        instance: A MultidimKnapsackInstance.

    Returns:
        MultidimKnapsackSolution with greedy selection.
    """
    n = instance.n
    d = instance.d

    # Compute aggregate efficiency
    normalized_weights = instance.weights / instance.capacities[np.newaxis, :]
    aggregate = normalized_weights.sum(axis=1)  # shape (n,)

    # Avoid division by zero
    efficiency = np.where(
        aggregate > 1e-10,
        instance.values / aggregate,
        instance.values * 1e10,
    )

    order = sorted(range(n), key=lambda i: efficiency[i], reverse=True)

    selected: list[int] = []
    remaining_cap = instance.capacities.copy()

    for idx in order:
        w = instance.weights[idx]
        if np.all(w <= remaining_cap + 1e-10):
            selected.append(idx)
            remaining_cap -= w

    total_value = float(np.sum(instance.values[selected])) if selected else 0.0

    return MultidimKnapsackSolution(items=selected, value=total_value)


def lp_relaxation_rounding(
    instance: MultidimKnapsackInstance,
) -> MultidimKnapsackSolution:
    """Solve MdKP using LP relaxation rounding.

    Solve the LP relaxation (allowing fractional variables), then
    fix items with x_i >= 1 - epsilon, and greedily add remaining
    items sorted by LP value descending.

    Uses scipy.optimize.linprog for the LP relaxation.

    Args:
        instance: A MultidimKnapsackInstance.

    Returns:
        MultidimKnapsackSolution with LP-rounded selection.
    """
    from scipy.optimize import linprog

    n = instance.n
    d = instance.d

    # LP: min -v^T x  s.t.  W^T x <= C, 0 <= x <= 1
    c = -instance.values

    # Capacity constraints: weights^T @ x <= capacities
    # linprog expects A_ub @ x <= b_ub
    A_ub = instance.weights.T  # shape (d, n)
    b_ub = instance.capacities

    bounds = [(0.0, 1.0) for _ in range(n)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        # Fallback to aggregate efficiency
        return greedy_aggregate_efficiency(instance)

    x_lp = result.x

    # Round: fix items with x >= 0.99, then greedily add rest
    selected: list[int] = []
    remaining_cap = instance.capacities.copy()

    # First pass: fix near-integer items
    fixed = set()
    order_by_lp = sorted(range(n), key=lambda i: x_lp[i], reverse=True)

    for idx in order_by_lp:
        if x_lp[idx] >= 0.99:
            w = instance.weights[idx]
            if np.all(w <= remaining_cap + 1e-10):
                selected.append(idx)
                remaining_cap -= w
                fixed.add(idx)

    # Second pass: greedily add remaining by LP value
    for idx in order_by_lp:
        if idx in fixed:
            continue
        w = instance.weights[idx]
        if np.all(w <= remaining_cap + 1e-10):
            selected.append(idx)
            remaining_cap -= w

    total_value = float(np.sum(instance.values[selected])) if selected else 0.0

    return MultidimKnapsackSolution(items=selected, value=total_value)


if __name__ == "__main__":
    _inst_mod = _load_mod("mdkp_inst_main", os.path.join(_parent_dir, "instance.py"))
    small_mdkp_5_2 = _inst_mod.small_mdkp_5_2
    medium_mdkp_8_3 = _inst_mod.medium_mdkp_8_3
    validate_solution = _inst_mod.validate_solution

    print("=== Multi-dimensional Knapsack Greedy Heuristics ===\n")

    for name, inst_fn in [("small_5_2", small_mdkp_5_2),
                           ("medium_8_3", medium_mdkp_8_3)]:
        inst = inst_fn()
        for algo_name, algo in [("aggregate", greedy_aggregate_efficiency),
                                 ("lp_round", lp_relaxation_rounding)]:
            sol = algo(inst)
            valid, errors = validate_solution(inst, sol)
            print(f"{name} {algo_name}: value={sol.value:.0f}, valid={valid}")

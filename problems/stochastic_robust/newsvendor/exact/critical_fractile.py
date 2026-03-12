"""
Critical Fractile Solution for the Newsvendor Problem

The optimal order quantity Q* satisfies P(D <= Q*) = c_u / (c_u + c_o),
where c_u = p - c (underage cost) and c_o = c - v (overage cost).

For discrete scenarios: sort demands, accumulate probabilities until
the critical fractile is reached.

Complexity: O(S log S) where S is the number of scenarios.

References:
    - Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy.
      Econometrica, 19(3), 250-272. https://doi.org/10.2307/1906813
"""
from __future__ import annotations

import sys
import os

import numpy as np

import importlib.util

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("nv_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
NewsvendorInstance = _inst.NewsvendorInstance
NewsvendorSolution = _inst.NewsvendorSolution


def critical_fractile(instance: NewsvendorInstance) -> NewsvendorSolution:
    """Solve newsvendor via the critical fractile method.

    Sort demand scenarios, accumulate probabilities until the cumulative
    probability reaches the critical fractile c_u/(c_u+c_o).

    Args:
        instance: NewsvendorInstance to solve.

    Returns:
        Optimal NewsvendorSolution.
    """
    cf = instance.critical_fractile

    # Sort scenarios by demand value
    order = np.argsort(instance.demand_scenarios)
    sorted_demands = instance.demand_scenarios[order]
    sorted_probs = instance.probabilities[order]

    # Find Q* where cumulative probability >= critical fractile
    cum_prob = np.cumsum(sorted_probs)
    idx = np.searchsorted(cum_prob, cf, side="left")
    idx = min(idx, len(sorted_demands) - 1)

    q_star = sorted_demands[idx]
    service_level = float(cum_prob[idx])

    return NewsvendorSolution(
        order_quantity=q_star,
        expected_cost=instance.expected_cost(q_star),
        expected_profit=instance.expected_profit(q_star),
        service_level=service_level,
    )


def grid_search(instance: NewsvendorInstance, n_points: int = 1000) -> NewsvendorSolution:
    """Solve newsvendor via brute-force grid search over demand range.

    Evaluates expected cost at evenly spaced points across the demand range.

    Args:
        instance: NewsvendorInstance to solve.
        n_points: Number of grid points.

    Returns:
        Best NewsvendorSolution found.
    """
    d_min = float(instance.demand_scenarios.min())
    d_max = float(instance.demand_scenarios.max())
    candidates = np.linspace(d_min, d_max, n_points)

    best_q = candidates[0]
    best_cost = instance.expected_cost(best_q)

    for q in candidates[1:]:
        cost = instance.expected_cost(q)
        if cost < best_cost:
            best_cost = cost
            best_q = q

    # Compute service level
    sl = float(np.dot(instance.demand_scenarios <= best_q, instance.probabilities))

    return NewsvendorSolution(
        order_quantity=best_q,
        expected_cost=best_cost,
        expected_profit=instance.expected_profit(best_q),
        service_level=sl,
    )


if __name__ == "__main__":
    inst = NewsvendorInstance(
        unit_cost=5.0,
        selling_price=10.0,
        salvage_value=2.0,
        demand_scenarios=np.array([40, 50, 60, 70, 80, 90, 100]),
    )
    print(f"Critical fractile: {inst.critical_fractile:.4f}")
    sol_cf = critical_fractile(inst)
    print(f"Critical fractile solution: {sol_cf}")
    sol_gs = grid_search(inst)
    print(f"Grid search solution: {sol_gs}")

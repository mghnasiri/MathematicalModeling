"""
Heuristics for Min-Max Regret Single Machine Scheduling

Three approaches:
    1. Midpoint WSPT: Use mean processing times, apply optimal WSPT rule.
    2. Scenario enumeration: Compute WSPT for each scenario, cross-evaluate.
    3. Worst-case WSPT: Use maximum processing times.

Complexity: O(S * n log n + S * n * S) for scenario enumeration.

References:
    - Kasperski, A. & Zielinski, P. (2008). A 2-approximation algorithm for
      interval data minmax regret sequencing problems. Oper. Res. Lett.,
      36(5), 561-564. https://doi.org/10.1016/j.orl.2008.07.004
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

_inst = _load_parent("rsched_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
RobustSchedulingInstance = _inst.RobustSchedulingInstance
RobustSchedulingSolution = _inst.RobustSchedulingSolution


def _wspt_order(processing_times: np.ndarray, weights: np.ndarray) -> list[int]:
    """WSPT (Weighted Shortest Processing Time) rule: sort by p_j/w_j."""
    ratios = processing_times / np.maximum(weights, 1e-9)
    return list(np.argsort(ratios))


def midpoint_wspt(instance: RobustSchedulingInstance) -> RobustSchedulingSolution:
    """WSPT rule using mean processing times.

    Args:
        instance: RobustSchedulingInstance.

    Returns:
        RobustSchedulingSolution.
    """
    mean_p = instance.mean_processing
    perm = _wspt_order(mean_p, instance.weights)

    optimal_vals = instance._compute_optimal_twc()
    max_regret = instance.max_regret_twc(perm, optimal_vals)
    scenario_vals = [
        instance.total_weighted_completion(perm, s)
        for s in range(instance.n_scenarios)
    ]
    expected = float(np.dot(scenario_vals, instance.probabilities))

    return RobustSchedulingSolution(
        permutation=perm,
        max_regret=max_regret,
        expected_twc=expected,
        scenario_values=scenario_vals,
    )


def scenario_enumeration(instance: RobustSchedulingInstance) -> RobustSchedulingSolution:
    """Generate WSPT schedule for each scenario, pick min max-regret.

    Args:
        instance: RobustSchedulingInstance.

    Returns:
        Best RobustSchedulingSolution from scenario-wise WSPT schedules.
    """
    S = instance.n_scenarios
    optimal_vals = instance._compute_optimal_twc()

    # Generate candidate permutations (WSPT for each scenario)
    candidates = []
    seen = set()
    for s in range(S):
        p_s = instance.processing_scenarios[s]
        perm = _wspt_order(p_s, instance.weights)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            candidates.append(perm)

    # Also add midpoint WSPT
    mean_p = instance.mean_processing
    perm_mid = _wspt_order(mean_p, instance.weights)
    key_mid = tuple(perm_mid)
    if key_mid not in seen:
        candidates.append(perm_mid)

    best_perm = candidates[0]
    best_regret = float("inf")
    best_vals: list[float] = []

    for perm in candidates:
        regret = instance.max_regret_twc(perm, optimal_vals)
        if regret < best_regret:
            best_regret = regret
            best_perm = perm
            best_vals = [
                instance.total_weighted_completion(perm, s) for s in range(S)
            ]

    expected = float(np.dot(best_vals, instance.probabilities))

    return RobustSchedulingSolution(
        permutation=best_perm,
        max_regret=best_regret,
        expected_twc=expected,
        scenario_values=best_vals,
    )


def worst_case_wspt(instance: RobustSchedulingInstance) -> RobustSchedulingSolution:
    """WSPT using maximum processing times across scenarios.

    Conservative approach: assumes worst-case processing times.

    Args:
        instance: RobustSchedulingInstance.

    Returns:
        RobustSchedulingSolution.
    """
    max_p = instance.processing_scenarios.max(axis=0)
    perm = _wspt_order(max_p, instance.weights)

    optimal_vals = instance._compute_optimal_twc()
    max_regret = instance.max_regret_twc(perm, optimal_vals)
    scenario_vals = [
        instance.total_weighted_completion(perm, s)
        for s in range(instance.n_scenarios)
    ]
    expected = float(np.dot(scenario_vals, instance.probabilities))

    return RobustSchedulingSolution(
        permutation=perm,
        max_regret=max_regret,
        expected_twc=expected,
        scenario_values=scenario_vals,
    )

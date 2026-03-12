"""
Moment-Based DRO Heuristic

Restricts the ambiguity set to distributions matching the first
moment (mean) within a tolerance. The worst case over such a set
can be approximated by tilting the empirical distribution towards
support points that increase cost.

Complexity: O(K * n) per decision evaluation.

References:
    - Delage, E. & Ye, Y. (2010). Distributionally robust optimization
      under moment uncertainty. Oper. Res., 58(3), 595-612.
      https://doi.org/10.1287/opre.1090.0741
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import linprog

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("dro_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
DROInstance = _inst.DROInstance
DROSolution = _inst.DROSolution


def worst_case_distribution(instance: DROInstance,
                             x: np.ndarray,
                             mean_tol: float = 1.0) -> tuple[float, np.ndarray]:
    """Find worst-case distribution over support matching mean constraint.

    Solve: max_p sum_k p_k * cost(x, xi_k)
           s.t. sum_k p_k = 1, p_k >= 0
                ||sum_k p_k * xi_k - mu_hat||_inf <= mean_tol

    Args:
        instance: DROInstance.
        x: Decision vector.
        mean_tol: Tolerance on mean matching.

    Returns:
        (worst_case_cost, worst_case_probs).
    """
    K = instance.n_support
    n = instance.n

    # Maximize E_P[cost] = sum p_k * cost_k
    costs = np.array([instance.cost(x, xi) for xi in instance.support_points])

    # LP: max costs^T p  => min -costs^T p
    # s.t. sum p = 1, p >= 0
    #      sum_k p_k * xi_{k,d} <= mu_d + tol   for each d
    #      sum_k p_k * xi_{k,d} >= mu_d - tol   for each d

    mean_target = (instance.mean_target if instance.mean_target is not None
                   else np.dot(instance.nominal_probs, instance.support_points))

    A_eq = np.ones((1, K))
    b_eq = np.array([1.0])

    A_ub_rows = []
    b_ub_list = []
    for d in range(n):
        # sum p_k * xi_{k,d} <= mu_d + tol
        row = instance.support_points[:, d]
        A_ub_rows.append(row)
        b_ub_list.append(mean_target[d] + mean_tol)

        # -sum p_k * xi_{k,d} <= -mu_d + tol  (i.e., sum >= mu_d - tol)
        A_ub_rows.append(-row)
        b_ub_list.append(-mean_target[d] + mean_tol)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_list)

    result = linprog(-costs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, 1)] * K, method="highs")

    if result.success:
        wc_probs = result.x
        wc_cost = float(np.dot(costs, wc_probs))
        return wc_cost, wc_probs
    else:
        # Fallback to nominal
        return float(np.dot(costs, instance.nominal_probs)), instance.nominal_probs


def solve_moment_dro(instance: DROInstance,
                      mean_tol: float = 1.0) -> DROSolution:
    """Solve DRO with moment-based ambiguity via grid search.

    Since the decision is constrained to a box [0,1]^n, we discretize
    and evaluate worst-case cost for each candidate.

    For small n, this is practical. For larger n, use the Wasserstein
    LP reformulation instead.

    Args:
        instance: DROInstance.
        mean_tol: Mean constraint tolerance.

    Returns:
        DROSolution.
    """
    n = instance.n

    # For small n, grid search over vertices of [0,1]^n
    if n <= 8:
        n_candidates = min(2 ** n, 256)
        best_x = np.zeros(n)
        best_wc = float("inf")
        best_probs = None

        for i in range(n_candidates):
            x = np.array([(i >> d) & 1 for d in range(n)], dtype=float)
            # Check feasibility
            if instance.A_ub is not None:
                if np.any(instance.A_ub @ x > instance.b_ub + 1e-9):
                    continue

            wc_cost, wc_probs = worst_case_distribution(instance, x, mean_tol)
            if wc_cost < best_wc:
                best_wc = wc_cost
                best_x = x.copy()
                best_probs = wc_probs
    else:
        # Fallback: use nominal solution as starting point
        from scipy.optimize import linprog as _lp
        mean_xi = np.dot(instance.nominal_probs, instance.support_points)
        eff_c = instance.c + mean_xi
        result = _lp(eff_c, A_ub=instance.A_ub, b_ub=instance.b_ub,
                      bounds=[(0, 1)] * n, method="highs")
        best_x = result.x if result.success else np.zeros(n)
        best_wc, best_probs = worst_case_distribution(instance, best_x, mean_tol)

    return DROSolution(
        x=best_x,
        worst_case_cost=best_wc,
        nominal_cost=instance.nominal_expected_cost(best_x),
        worst_case_probs=best_probs,
    )

"""
Wasserstein DRO for Linear Cost Functions

The Wasserstein DRO problem with a linear cost f(x, xi) = (c+xi)^T x
and a type-1 Wasserstein ball of radius epsilon around the empirical
distribution can be reformulated as:

    min_x  max_{||xi||_inf <= epsilon}  (c + xi)^T x
    s.t. feasibility constraints on x

For the inner maximization, the worst-case distribution places mass
on the support point that maximizes cost, shifted within epsilon.

The tractable reformulation:
    min_x  c^T x + epsilon * ||x||_1
    s.t. constraints

This is equivalent to a linear program.

Complexity: Polynomial (LP reformulation).

References:
    - Esfahani, P.M. & Kuhn, D. (2018). Data-driven distributionally robust
      optimization using the Wasserstein metric. Math. Program., 171, 115-166.
      https://doi.org/10.1007/s10107-017-1172-1
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


def solve_wasserstein_dro(instance: DROInstance) -> DROSolution | None:
    """Solve DRO with Wasserstein ambiguity set via LP reformulation.

    The worst-case expected cost over a Wasserstein ball is:
        sup_{P: W(P, P_hat) <= eps} E_P[(c+xi)^T x]
        = E_{P_hat}[(c+xi)^T x] + eps * ||x||_inf

    We reformulate to minimize worst-case cost:
        min_x  max_k (c + xi_k)^T x + eps * ||x||_1

    Using LP variables: x, t (for L1 norm), s.t. t_i >= x_i, t_i >= -x_i.

    Args:
        instance: DROInstance.

    Returns:
        DROSolution or None if infeasible.
    """
    n = instance.n
    K = instance.n_support
    eps = instance.wasserstein_radius

    # Variables: [x (n), t (n)] where t_i >= |x_i|
    total_vars = 2 * n

    # For each support point k, cost_k(x) = (c + xi_k)^T x
    # Worst-case objective: max_k cost_k(x) + eps * sum(t)
    # Introduce auxiliary z >= max_k cost_k(x)
    # Variables: [x (n), t (n), z (1)]
    total_vars = 2 * n + 1

    # Objective: min z + eps * sum(t)
    obj = np.zeros(total_vars)
    obj[n:2*n] = eps  # t coefficients
    obj[2*n] = 1.0    # z coefficient

    A_ub_rows = []
    b_ub_list = []

    # Constraint: (c + xi_k)^T x <= z for each k
    for k in range(K):
        row = np.zeros(total_vars)
        row[:n] = instance.c + instance.support_points[k]
        row[2*n] = -1.0
        A_ub_rows.append(row)
        b_ub_list.append(0.0)

    # Constraints: t_i >= x_i  =>  x_i - t_i <= 0
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = 1.0
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_list.append(0.0)

    # Constraints: t_i >= -x_i  =>  -x_i - t_i <= 0
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_list.append(0.0)

    # Original constraints on x
    if instance.A_ub is not None:
        m_orig = instance.A_ub.shape[0]
        for r in range(m_orig):
            row = np.zeros(total_vars)
            row[:n] = instance.A_ub[r]
            A_ub_rows.append(row)
            b_ub_list.append(instance.b_ub[r])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    bounds = [(None, None)] * n + [(0, None)] * n + [(None, None)]

    result = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return None

    x = result.x[:n]
    wc_cost = result.fun

    # Find worst-case distribution
    costs_per_k = np.array([instance.cost(x, xi) for xi in instance.support_points])
    worst_k = np.argmax(costs_per_k)
    wc_probs = np.zeros(K)
    wc_probs[worst_k] = 1.0

    return DROSolution(
        x=x,
        worst_case_cost=wc_cost,
        nominal_cost=instance.nominal_expected_cost(x),
        worst_case_probs=wc_probs,
    )


def solve_nominal(instance: DROInstance) -> DROSolution | None:
    """Solve with nominal distribution (no robustness).

    min_x  E_{P_hat}[(c + xi)^T x] = (c + E[xi])^T x

    Args:
        instance: DROInstance.

    Returns:
        DROSolution under nominal distribution.
    """
    n = instance.n
    mean_xi = np.dot(instance.nominal_probs, instance.support_points)
    effective_c = instance.c + mean_xi

    bounds = [(None, None)] * n
    A_ub = instance.A_ub
    b_ub = instance.b_ub

    result = linprog(effective_c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return None

    x = result.x
    nominal_cost = instance.nominal_expected_cost(x)

    # Compute worst-case cost by finding worst distribution
    costs_per_k = np.array([instance.cost(x, xi) for xi in instance.support_points])
    wc_cost = float(costs_per_k.max())

    return DROSolution(
        x=x,
        worst_case_cost=wc_cost,
        nominal_cost=nominal_cost,
    )

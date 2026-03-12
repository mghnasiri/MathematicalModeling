"""
ILP Formulation for Set Covering via SciPy HiGHS.

    min  c^T x
    s.t. A x >= 1   (coverage constraints)
         x in {0,1}^n

Relaxed to LP for lower bound; rounded for feasible solution.

Complexity: NP-hard in general; LP relaxation is polynomial.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("scp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
SetCoveringInstance = _inst.SetCoveringInstance
SetCoveringSolution = _inst.SetCoveringSolution


def solve_lp_relaxation(instance: SetCoveringInstance) -> tuple[float, np.ndarray]:
    """Solve LP relaxation for lower bound.

    Args:
        instance: SetCoveringInstance.

    Returns:
        (lower_bound, fractional_solution).
    """
    m, n = instance.m, instance.n
    A = np.zeros((m, n))
    for j in range(n):
        for i in instance.subsets[j]:
            A[i, j] = 1.0

    result = linprog(instance.costs, A_ub=-A, b_ub=-np.ones(m),
                     bounds=[(0, 1)] * n, method="highs")

    if result.success:
        return result.fun, result.x
    return 0.0, np.zeros(n)


def solve_ilp(instance: SetCoveringInstance) -> SetCoveringSolution | None:
    """Solve SCP exactly via MILP (SciPy HiGHS).

    Args:
        instance: SetCoveringInstance.

    Returns:
        Optimal SetCoveringSolution or None.
    """
    m, n = instance.m, instance.n
    A = np.zeros((m, n))
    for j in range(n):
        for i in instance.subsets[j]:
            A[i, j] = 1.0

    try:
        constraints = LinearConstraint(A, lb=np.ones(m))
        integrality = np.ones(n)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        result = milp(instance.costs, constraints=constraints,
                      integrality=integrality, bounds=bounds)

        if result.success:
            x = np.round(result.x).astype(int)
            selected = [j for j in range(n) if x[j] == 1]
            return SetCoveringSolution(
                selected=selected,
                total_cost=instance.total_cost(selected),
                n_selected=len(selected),
            )
    except Exception:
        pass

    return None

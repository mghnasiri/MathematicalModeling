"""QP Solver using SciPy minimize (SLSQP / trust-constr).

Solves convex QP: min (1/2) x^T Q x + c^T x subject to constraints
using scipy.optimize.minimize with SLSQP method.

Complexity: Polynomial for convex QP (interior point / active set).

References:
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
    Springer.
    Kraft, D. (1988). A software package for sequential quadratic programming.
    DFVLR-FB 88-28.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "qp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
QPInstance = _inst.QPInstance
QPSolution = _inst.QPSolution


def solve_qp_slsqp(instance: QPInstance) -> QPSolution:
    """Solve QP using SciPy SLSQP method.

    Args:
        instance: A QPInstance.

    Returns:
        A QPSolution.
    """
    def objective(x):
        return 0.5 * x @ instance.Q @ x + instance.c @ x

    def gradient(x):
        return instance.Q @ x + instance.c

    constraints = []
    if instance.A_ub is not None and instance.b_ub is not None:
        for i in range(len(instance.b_ub)):
            constraints.append({
                "type": "ineq",
                "fun": lambda x, i=i: float(instance.b_ub[i]
                                             - instance.A_ub[i] @ x),
            })
    if instance.A_eq is not None and instance.b_eq is not None:
        for i in range(len(instance.b_eq)):
            constraints.append({
                "type": "eq",
                "fun": lambda x, i=i: float(instance.A_eq[i] @ x
                                             - instance.b_eq[i]),
            })

    bounds = instance.bounds
    x0 = np.zeros(instance.n)

    result = minimize(objective, x0, method="SLSQP", jac=gradient,
                      constraints=constraints, bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-12})

    return QPSolution(
        x=result.x,
        objective=float(result.fun),
        success=result.success,
        message=result.message,
    )


def solve_qp_trust(instance: QPInstance) -> QPSolution:
    """Solve QP using SciPy trust-constr method.

    Args:
        instance: A QPInstance.

    Returns:
        A QPSolution.
    """
    def objective(x):
        return 0.5 * x @ instance.Q @ x + instance.c @ x

    def gradient(x):
        return instance.Q @ x + instance.c

    def hessian(x):
        return instance.Q

    constraints = []
    if instance.A_ub is not None and instance.b_ub is not None:
        constraints.append(LinearConstraint(
            instance.A_ub, -np.inf, instance.b_ub
        ))
    if instance.A_eq is not None and instance.b_eq is not None:
        constraints.append(LinearConstraint(
            instance.A_eq, instance.b_eq, instance.b_eq
        ))

    if instance.bounds is not None:
        lb = np.array([b[0] if b[0] is not None else -np.inf
                       for b in instance.bounds])
        ub = np.array([b[1] if b[1] is not None else np.inf
                       for b in instance.bounds])
        scipy_bounds = Bounds(lb, ub)
    else:
        scipy_bounds = None

    x0 = np.zeros(instance.n)

    result = minimize(objective, x0, method="trust-constr",
                      jac=gradient, hess=hessian,
                      constraints=constraints, bounds=scipy_bounds,
                      options={"maxiter": 1000})

    return QPSolution(
        x=result.x,
        objective=float(result.fun),
        success=result.success,
        message=result.message if hasattr(result, 'message') else str(result.status),
    )


if __name__ == "__main__":
    inst = QPInstance.random(n=5, m_ub=3, seed=42)
    print(f"Instance: {inst.n} variables, Q shape={inst.Q.shape}")

    sol1 = solve_qp_slsqp(inst)
    print(f"\nSLSQP: {sol1}")
    print(f"  x = {sol1.x}")

    sol2 = solve_qp_trust(inst)
    print(f"\nTrust-constr: {sol2}")
    print(f"  x = {sol2.x}")

    print(f"\nObjective difference: {abs(sol1.objective - sol2.objective):.2e}")

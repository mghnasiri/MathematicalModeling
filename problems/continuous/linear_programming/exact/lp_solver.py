"""LP Solver with Sensitivity Analysis using SciPy linprog.

Solves LP using HiGHS backend and extracts sensitivity information:
shadow prices (dual variables), reduced costs, and binding constraints.

Complexity: Polynomial (interior point) or exponential worst-case (simplex).

References:
    Dantzig, G. B. (1963). Linear Programming and Extensions. Princeton
    University Press.
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


_inst = _load_parent(
    "lp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
LPInstance = _inst.LPInstance
LPSolution = _inst.LPSolution


def solve_lp(instance: LPInstance, method: str = "highs") -> LPSolution:
    """Solve LP using scipy.optimize.linprog.

    Args:
        instance: An LPInstance.
        method: Solver method ("highs", "highs-ds", "highs-ipm").

    Returns:
        An LPSolution with sensitivity information.
    """
    result = linprog(
        c=instance.c,
        A_ub=instance.A_ub,
        b_ub=instance.b_ub,
        A_eq=instance.A_eq,
        b_eq=instance.b_eq,
        bounds=instance.bounds,
        method=method,
    )

    shadow_prices = None
    reduced_costs = None

    if result.success:
        # Extract sensitivity info
        # Shadow prices from ineqlin dual
        if hasattr(result, "ineqlin") and result.ineqlin is not None:
            shadow_prices = np.array(result.ineqlin.marginals)
        elif instance.A_ub is not None:
            # Compute shadow prices via perturbation
            shadow_prices = _compute_shadow_prices(instance, method)

        # Reduced costs
        if hasattr(result, "x") and result.x is not None:
            reduced_costs = _compute_reduced_costs(instance, result.x, method)

    return LPSolution(
        x=result.x if result.x is not None else np.zeros(instance.n),
        objective=float(result.fun) if result.fun is not None else float("inf"),
        shadow_prices=shadow_prices,
        reduced_costs=reduced_costs,
        success=result.success,
        message=result.message,
    )


def _compute_shadow_prices(instance: LPInstance,
                            method: str = "highs") -> np.ndarray | None:
    """Compute shadow prices via RHS perturbation.

    Args:
        instance: An LPInstance.
        method: Solver method.

    Returns:
        Array of shadow prices or None.
    """
    if instance.A_ub is None or instance.b_ub is None:
        return None

    base = linprog(c=instance.c, A_ub=instance.A_ub, b_ub=instance.b_ub,
                   A_eq=instance.A_eq, b_eq=instance.b_eq,
                   bounds=instance.bounds, method=method)
    if not base.success:
        return None

    m = len(instance.b_ub)
    shadow = np.zeros(m)
    delta = 1e-5

    for i in range(m):
        b_pert = instance.b_ub.copy()
        b_pert[i] += delta
        pert = linprog(c=instance.c, A_ub=instance.A_ub, b_ub=b_pert,
                       A_eq=instance.A_eq, b_eq=instance.b_eq,
                       bounds=instance.bounds, method=method)
        if pert.success:
            shadow[i] = (pert.fun - base.fun) / delta

    return shadow


def _compute_reduced_costs(instance: LPInstance, x_opt: np.ndarray,
                            method: str = "highs") -> np.ndarray:
    """Compute reduced costs via objective perturbation.

    Args:
        instance: An LPInstance.
        x_opt: Optimal solution.
        method: Solver method.

    Returns:
        Array of reduced costs.
    """
    base_obj = instance.objective(x_opt)
    rc = np.zeros(instance.n)
    delta = 1e-5

    for j in range(instance.n):
        c_pert = instance.c.copy()
        c_pert[j] += delta
        pert = linprog(c=c_pert, A_ub=instance.A_ub, b_ub=instance.b_ub,
                       A_eq=instance.A_eq, b_eq=instance.b_eq,
                       bounds=instance.bounds, method=method)
        if pert.success:
            rc[j] = (pert.fun - base_obj) / delta

    return rc


def sensitivity_report(instance: LPInstance,
                       solution: LPSolution) -> dict[str, object]:
    """Generate a sensitivity analysis report.

    Args:
        instance: An LPInstance.
        solution: An LPSolution with sensitivity info.

    Returns:
        Dictionary with binding constraints, shadow prices, reduced costs.
    """
    report: dict[str, object] = {
        "optimal_objective": solution.objective,
        "optimal_x": solution.x.tolist(),
    }

    # Identify binding constraints
    if instance.A_ub is not None and instance.b_ub is not None:
        slack = instance.b_ub - instance.A_ub @ solution.x
        binding = [i for i in range(len(slack)) if abs(slack[i]) < 1e-6]
        report["binding_constraints"] = binding
        report["slack"] = slack.tolist()

    if solution.shadow_prices is not None:
        report["shadow_prices"] = solution.shadow_prices.tolist()

    if solution.reduced_costs is not None:
        report["reduced_costs"] = solution.reduced_costs.tolist()

    return report


if __name__ == "__main__":
    # Classic production problem
    # max 5x1 + 4x2  =>  min -5x1 - 4x2
    # s.t. 6x1 + 4x2 <= 24
    #      x1 + 2x2 <= 6
    #      x1, x2 >= 0
    inst = LPInstance(
        n=2,
        c=np.array([-5.0, -4.0]),
        A_ub=np.array([[6.0, 4.0], [1.0, 2.0]]),
        b_ub=np.array([24.0, 6.0]),
        bounds=[(0, None), (0, None)],
    )

    sol = solve_lp(inst)
    print(f"Solution: {sol}")
    print(f"x = {sol.x}")

    report = sensitivity_report(inst, sol)
    print(f"\nSensitivity Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

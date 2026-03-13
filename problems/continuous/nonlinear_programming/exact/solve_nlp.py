"""Nonlinear Programming solver using scipy.optimize.minimize.

Supports unconstrained (L-BFGS-B) and constrained (SLSQP) optimization.

References:
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization. Springer.
    Kraft, D. (1988). A software package for sequential quadratic programming.
    Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center.
"""
import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import minimize

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "nlp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

NLPInstance = _inst.NLPInstance
NLPSolution = _inst.NLPSolution


def solve_nlp(instance: NLPInstance, method: str | None = None,
              maxiter: int = 1000, tol: float = 1e-10) -> NLPSolution:
    """Solve a nonlinear programming problem.

    Automatically selects SLSQP for constrained problems and L-BFGS-B
    for unconstrained (or bounds-only) problems.

    Args:
        instance: An NLPInstance.
        method: Solver method ('SLSQP', 'L-BFGS-B', etc.). Auto-selected if None.
        maxiter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        NLPSolution with optimal x and objective value.
    """
    has_constraints = bool(instance.ineq_constraints or instance.eq_constraints)

    if method is None:
        method = 'SLSQP' if has_constraints else 'L-BFGS-B'

    # Build constraints for scipy
    constraints = []
    for g in instance.ineq_constraints:
        constraints.append({'type': 'ineq', 'fun': lambda x, g=g: -g(x)})
    for h in instance.eq_constraints:
        constraints.append({'type': 'eq', 'fun': h})

    options = {'maxiter': maxiter}
    if method == 'L-BFGS-B':
        options['ftol'] = tol
    elif method == 'SLSQP':
        options['ftol'] = tol

    result = minimize(
        fun=instance.objective,
        x0=instance.x0,
        method=method,
        jac=instance.gradient,
        bounds=instance.bounds,
        constraints=constraints if constraints else (),
        options=options,
        tol=tol
    )

    return NLPSolution(
        x=result.x,
        objective_value=float(result.fun),
        success=bool(result.success),
        method=method,
        n_iterations=int(result.get('nit', 0)),
        message=str(result.message)
    )


def solve_multistart(instance: NLPInstance, n_starts: int = 5,
                     seed: int = 42, **kwargs) -> NLPSolution:
    """Solve NLP with multiple random starting points.

    Args:
        instance: An NLPInstance.
        n_starts: Number of random starts.
        seed: Random seed.
        **kwargs: Additional arguments passed to solve_nlp.

    Returns:
        Best NLPSolution found across all starts.
    """
    rng = np.random.default_rng(seed)
    best_sol = solve_nlp(instance, **kwargs)

    for _ in range(n_starts - 1):
        # Random perturbation of initial guess
        perturbed_x0 = instance.x0 + rng.normal(0, 2.0, size=instance.n_vars)

        # Apply bounds if present
        if instance.bounds:
            for i, (lb, ub) in enumerate(instance.bounds):
                if lb is not None:
                    perturbed_x0[i] = max(perturbed_x0[i], lb)
                if ub is not None:
                    perturbed_x0[i] = min(perturbed_x0[i], ub)

        # Create modified instance
        mod_instance = NLPInstance(
            objective=instance.objective,
            n_vars=instance.n_vars,
            x0=perturbed_x0,
            bounds=instance.bounds,
            ineq_constraints=instance.ineq_constraints,
            eq_constraints=instance.eq_constraints,
            gradient=instance.gradient,
            name=instance.name
        )
        sol = solve_nlp(mod_instance, **kwargs)
        if sol.success and sol.objective_value < best_sol.objective_value:
            best_sol = sol

    return best_sol


if __name__ == "__main__":
    # Rosenbrock
    inst = NLPInstance.rosenbrock(2)
    sol = solve_nlp(inst)
    print(f"Rosenbrock: {sol}")
    print(f"  x = {sol.x}")

    # Sphere
    inst = NLPInstance.sphere(3)
    sol = solve_nlp(inst)
    print(f"\nSphere: {sol}")
    print(f"  x = {sol.x}")

    # Constrained
    inst = NLPInstance.constrained_quadratic()
    sol = solve_nlp(inst)
    print(f"\nConstrained quadratic: {sol}")
    print(f"  x = {sol.x}")

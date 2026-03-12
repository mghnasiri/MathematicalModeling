"""Tests for Nonlinear Programming solver."""
import sys
import os
import importlib.util

import numpy as np
import pytest

def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")

_instance_mod = _load_mod("nlp_inst", os.path.join(_base, "instance.py"))
_solver_mod = _load_mod("nlp_solver", os.path.join(_base, "exact", "solve_nlp.py"))

NLPInstance = _instance_mod.NLPInstance
NLPSolution = _instance_mod.NLPSolution
solve_nlp = _solver_mod.solve_nlp
solve_multistart = _solver_mod.solve_multistart


class TestNLPInstance:
    """Tests for NLPInstance."""

    def test_rosenbrock_creation(self):
        inst = NLPInstance.rosenbrock(2)
        assert inst.n_vars == 2
        assert inst.name == "Rosenbrock"

    def test_sphere_creation(self):
        inst = NLPInstance.sphere(3)
        assert inst.n_vars == 3

    def test_constrained_creation(self):
        inst = NLPInstance.constrained_quadratic()
        assert inst.n_vars == 2
        assert len(inst.ineq_constraints) == 1


class TestSolveNLP:
    """Tests for solve_nlp."""

    def test_rosenbrock_2d(self):
        inst = NLPInstance.rosenbrock(2)
        sol = solve_nlp(inst)
        assert sol.success
        assert sol.objective_value < 1e-6
        np.testing.assert_allclose(sol.x, [1.0, 1.0], atol=1e-4)

    def test_sphere_3d(self):
        inst = NLPInstance.sphere(3)
        sol = solve_nlp(inst)
        assert sol.success
        assert sol.objective_value < 1e-8
        np.testing.assert_allclose(sol.x, [0.0, 0.0, 0.0], atol=1e-4)

    def test_constrained_quadratic(self):
        inst = NLPInstance.constrained_quadratic()
        sol = solve_nlp(inst)
        assert sol.success
        np.testing.assert_allclose(sol.x, [1.0, 2.0], atol=1e-4)

    def test_constrained_active_constraint(self):
        """Constraint forces solution away from unconstrained optimum."""
        def obj(x):
            return float((x[0] - 3)**2 + (x[1] - 3)**2)
        def ineq(x):
            return float(x[0] + x[1] - 4)  # x + y <= 4

        inst = NLPInstance(
            objective=obj, n_vars=2,
            x0=np.array([1.0, 1.0]),
            bounds=[(0.0, None), (0.0, None)],
            ineq_constraints=[ineq]
        )
        sol = solve_nlp(inst)
        assert sol.success
        # Optimal should be at x+y=4, x=y=2
        np.testing.assert_allclose(sol.x, [2.0, 2.0], atol=1e-3)
        assert sol.x[0] + sol.x[1] <= 4.0 + 1e-6

    def test_bounds_respected(self):
        inst = NLPInstance(
            objective=lambda x: float((x[0] - 5)**2),
            n_vars=1,
            x0=np.array([0.0]),
            bounds=[(0.0, 3.0)]
        )
        sol = solve_nlp(inst)
        assert sol.success
        assert sol.x[0] <= 3.0 + 1e-6
        assert sol.x[0] >= 0.0 - 1e-6

    def test_method_selection_unconstrained(self):
        inst = NLPInstance.sphere(2)
        sol = solve_nlp(inst)
        assert sol.method == 'L-BFGS-B'

    def test_method_selection_constrained(self):
        inst = NLPInstance.constrained_quadratic()
        sol = solve_nlp(inst)
        assert sol.method == 'SLSQP'

    def test_multistart(self):
        inst = NLPInstance.rosenbrock(2)
        sol = solve_multistart(inst, n_starts=3)
        assert sol.success
        assert sol.objective_value < 1e-4

    def test_solution_repr(self):
        sol = NLPSolution(
            x=np.array([1.0]), objective_value=0.0,
            success=True, method="SLSQP", n_iterations=10, message="ok"
        )
        assert "success=True" in repr(sol)

    def test_equality_constraint(self):
        """min x^2 + y^2 s.t. x + y = 1."""
        def obj(x):
            return float(x[0]**2 + x[1]**2)
        def eq(x):
            return float(x[0] + x[1] - 1)

        inst = NLPInstance(
            objective=obj, n_vars=2,
            x0=np.array([0.0, 0.0]),
            eq_constraints=[eq]
        )
        sol = solve_nlp(inst)
        assert sol.success
        np.testing.assert_allclose(sol.x, [0.5, 0.5], atol=1e-4)

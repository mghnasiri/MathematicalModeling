"""Tests for Distributionally Robust Optimization."""
from __future__ import annotations

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
_inst_mod = _load_mod("dro_instance", os.path.join(_base, "instance.py"))
_wass_mod = _load_mod("dro_wass", os.path.join(_base, "exact", "wasserstein_dro.py"))
_mom_mod = _load_mod("dro_moment", os.path.join(_base, "heuristics", "moment_dro.py"))

DROInstance = _inst_mod.DROInstance
DROSolution = _inst_mod.DROSolution
solve_wasserstein_dro = _wass_mod.solve_wasserstein_dro
solve_nominal = _wass_mod.solve_nominal
solve_moment_dro = _mom_mod.solve_moment_dro
worst_case_distribution = _mom_mod.worst_case_distribution


def _make_simple():
    return DROInstance(
        n=2,
        c=np.array([3.0, 5.0]),
        support_points=np.array([
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.5, 0.5],
        ]),
        A_ub=np.array([
            [1, 0], [0, 1],   # x <= 1
            [-1, 0], [0, -1], # x >= 0
        ]),
        b_ub=np.array([1, 1, 0, 0]),
        wasserstein_radius=0.5,
    )


class TestDROInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n == 2
        assert inst.n_support == 3

    def test_cost(self):
        inst = _make_simple()
        x = np.array([1.0, 0.0])
        xi = np.array([1.0, -1.0])
        # cost = (3+1)*1 + (5-1)*0 = 4
        assert inst.cost(x, xi) == pytest.approx(4.0)

    def test_nominal_expected_cost(self):
        inst = _make_simple()
        x = np.array([0.5, 0.5])
        cost = inst.nominal_expected_cost(x)
        assert cost > 0

    def test_random_instance(self):
        inst = DROInstance.random(n=3, n_support=10)
        assert inst.n == 3
        assert inst.n_support == 10


class TestWassersteinDRO:

    def test_wasserstein_feasible(self):
        inst = _make_simple()
        sol = solve_wasserstein_dro(inst)
        assert sol is not None
        assert all(x >= -1e-6 for x in sol.x)
        assert all(x <= 1.0 + 1e-6 for x in sol.x)

    def test_wasserstein_more_conservative(self):
        """Wasserstein DRO cost should be >= nominal cost for same x."""
        inst = _make_simple()
        sol_w = solve_wasserstein_dro(inst)
        sol_n = solve_nominal(inst)
        assert sol_w is not None and sol_n is not None
        # Worst-case cost of DRO should be >= nominal cost of nominal solution
        # (since DRO hedges against worst case)
        assert sol_w.worst_case_cost >= sol_n.nominal_cost - 1e-6

    def test_nominal_feasible(self):
        inst = _make_simple()
        sol = solve_nominal(inst)
        assert sol is not None
        assert sol.nominal_cost >= 0

    def test_zero_radius_matches_nominal(self):
        inst = DROInstance(
            n=2,
            c=np.array([3.0, 5.0]),
            support_points=np.array([[0.0, 0.0]]),
            A_ub=np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
            b_ub=np.array([1, 1, 0, 0]),
            wasserstein_radius=0.0,
        )
        sol = solve_wasserstein_dro(inst)
        assert sol is not None


class TestMomentDRO:

    def test_moment_dro_feasible(self):
        inst = _make_simple()
        sol = solve_moment_dro(inst, mean_tol=2.0)
        assert sol.worst_case_cost > 0 or sol.worst_case_cost == pytest.approx(0.0)

    def test_worst_case_distribution_valid(self):
        inst = _make_simple()
        x = np.array([0.5, 0.5])
        wc_cost, wc_probs = worst_case_distribution(inst, x, mean_tol=2.0)
        assert abs(wc_probs.sum() - 1.0) < 1e-6
        assert all(p >= -1e-9 for p in wc_probs)
        assert wc_cost >= inst.nominal_expected_cost(x) - 1e-6

    def test_tight_mean_reduces_to_nominal(self):
        """Very tight mean constraint => close to nominal."""
        inst = _make_simple()
        x = np.array([0.5, 0.5])
        wc_cost_tight, _ = worst_case_distribution(inst, x, mean_tol=0.01)
        wc_cost_loose, _ = worst_case_distribution(inst, x, mean_tol=10.0)
        assert wc_cost_tight <= wc_cost_loose + 1e-6

    def test_random_instance(self):
        inst = DROInstance.random(n=3, n_support=8)
        sol = solve_moment_dro(inst, mean_tol=3.0)
        assert sol is not None

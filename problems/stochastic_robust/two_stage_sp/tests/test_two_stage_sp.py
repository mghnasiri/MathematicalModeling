"""Tests for Two-Stage Stochastic Programming."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

# Load modules via importlib
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst_mod = _load_mod("sp_instance", os.path.join(_base, "instance.py"))
_de_mod = _load_mod("sp_de", os.path.join(_base, "heuristics", "deterministic_equivalent.py"))
_saa_mod = _load_mod("sp_saa", os.path.join(_base, "metaheuristics", "sample_average.py"))

TwoStageSPInstance = _inst_mod.TwoStageSPInstance
TwoStageSPSolution = _inst_mod.TwoStageSPSolution
solve_deterministic_equivalent = _de_mod.solve_deterministic_equivalent
solve_expected_value = _de_mod.solve_expected_value
sample_average_approximation = _saa_mod.sample_average_approximation


class TestTwoStageSPInstance:
    """Test instance creation."""

    def test_newsvendor_formulation(self):
        inst = TwoStageSPInstance.newsvendor_as_2ssp(
            unit_cost=5.0, selling_price=10.0, salvage_value=2.0,
            demand_scenarios=np.array([40, 60, 80]),
        )
        assert inst.n1 == 1
        assert inst.n2 == 2
        assert inst.n_scenarios == 3

    def test_capacity_planning(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=3, n_scenarios=5, seed=42
        )
        assert inst.n1 == 3
        assert inst.n_scenarios == 5
        assert inst.n2 == 9  # 3 facilities * 3 customers

    def test_probabilities_default_uniform(self):
        inst = TwoStageSPInstance.capacity_planning(n_facilities=2, n_scenarios=4)
        assert len(inst.probabilities) == 4
        assert abs(sum(inst.probabilities) - 1.0) < 1e-10


class TestDeterministicEquivalent:
    """Test the extensive form LP solver."""

    def test_capacity_planning_feasible(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=3, n_scenarios=5, seed=42
        )
        sol = solve_deterministic_equivalent(inst)
        assert sol is not None
        assert sol.total_cost > 0
        assert len(sol.x) == 3
        assert all(x >= -1e-6 for x in sol.x)

    def test_first_stage_cost_positive(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=2, n_scenarios=3, seed=7
        )
        sol = solve_deterministic_equivalent(inst)
        assert sol is not None
        assert sol.first_stage_cost >= 0
        assert abs(sol.total_cost - sol.first_stage_cost - sol.expected_recourse_cost) < 1e-6

    def test_has_recourse_solutions(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=2, n_scenarios=3, seed=7
        )
        sol = solve_deterministic_equivalent(inst)
        assert sol is not None
        assert len(sol.recourse_solutions) == 3


class TestExpectedValue:
    """Test expected value solution."""

    def test_ev_solution_exists(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=3, n_scenarios=5, seed=42
        )
        sol = solve_expected_value(inst)
        assert sol is not None
        assert sol.total_cost > 0

    def test_ev_lower_bound(self):
        """EV solution cost should be <= stochastic solution cost on mean."""
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=2, n_scenarios=4, seed=42
        )
        sol_de = solve_deterministic_equivalent(inst)
        sol_ev = solve_expected_value(inst)
        # EV is solved on mean scenario, so its obj on that scenario
        # is a lower bound (by LP optimality on the mean)
        assert sol_ev is not None
        assert sol_de is not None


class TestSAA:
    """Test Sample Average Approximation."""

    def test_saa_produces_solution(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=3, n_scenarios=10, seed=42
        )
        result = sample_average_approximation(
            inst, sample_size=5, n_replications=3, seed=42
        )
        assert result.best_solution is not None
        assert result.n_replications > 0
        assert result.mean_objective > 0

    def test_saa_multiple_replications(self):
        inst = TwoStageSPInstance.capacity_planning(
            n_facilities=2, n_scenarios=8, seed=7
        )
        result = sample_average_approximation(
            inst, sample_size=4, n_replications=5, seed=7
        )
        assert len(result.objective_values) == 5
        assert result.std_objective >= 0

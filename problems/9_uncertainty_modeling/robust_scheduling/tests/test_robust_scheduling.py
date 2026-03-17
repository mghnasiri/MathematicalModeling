"""Tests for Robust Single Machine Scheduling."""
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
_inst_mod = _load_mod("rsched_instance", os.path.join(_base, "instance.py"))
_heur_mod = _load_mod("rsched_heur", os.path.join(_base, "heuristics", "minmax_regret_heuristics.py"))
_sa_mod = _load_mod("rsched_sa", os.path.join(_base, "metaheuristics", "simulated_annealing.py"))

RobustSchedulingInstance = _inst_mod.RobustSchedulingInstance
RobustSchedulingSolution = _inst_mod.RobustSchedulingSolution
midpoint_wspt = _heur_mod.midpoint_wspt
scenario_enumeration = _heur_mod.scenario_enumeration
worst_case_wspt = _heur_mod.worst_case_wspt
simulated_annealing = _sa_mod.simulated_annealing


def _make_simple():
    return RobustSchedulingInstance(
        n=4,
        processing_scenarios=np.array([
            [3, 5, 2, 4],    # scenario 0
            [5, 3, 4, 2],    # scenario 1 (reversed difficulty)
            [4, 4, 3, 3],    # scenario 2 (uniform-ish)
        ]),
        weights=np.array([2, 1, 3, 1]),
    )


class TestRobustSchedulingInstance:

    def test_creation(self):
        inst = _make_simple()
        assert inst.n == 4
        assert inst.n_scenarios == 3

    def test_mean_processing(self):
        inst = _make_simple()
        mp = inst.mean_processing
        assert len(mp) == 4
        assert mp[0] == pytest.approx(4.0)  # (3+5+4)/3

    def test_total_weighted_completion(self):
        inst = _make_simple()
        # Order [0,1,2,3], scenario 0: p=[3,5,2,4]
        # C = [3, 8, 10, 14], wC = [6, 8, 30, 14] = 58
        perm = [0, 1, 2, 3]
        twc = inst.total_weighted_completion(perm, 0)
        assert twc == pytest.approx(58.0)

    def test_regret_nonnegative(self):
        inst = _make_simple()
        perm = list(range(inst.n))
        regret = inst.max_regret_twc(perm)
        assert regret >= -1e-9

    def test_random_instance(self):
        inst = RobustSchedulingInstance.random(n=6, n_scenarios=8)
        assert inst.n == 6
        assert inst.n_scenarios == 8


class TestHeuristics:

    def test_midpoint_wspt(self):
        inst = _make_simple()
        sol = midpoint_wspt(inst)
        assert len(sol.permutation) == 4
        assert set(sol.permutation) == {0, 1, 2, 3}
        assert sol.max_regret >= 0

    def test_scenario_enumeration(self):
        inst = _make_simple()
        sol = scenario_enumeration(inst)
        assert len(sol.permutation) == 4
        assert sol.max_regret >= 0

    def test_enumeration_best_among_candidates(self):
        """Scenario enumeration should find best among its candidates."""
        inst = _make_simple()
        sol_enum = scenario_enumeration(inst)
        sol_mid = midpoint_wspt(inst)
        assert sol_enum.max_regret <= sol_mid.max_regret + 1e-9

    def test_worst_case_wspt(self):
        inst = _make_simple()
        sol = worst_case_wspt(inst)
        assert len(sol.permutation) == 4
        assert sol.max_regret >= 0

    def test_random_instance(self):
        inst = RobustSchedulingInstance.random(n=8, n_scenarios=10)
        sol = midpoint_wspt(inst)
        assert sol.expected_twc > 0


class TestSimulatedAnnealing:

    def test_sa_simple(self):
        inst = _make_simple()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        assert len(sol.permutation) == 4
        assert sol.max_regret >= 0

    def test_sa_improves_or_matches(self):
        inst = RobustSchedulingInstance.random(n=6, n_scenarios=8, seed=7)
        sol_h = scenario_enumeration(inst)
        sol_sa = simulated_annealing(inst, max_iterations=3000, seed=7)
        assert sol_sa.max_regret <= sol_h.max_regret + 1.0

    def test_sa_deterministic(self):
        inst = _make_simple()
        sol1 = simulated_annealing(inst, seed=99, max_iterations=500)
        sol2 = simulated_annealing(inst, seed=99, max_iterations=500)
        assert sol1.max_regret == pytest.approx(sol2.max_regret)

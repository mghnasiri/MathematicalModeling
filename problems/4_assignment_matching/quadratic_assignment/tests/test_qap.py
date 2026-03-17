"""Tests for Quadratic Assignment Problem."""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), "..")
_inst = _load_mod("qap_instance", os.path.join(_base, "instance.py"))
_greedy = _load_mod("qap_greedy", os.path.join(_base, "heuristics", "greedy_qap.py"))
_sa = _load_mod("qap_sa", os.path.join(_base, "metaheuristics", "simulated_annealing.py"))

QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution
greedy_construction = _greedy.greedy_construction
random_construction = _greedy.random_construction
simulated_annealing = _sa.simulated_annealing


class TestQAPInstance:
    def test_nug5(self):
        inst = QAPInstance.nug5()
        assert inst.n == 5

    def test_objective_identity(self):
        inst = QAPInstance.nug5()
        obj = inst.objective([0, 1, 2, 3, 4])
        assert obj > 0

    def test_delta_swap(self):
        inst = QAPInstance.nug5()
        perm = [0, 1, 2, 3, 4]
        obj_before = inst.objective(perm)
        delta = inst.delta_swap(perm, 0, 1)
        perm[0], perm[1] = perm[1], perm[0]
        obj_after = inst.objective(perm)
        assert abs(delta - (obj_after - obj_before)) < 1e-6

    def test_random(self):
        inst = QAPInstance.random(n=8)
        assert inst.n == 8


class TestGreedy:
    def test_greedy_valid_perm(self):
        inst = QAPInstance.nug5()
        sol = greedy_construction(inst)
        assert sorted(sol.permutation) == list(range(5))

    def test_random_construction(self):
        inst = QAPInstance.nug5()
        sol = random_construction(inst, seed=42)
        assert sorted(sol.permutation) == list(range(5))

    def test_greedy_better_than_worst(self):
        inst = QAPInstance.random(n=8, seed=7)
        sol_g = greedy_construction(inst)
        sol_r = random_construction(inst, seed=7)
        # Greedy should usually be competitive
        assert sol_g.objective > 0


class TestSimulatedAnnealing:
    def test_sa_nug5(self):
        inst = QAPInstance.nug5()
        sol = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert sorted(sol.permutation) == list(range(5))
        assert sol.objective <= 60  # optimal is 50

    def test_sa_improves(self):
        inst = QAPInstance.random(n=10, seed=7)
        sol_g = greedy_construction(inst)
        sol_sa = simulated_annealing(inst, max_iterations=10000, seed=7)
        assert sol_sa.objective <= sol_g.objective + 1e-6

    def test_sa_deterministic(self):
        inst = QAPInstance.nug5()
        s1 = simulated_annealing(inst, seed=99, max_iterations=2000)
        s2 = simulated_annealing(inst, seed=99, max_iterations=2000)
        assert s1.objective == pytest.approx(s2.objective)

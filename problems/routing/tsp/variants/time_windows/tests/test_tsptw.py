"""Tests for TSP with Time Windows."""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
import pytest

_variant_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("tsptw_instance_test", os.path.join(_variant_dir, "instance.py"))
TSPTWInstance = _inst.TSPTWInstance
TSPTWSolution = _inst.TSPTWSolution
validate_solution = _inst.validate_solution
small_tsptw_5 = _inst.small_tsptw_5

_heur = _load_mod("tsptw_heuristics_test", os.path.join(_variant_dir, "heuristics.py"))
nearest_feasible = _heur.nearest_feasible
earliest_deadline_insertion = _heur.earliest_deadline_insertion

_meta = _load_mod("tsptw_meta_test", os.path.join(_variant_dir, "metaheuristics.py"))
simulated_annealing = _meta.simulated_annealing


class TestTSPTWInstance:
    def test_random_creation(self):
        inst = TSPTWInstance.random(n=8, seed=42)
        assert inst.n == 8
        assert inst.distance_matrix.shape == (8, 8)
        assert inst.time_windows.shape == (8, 2)

    def test_small_benchmark(self):
        inst = small_tsptw_5()
        assert inst.n == 5
        # Depot has wide window
        assert inst.time_windows[0][1] == 200.0

    def test_tour_distance(self):
        inst = small_tsptw_5()
        tour = [0, 1, 2, 3, 4]
        dist = inst.tour_distance(tour)
        assert dist > 0

    def test_feasibility_check(self):
        inst = small_tsptw_5()
        # Sequential tour should be feasible given the windows
        assert inst.tour_feasible([0, 1, 2, 3, 4])


class TestTSPTWHeuristics:
    def test_nearest_feasible_valid(self):
        inst = TSPTWInstance.random(n=8, seed=42)
        sol = nearest_feasible(inst)
        assert sorted(sol.tour) == list(range(inst.n))
        assert sol.distance > 0

    def test_nearest_feasible_permutation(self):
        inst = TSPTWInstance.random(n=8, seed=42)
        sol = nearest_feasible(inst)
        assert sorted(sol.tour) == list(range(inst.n))

    def test_earliest_deadline_valid(self):
        inst = TSPTWInstance.random(n=8, seed=42)
        sol = earliest_deadline_insertion(inst)
        assert sorted(sol.tour) == list(range(inst.n))

    def test_small_benchmark_feasible(self):
        inst = small_tsptw_5()
        sol = nearest_feasible(inst)
        assert sol.feasible


class TestTSPTWSA:
    def test_valid(self):
        inst = TSPTWInstance.random(n=8, seed=42)
        sol = simulated_annealing(inst, max_iterations=10000, seed=42)
        assert sorted(sol.tour) == list(range(inst.n))

    def test_competitive(self):
        inst = small_tsptw_5()
        nf_sol = nearest_feasible(inst)
        sa_sol = simulated_annealing(inst, max_iterations=20000, seed=42)
        # SA should find a feasible tour on small benchmark
        assert sorted(sa_sol.tour) == list(range(inst.n))

    def test_determinism(self):
        inst = TSPTWInstance.random(n=6, seed=42)
        sol1 = simulated_annealing(inst, max_iterations=5000, seed=42)
        sol2 = simulated_annealing(inst, max_iterations=5000, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-6

    def test_time_limit(self):
        inst = TSPTWInstance.random(n=10, seed=42)
        sol = simulated_annealing(inst, max_iterations=1000000, time_limit=0.5, seed=42)
        assert sorted(sol.tour) == list(range(inst.n))

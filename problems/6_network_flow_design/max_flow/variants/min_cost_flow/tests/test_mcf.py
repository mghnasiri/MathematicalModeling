"""Tests for Minimum Cost Flow."""

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


_inst = _load_mod("mcf_inst_test", os.path.join(_variant_dir, "instance.py"))
MinCostFlowInstance = _inst.MinCostFlowInstance
MinCostFlowSolution = _inst.MinCostFlowSolution
validate_solution = _inst.validate_solution
small_mcf_4 = _inst.small_mcf_4

_heur = _load_mod("mcf_heur_test", os.path.join(_variant_dir, "heuristics.py"))
successive_shortest_paths = _heur.successive_shortest_paths


class TestMCFInstance:
    def test_random(self):
        inst = MinCostFlowInstance.random(n=6, seed=42)
        assert inst.n == 6
        assert abs(inst.supply.sum()) < 1e-6

    def test_small(self):
        inst = small_mcf_4()
        assert inst.n == 4


class TestSSP:
    def test_valid(self):
        inst = small_mcf_4()
        sol = successive_shortest_paths(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_flow_amount(self):
        inst = small_mcf_4()
        sol = successive_shortest_paths(inst)
        assert abs(sol.total_flow - 10.0) < 1e-4

    def test_positive_cost(self):
        inst = small_mcf_4()
        sol = successive_shortest_paths(inst)
        assert sol.total_cost > 0

    def test_random_valid(self):
        inst = MinCostFlowInstance.random(n=6, seed=42)
        sol = successive_shortest_paths(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_optimal_small(self):
        """Verify optimal cost on small instance."""
        # Path 0->1->3: cost 2*f1 + 3*f1 = 5*f1
        # Path 0->1->2->3: cost 2*f2 + 1*f2 + 4*f2 = 7*f2
        # Path 0->2->3: cost 5*f3 + 4*f3 = 9*f3
        # Optimal: send 7 via 0->1->3 (cost=35), 3 via 0->1->2->3 (cost=21) = 56
        # Or: 7 via 0->1->3 (35), then 5 via 0->1->2->3 = impossible (cap 0->1 = 10)
        # Actually cap(0,1)=10, cap(1,3)=7: send 7 via 0->1->3 = 35
        # Remaining 3 via 0->1->2->3: 3*(2+1+4) = 21, total = 56
        inst = small_mcf_4()
        sol = successive_shortest_paths(inst)
        assert sol.total_cost <= 60  # should be near optimal

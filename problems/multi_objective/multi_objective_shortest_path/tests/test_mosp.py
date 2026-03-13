"""Tests for Multi-Objective Shortest Path problem.

Tests: path validity, Pareto optimality, dominance, edge cases.
"""
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

_instance_mod = _load_mod("mosp_inst_test", os.path.join(_base, "instance.py"))
_ls_mod = _load_mod("mosp_ls_test", os.path.join(_base, "exact", "label_setting.py"))

MultiObjectiveSPInstance = _instance_mod.MultiObjectiveSPInstance
MOSPSolution = _instance_mod.MOSPSolution
label_setting = _ls_mod.label_setting
dominates = _ls_mod.dominates


class TestDominance:
    """Test the dominance relation."""

    def test_dominates_true(self):
        assert dominates((1.0, 2.0), (2.0, 3.0))
        assert dominates((1.0, 2.0), (1.0, 3.0))

    def test_dominates_false_equal(self):
        assert not dominates((1.0, 2.0), (1.0, 2.0))

    def test_dominates_false_incomparable(self):
        assert not dominates((1.0, 3.0), (2.0, 2.0))


class TestMultiObjectiveSPInstance:
    """Test instance creation."""

    def test_random_instance(self):
        inst = MultiObjectiveSPInstance.random(n=6, n_objectives=2, seed=42)
        assert inst.n == 6
        assert inst.n_objectives == 2
        assert inst.source == 0
        assert inst.target == 5

    def test_adjacency_list(self):
        inst = MultiObjectiveSPInstance(
            n=3, n_objectives=2,
            edges=[(0, 1, (1.0, 2.0)), (1, 2, (3.0, 1.0))],
            source=0, target=2,
        )
        adj = inst.get_adjacency()
        assert len(adj[0]) == 1
        assert adj[0][0] == (1, (1.0, 2.0))


class TestLabelSetting:
    """Test multi-objective label-setting algorithm."""

    def test_simple_two_paths(self):
        """Two paths: one cheap in obj1, one in obj2."""
        inst = MultiObjectiveSPInstance(
            n=3, n_objectives=2,
            edges=[
                (0, 1, (1.0, 10.0)),
                (1, 2, (1.0, 10.0)),
                (0, 2, (10.0, 1.0)),
            ],
            source=0, target=2,
        )
        sol = label_setting(inst)
        assert len(sol.pareto_paths) == 2
        costs = set(sol.pareto_costs)
        assert (2.0, 20.0) in costs
        assert (10.0, 1.0) in costs

    def test_single_path(self):
        inst = MultiObjectiveSPInstance(
            n=2, n_objectives=2,
            edges=[(0, 1, (5.0, 5.0))],
            source=0, target=1,
        )
        sol = label_setting(inst)
        assert len(sol.pareto_paths) == 1
        assert sol.pareto_paths[0] == [0, 1]
        assert sol.pareto_costs[0] == (5.0, 5.0)

    def test_paths_are_valid(self):
        """All paths start at source and end at target."""
        inst = MultiObjectiveSPInstance.random(n=6, n_objectives=2, seed=42)
        sol = label_setting(inst)
        for path in sol.pareto_paths:
            assert path[0] == inst.source
            assert path[-1] == inst.target
            # No repeated nodes
            assert len(path) == len(set(path))

    def test_pareto_optimality(self):
        """No returned solution should dominate another."""
        inst = MultiObjectiveSPInstance.random(n=6, n_objectives=2, seed=42)
        sol = label_setting(inst)
        for i, ci in enumerate(sol.pareto_costs):
            for j, cj in enumerate(sol.pareto_costs):
                if i != j:
                    assert not dominates(ci, cj), \
                        f"Path {i} dominates path {j}: {ci} vs {cj}"

    def test_path_cost_matches(self):
        """Verify cost vectors match actual edge costs along path."""
        inst = MultiObjectiveSPInstance(
            n=4, n_objectives=2,
            edges=[
                (0, 1, (2.0, 3.0)),
                (1, 2, (4.0, 1.0)),
                (2, 3, (1.0, 5.0)),
                (0, 3, (10.0, 2.0)),
            ],
            source=0, target=3,
        )
        sol = label_setting(inst)
        # Build edge lookup
        edge_costs = {(u, v): c for u, v, c in inst.edges}
        for path, cost in zip(sol.pareto_paths, sol.pareto_costs):
            computed = [0.0] * inst.n_objectives
            for i in range(len(path) - 1):
                ec = edge_costs[(path[i], path[i + 1])]
                for k in range(inst.n_objectives):
                    computed[k] += ec[k]
            assert tuple(computed) == cost

    def test_no_path(self):
        """No path from source to target -> empty result."""
        inst = MultiObjectiveSPInstance(
            n=3, n_objectives=2,
            edges=[(0, 1, (1.0, 1.0))],  # no edge to node 2
            source=0, target=2,
        )
        sol = label_setting(inst)
        assert len(sol.pareto_paths) == 0

    def test_solution_repr(self):
        inst = MultiObjectiveSPInstance.random(n=4, n_objectives=2, seed=42)
        sol = label_setting(inst)
        r = repr(sol)
        assert "MOSPSolution" in r

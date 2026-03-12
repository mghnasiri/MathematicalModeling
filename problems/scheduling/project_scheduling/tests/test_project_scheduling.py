"""Tests for the Multi-Project Scheduling problem."""
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
_instance_mod = _load_mod("mps_instance", os.path.join(_base, "instance.py"))
_sgs_mod = _load_mod(
    "mps_sgs", os.path.join(_base, "heuristics", "priority_sgs.py")
)

MultiProjectInstance = _instance_mod.MultiProjectInstance
MultiProjectSolution = _instance_mod.MultiProjectSolution
Project = _instance_mod.Project
priority_sgs = _sgs_mod.priority_sgs


def _make_simple_instance() -> MultiProjectInstance:
    """Two projects, 4 activities each, 1 resource."""
    proj0 = Project(
        project_id=0, n_activities=4,
        durations=np.array([0, 3, 2, 0]),
        predecessors=[[], [0], [0], [1, 2]],
        resource_requirements=np.array([[0], [1], [1], [0]]),
        deadline=10, weight=1.0,
    )
    proj1 = Project(
        project_id=1, n_activities=4,
        durations=np.array([0, 4, 1, 0]),
        predecessors=[[], [0], [0], [1, 2]],
        resource_requirements=np.array([[0], [2], [1], [0]]),
        deadline=10, weight=1.0,
    )
    return MultiProjectInstance(
        projects=[proj0, proj1],
        n_resources=1,
        resource_capacities=np.array([2]),
    )


class TestMultiProjectInstance:
    """Tests for instance construction."""

    def test_random_instance(self):
        inst = MultiProjectInstance.random(n_projects=3, n_activities=6,
                                            n_resources=2, seed=1)
        assert inst.n_projects == 3
        assert inst.n_resources == 2
        assert len(inst.resource_capacities) == 2

    def test_project_structure(self):
        inst = MultiProjectInstance.random(n_projects=2, n_activities=5,
                                            seed=2)
        for proj in inst.projects:
            assert proj.n_activities == 5
            assert len(proj.durations) == 5
            assert proj.durations[0] == 0   # Source
            assert proj.durations[-1] == 0  # Sink

    def test_simple_instance(self):
        inst = _make_simple_instance()
        assert inst.n_projects == 2
        assert inst.n_resources == 1


class TestPrioritySGS:
    """Tests for the priority-based SGS."""

    def test_all_activities_scheduled(self):
        inst = _make_simple_instance()
        sol = priority_sgs(inst, rule="spt")
        for p_idx, proj in enumerate(inst.projects):
            for j in range(proj.n_activities):
                assert sol.start_times[p_idx][j] >= 0

    def test_precedence_respected(self):
        inst = _make_simple_instance()
        sol = priority_sgs(inst, rule="spt")
        for p_idx, proj in enumerate(inst.projects):
            for j in range(proj.n_activities):
                for pred in proj.predecessors[j]:
                    assert (sol.start_times[p_idx][j] >=
                            sol.start_times[p_idx][pred] + proj.durations[pred])

    def test_source_starts_at_zero(self):
        inst = _make_simple_instance()
        sol = priority_sgs(inst, rule="est")
        for p_idx in range(inst.n_projects):
            assert sol.start_times[p_idx][0] == 0

    def test_makespans_positive(self):
        inst = MultiProjectInstance.random(n_projects=3, n_activities=6,
                                            n_resources=2, seed=10)
        sol = priority_sgs(inst, rule="spt")
        for ms in sol.project_makespans:
            assert ms >= 0

    def test_different_rules_produce_solutions(self):
        inst = MultiProjectInstance.random(n_projects=2, n_activities=5,
                                            n_resources=1, seed=11)
        for rule in ["spt", "est", "weight"]:
            sol = priority_sgs(inst, rule=rule)
            assert sol.objective >= 0

    def test_objective_is_weighted_tardiness(self):
        inst = _make_simple_instance()
        sol = priority_sgs(inst, rule="spt")
        expected = 0.0
        for p_idx, proj in enumerate(inst.projects):
            tardiness = max(0, sol.project_makespans[p_idx] - proj.deadline)
            expected += proj.weight * tardiness
        assert abs(sol.objective - expected) < 1e-6

    def test_single_project(self):
        proj = Project(
            project_id=0, n_activities=3,
            durations=np.array([0, 5, 0]),
            predecessors=[[], [0], [1]],
            resource_requirements=np.array([[0], [1], [0]]),
            deadline=10, weight=1.0,
        )
        inst = MultiProjectInstance(
            projects=[proj], n_resources=1,
            resource_capacities=np.array([2]),
        )
        sol = priority_sgs(inst, rule="spt")
        assert sol.project_makespans[0] == 5

    def test_invalid_rule_raises(self):
        inst = _make_simple_instance()
        with pytest.raises(ValueError):
            priority_sgs(inst, rule="invalid")

    def test_random_instance_completes(self):
        inst = MultiProjectInstance.random(n_projects=4, n_activities=8,
                                            n_resources=2, seed=12)
        sol = priority_sgs(inst, rule="est")
        assert len(sol.project_makespans) == 4

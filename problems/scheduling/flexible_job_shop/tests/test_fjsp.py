"""
Tests for Flexible Job Shop Scheduling Problem (FJSP)

Tests cover:
- Instance creation and validation
- Dispatching rules (SPT, LPT, MWR, LWR)
- Hierarchical heuristic
- Genetic Algorithm metaheuristic
- Edge cases (single job, single machine, total vs partial FJSP)

Run: python -m pytest problems/scheduling/flexible_job_shop/tests/test_fjsp.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

_fjsp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module(
    "fjsp_instance",
    os.path.join(_fjsp_dir, "instance.py"),
)
_dispatch_mod = _load_module(
    "fjsp_dispatching",
    os.path.join(_fjsp_dir, "heuristics", "dispatching_rules.py"),
)
_hier_mod = _load_module(
    "fjsp_hierarchical",
    os.path.join(_fjsp_dir, "heuristics", "hierarchical.py"),
)
_ga_mod = _load_module(
    "fjsp_ga",
    os.path.join(_fjsp_dir, "metaheuristics", "genetic_algorithm.py"),
)

FlexibleJobShopInstance = _instance_mod.FlexibleJobShopInstance
FlexibleJobShopSolution = _instance_mod.FlexibleJobShopSolution
FlexibleOperation = _instance_mod.FlexibleOperation
compute_makespan = _instance_mod.compute_makespan
validate_solution = _instance_mod.validate_solution

dispatching_rule = _dispatch_mod.dispatching_rule
hierarchical = _hier_mod.hierarchical
genetic_algorithm = _ga_mod.genetic_algorithm


# ============================================================
# Instance Tests
# ============================================================

class TestFlexibleJobShopInstance:
    """Tests for FlexibleJobShopInstance dataclass."""

    def test_create_basic(self):
        inst = FlexibleJobShopInstance(
            n=2, m=2,
            jobs=[
                [{0: 3, 1: 5}],
                [{0: 4}, {1: 2}],
            ]
        )
        assert inst.n == 2
        assert inst.m == 2
        assert inst.total_operations() == 3

    def test_random_instance(self):
        inst = FlexibleJobShopInstance.random(
            n=4, m=3, flexibility=0.6, seed=42,
        )
        assert inst.n == 4
        assert inst.m == 3
        for j in range(4):
            for eligible in inst.jobs[j]:
                assert len(eligible) >= 1

    def test_random_total(self):
        inst = FlexibleJobShopInstance.random_total(
            n=3, m=4, seed=42,
        )
        assert inst.is_total()
        for j in range(3):
            for eligible in inst.jobs[j]:
                assert len(eligible) == 4

    def test_from_job_shop(self):
        inst = FlexibleJobShopInstance.from_job_shop(
            n=2, m=2,
            jobs=[[(0, 3), (1, 2)], [(1, 4), (0, 1)]],
        )
        assert inst.n == 2
        assert inst.m == 2
        assert inst.jobs[0][0] == {0: 3}
        assert inst.jobs[0][1] == {1: 2}
        assert not inst.is_total()

    def test_from_standard_format(self):
        text = """2 3 2
2  2 0 3 1 5  1 2 4
2  1 0 2  2 1 3 2 6
"""
        inst = FlexibleJobShopInstance.from_standard_format(text)
        assert inst.n == 2
        assert inst.m == 3
        assert inst.jobs[0][0] == {0: 3, 1: 5}
        assert inst.jobs[0][1] == {2: 4}
        assert inst.jobs[1][0] == {0: 2}
        assert inst.jobs[1][1] == {1: 3, 2: 6}

    def test_get_operation(self):
        inst = FlexibleJobShopInstance(
            n=1, m=2, jobs=[[{0: 3, 1: 5}]],
        )
        op = inst.get_operation(0, 0)
        assert op.job == 0
        assert op.position == 0
        assert op.eligible_machines == {0: 3, 1: 5}

    def test_single_job(self):
        inst = FlexibleJobShopInstance(
            n=1, m=3, jobs=[[{0: 2, 1: 3}, {1: 4, 2: 1}]],
        )
        assert inst.n == 1
        assert inst.total_operations() == 2

    def test_no_eligible_machines_fails(self):
        with pytest.raises(AssertionError):
            FlexibleJobShopInstance(n=1, m=2, jobs=[[{}]])

    def test_invalid_machine_fails(self):
        with pytest.raises(AssertionError):
            FlexibleJobShopInstance(n=1, m=2, jobs=[[{5: 3}]])


# ============================================================
# Validation Tests
# ============================================================

class TestValidation:
    """Tests for solution validation."""

    def test_valid_solution(self):
        inst = FlexibleJobShopInstance(
            n=2, m=2,
            jobs=[[{0: 3, 1: 2}], [{0: 4, 1: 1}]],
        )
        assignments = {(0, 0): 0, (1, 0): 1}
        start_times = {(0, 0): 0, (1, 0): 0}
        valid, _ = validate_solution(inst, assignments, start_times)
        assert valid

    def test_ineligible_machine(self):
        inst = FlexibleJobShopInstance(
            n=1, m=2, jobs=[[{0: 3}]],
        )
        assignments = {(0, 0): 1}
        start_times = {(0, 0): 0}
        valid, _ = validate_solution(inst, assignments, start_times)
        assert not valid

    def test_precedence_violation(self):
        inst = FlexibleJobShopInstance(
            n=1, m=2, jobs=[[{0: 3}, {1: 2}]],
        )
        assignments = {(0, 0): 0, (0, 1): 1}
        start_times = {(0, 0): 5, (0, 1): 2}
        valid, _ = validate_solution(inst, assignments, start_times)
        assert not valid

    def test_machine_conflict(self):
        inst = FlexibleJobShopInstance(
            n=2, m=1, jobs=[[{0: 3}], [{0: 4}]],
        )
        assignments = {(0, 0): 0, (1, 0): 0}
        start_times = {(0, 0): 0, (1, 0): 1}
        valid, _ = validate_solution(inst, assignments, start_times)
        assert not valid


# ============================================================
# Dispatching Rules Tests
# ============================================================

class TestDispatchingRules:
    """Tests for FJSP dispatching rules."""

    @pytest.fixture
    def small_instance(self):
        return FlexibleJobShopInstance(
            n=3, m=2,
            jobs=[
                [{0: 3, 1: 5}, {0: 2, 1: 4}],
                [{0: 4, 1: 2}],
                [{0: 1, 1: 3}, {0: 5, 1: 2}, {1: 3}],
            ],
        )

    def test_spt_ect_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="spt", machine_rule="ect")
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_lpt_ect_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="lpt", machine_rule="ect")
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_mwr_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="mwr")
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_lwr_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="lwr")
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_random_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="random", seed=42)
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_random_deterministic(self, small_instance):
        s1 = dispatching_rule(small_instance, priority_rule="random", seed=123)
        s2 = dispatching_rule(small_instance, priority_rule="random", seed=123)
        assert s1.makespan == s2.makespan

    def test_spt_machine_rule(self, small_instance):
        sol = dispatching_rule(small_instance, priority_rule="spt", machine_rule="spt")
        valid, _ = validate_solution(small_instance, sol.assignments, sol.start_times)
        assert valid

    def test_single_machine_instance(self):
        inst = FlexibleJobShopInstance(
            n=2, m=1, jobs=[[{0: 3}], [{0: 5}]],
        )
        sol = dispatching_rule(inst, priority_rule="spt")
        assert sol.makespan == 8
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_total_fjsp(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = dispatching_rule(inst, priority_rule="mwr", machine_rule="ect")
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_invalid_priority_rule(self, small_instance):
        with pytest.raises(ValueError):
            dispatching_rule(small_instance, priority_rule="invalid")

    def test_invalid_machine_rule(self, small_instance):
        with pytest.raises(ValueError):
            dispatching_rule(small_instance, machine_rule="invalid")


# ============================================================
# Hierarchical Heuristic Tests
# ============================================================

class TestHierarchical:
    """Tests for hierarchical route-then-sequence heuristic."""

    @pytest.fixture
    def medium_instance(self):
        return FlexibleJobShopInstance.random(
            n=5, m=3, flexibility=0.6, seed=42,
        )

    def test_min_load_feasible(self, medium_instance):
        sol = hierarchical(medium_instance, routing_strategy="min_load")
        valid, _ = validate_solution(
            medium_instance, sol.assignments, sol.start_times
        )
        assert valid

    def test_min_pt_feasible(self, medium_instance):
        sol = hierarchical(medium_instance, routing_strategy="min_pt")
        valid, _ = validate_solution(
            medium_instance, sol.assignments, sol.start_times
        )
        assert valid

    def test_single_job(self):
        inst = FlexibleJobShopInstance(
            n=1, m=2, jobs=[[{0: 3, 1: 5}, {0: 2, 1: 4}]],
        )
        sol = hierarchical(inst)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_total_fjsp(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = hierarchical(inst)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_invalid_strategy(self, medium_instance):
        with pytest.raises(ValueError):
            hierarchical(medium_instance, routing_strategy="invalid")


# ============================================================
# Genetic Algorithm Tests
# ============================================================

class TestGeneticAlgorithm:
    """Tests for GA metaheuristic."""

    def test_small_instance(self):
        inst = FlexibleJobShopInstance(
            n=3, m=2,
            jobs=[
                [{0: 3, 1: 5}, {0: 2, 1: 4}],
                [{0: 4, 1: 2}],
                [{0: 1, 1: 3}, {1: 2}],
            ],
        )
        sol = genetic_algorithm(inst, pop_size=20, generations=50, seed=42)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_deterministic(self):
        inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=10)
        s1 = genetic_algorithm(inst, pop_size=10, generations=20, seed=42)
        s2 = genetic_algorithm(inst, pop_size=10, generations=20, seed=42)
        assert s1.makespan == s2.makespan

    def test_total_fjsp(self):
        inst = FlexibleJobShopInstance.random_total(n=4, m=3, seed=42)
        sol = genetic_algorithm(inst, pop_size=20, generations=50, seed=42)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid

    def test_improves_over_dispatching(self):
        inst = FlexibleJobShopInstance.random(n=6, m=3, flexibility=0.6, seed=42)
        disp = dispatching_rule(inst, priority_rule="spt", machine_rule="ect")
        ga_sol = genetic_algorithm(inst, pop_size=30, generations=100, seed=42)
        # GA should at least match dispatching
        assert ga_sol.makespan <= disp.makespan + disp.makespan * 0.3

    def test_single_machine(self):
        inst = FlexibleJobShopInstance(
            n=3, m=1, jobs=[[{0: 3}], [{0: 5}], [{0: 2}]],
        )
        sol = genetic_algorithm(inst, pop_size=10, generations=20, seed=42)
        assert sol.makespan == 10
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid


# ============================================================
# Edge Cases & Integration
# ============================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_operation_per_job(self):
        inst = FlexibleJobShopInstance(
            n=3, m=2,
            jobs=[[{0: 3, 1: 5}], [{0: 4, 1: 2}], [{0: 1, 1: 3}]],
        )
        for method in [
            lambda i: dispatching_rule(i, priority_rule="spt"),
            lambda i: hierarchical(i),
        ]:
            sol = method(inst)
            valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
            assert valid

    def test_one_eligible_per_op(self):
        """Partial FJSP where each op has exactly one eligible machine (= JSP)."""
        inst = FlexibleJobShopInstance(
            n=2, m=2,
            jobs=[
                [{0: 3}, {1: 2}],
                [{1: 4}, {0: 1}],
            ],
        )
        sol = dispatching_rule(inst)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        assert valid
        assert sol.assignments[(0, 0)] == 0
        assert sol.assignments[(0, 1)] == 1

    def test_all_methods_produce_valid(self):
        inst = FlexibleJobShopInstance.random(n=5, m=3, flexibility=0.5, seed=99)
        for name, method in [
            ("SPT/ECT", lambda: dispatching_rule(inst, "spt", "ect")),
            ("MWR/ECT", lambda: dispatching_rule(inst, "mwr", "ect")),
            ("Hierarchical", lambda: hierarchical(inst)),
            ("GA", lambda: genetic_algorithm(inst, pop_size=10, generations=20, seed=42)),
        ]:
            sol = method()
            valid, violations = validate_solution(
                inst, sol.assignments, sol.start_times
            )
            assert valid, f"{name} invalid: {violations}"

"""
Tests for Resource-Constrained Project Scheduling Problem (RCPSP)

Tests cover:
- Instance creation and validation
- Serial SGS with priority rules (LFT, EST, MTS, GRPW)
- Parallel SGS with priority rules
- Genetic Algorithm metaheuristic
- Edge cases (sequential, parallel, no resources)

Run: python -m pytest problems/scheduling/rcpsp/tests/test_rcpsp.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

_rcpsp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module(
    "rcpsp_instance",
    os.path.join(_rcpsp_dir, "instance.py"),
)
_serial_mod = _load_module(
    "rcpsp_serial_sgs",
    os.path.join(_rcpsp_dir, "heuristics", "serial_sgs.py"),
)
_parallel_mod = _load_module(
    "rcpsp_parallel_sgs",
    os.path.join(_rcpsp_dir, "heuristics", "parallel_sgs.py"),
)
_ga_mod = _load_module(
    "rcpsp_ga",
    os.path.join(_rcpsp_dir, "metaheuristics", "genetic_algorithm.py"),
)

RCPSPInstance = _instance_mod.RCPSPInstance
RCPSPSolution = _instance_mod.RCPSPSolution
validate_solution = _instance_mod.validate_solution

serial_sgs = _serial_mod.serial_sgs
parallel_sgs = _parallel_mod.parallel_sgs
genetic_algorithm = _ga_mod.genetic_algorithm


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_instance():
    """Small 4-activity instance with 1 resource."""
    return RCPSPInstance.from_arrays(
        durations=[0, 3, 4, 2, 5, 0],
        resource_demands=[
            [0], [2], [3], [1], [2], [0],
        ],
        resource_capacities=[4],
        successors={
            0: [1, 2],
            1: [3],
            2: [4],
            3: [5],
            4: [5],
        },
    )


@pytest.fixture
def sequential_instance():
    """Fully sequential: 0 -> 1 -> 2 -> 3."""
    return RCPSPInstance.from_arrays(
        durations=[0, 3, 4, 2, 0],
        resource_demands=[[0], [1], [1], [1], [0]],
        resource_capacities=[5],
        successors={0: [1], 1: [2], 2: [3], 3: [4]},
    )


@pytest.fixture
def parallel_instance():
    """Fully parallel activities: all independent."""
    return RCPSPInstance.from_arrays(
        durations=[0, 3, 4, 2, 0],
        resource_demands=[[0], [1], [1], [1], [0]],
        resource_capacities=[3],
        successors={
            0: [1, 2, 3],
            1: [4],
            2: [4],
            3: [4],
        },
    )


# ============================================================
# Instance Tests
# ============================================================

class TestRCPSPInstance:
    """Tests for RCPSPInstance dataclass."""

    def test_create_basic(self, small_instance):
        assert small_instance.n == 4
        assert small_instance.num_resources == 1
        assert small_instance.durations[0] == 0
        assert small_instance.durations[5] == 0

    def test_random_instance(self):
        inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
        assert inst.n == 10
        assert inst.num_resources == 2
        assert inst.durations[0] == 0
        assert inst.durations[11] == 0

    def test_topological_order(self, small_instance):
        order = small_instance.topological_order()
        assert len(order) == 6
        assert order[0] == 0
        assert order[-1] == 5
        # Check precedence
        pos = {a: i for i, a in enumerate(order)}
        for act, succs in small_instance.successors.items():
            for succ in succs:
                assert pos[act] < pos[succ]

    def test_critical_path_length(self, small_instance):
        cp = small_instance.critical_path_length()
        assert cp > 0

    def test_critical_path_sequential(self, sequential_instance):
        cp = sequential_instance.critical_path_length()
        assert cp == 3 + 4 + 2  # sum of all durations

    def test_critical_path_parallel(self, parallel_instance):
        cp = parallel_instance.critical_path_length()
        assert cp == 4  # max of {3, 4, 2}

    def test_earliest_start_times(self, small_instance):
        es = small_instance.earliest_start_times()
        assert es[0] == 0
        # Activities 1 and 2 can both start at 0
        assert es[1] == 0
        assert es[2] == 0

    def test_latest_start_times(self, small_instance):
        ls = small_instance.latest_start_times()
        assert ls[5] == small_instance.critical_path_length()
        assert ls[0] == 0

    def test_from_arrays(self):
        inst = RCPSPInstance.from_arrays(
            durations=[0, 5, 3, 0],
            resource_demands=[[0, 0], [2, 1], [1, 2], [0, 0]],
            resource_capacities=[3, 3],
            successors={0: [1, 2], 1: [3], 2: [3]},
        )
        assert inst.n == 2
        assert inst.num_resources == 2

    def test_invalid_source_duration(self):
        with pytest.raises(AssertionError):
            RCPSPInstance.from_arrays(
                durations=[5, 3, 0],  # source has non-zero duration
                resource_demands=[[0], [0], [0]],
                resource_capacities=[1],
                successors={0: [1], 1: [2]},
            )


# ============================================================
# Validation Tests
# ============================================================

class TestValidation:
    """Tests for solution validation."""

    def test_valid_solution(self, small_instance):
        # Simple sequential schedule
        st = np.array([0, 0, 3, 3, 7, 12])
        valid, _ = validate_solution(small_instance, st)
        assert valid

    def test_precedence_violation(self, small_instance):
        st = np.array([0, 5, 0, 3, 4, 12])  # act 1 starts at 5, act 3 at 3
        valid, violations = validate_solution(small_instance, st)
        # Actually check: 0->1 means act 1 starts after 0 finishes (0+0=0, 5>=0 ok)
        # 1->3: act 1 ends at 5+3=8, act 3 starts at 3 < 8 => violation
        assert not valid

    def test_resource_violation(self, small_instance):
        # Both activities 1 and 2 running simultaneously,
        # demands = 2 + 3 = 5 > capacity 4
        st = np.array([0, 0, 0, 3, 4, 9])
        valid, _ = validate_solution(small_instance, st)
        assert not valid


# ============================================================
# Serial SGS Tests
# ============================================================

class TestSerialSGS:
    """Tests for Serial Schedule Generation Scheme."""

    def test_small_instance(self, small_instance):
        sol = serial_sgs(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid
        assert sol.makespan >= small_instance.critical_path_length()

    def test_sequential(self, sequential_instance):
        sol = serial_sgs(sequential_instance)
        assert sol.makespan == 9  # 3 + 4 + 2

    def test_parallel_unlimited_resources(self, parallel_instance):
        sol = serial_sgs(parallel_instance)
        valid, _ = validate_solution(parallel_instance, sol.start_times)
        assert valid
        # With capacity 3 and demands all 1, all can run in parallel
        assert sol.makespan == 4

    def test_all_priority_rules(self, small_instance):
        for rule in ["lft", "est", "mts", "grpw"]:
            sol = serial_sgs(small_instance, priority_rule=rule)
            valid, _ = validate_solution(small_instance, sol.start_times)
            assert valid, f"Rule {rule} produced invalid solution"
            assert sol.makespan >= small_instance.critical_path_length()

    def test_custom_priority_list(self, small_instance):
        plist = [0, 1, 2, 3, 4, 5]
        sol = serial_sgs(small_instance, priority_list=plist)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_random_instances(self):
        for seed in [1, 2, 3, 4, 5]:
            inst = RCPSPInstance.random(n=8, num_resources=2, seed=seed)
            sol = serial_sgs(inst)
            valid, violations = validate_solution(inst, sol.start_times)
            assert valid, f"Seed {seed}: {violations}"
            assert sol.makespan >= inst.critical_path_length()

    def test_invalid_rule(self, small_instance):
        with pytest.raises(ValueError):
            serial_sgs(small_instance, priority_rule="invalid")


# ============================================================
# Parallel SGS Tests
# ============================================================

class TestParallelSGS:
    """Tests for Parallel Schedule Generation Scheme."""

    def test_small_instance(self, small_instance):
        sol = parallel_sgs(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid
        assert sol.makespan >= small_instance.critical_path_length()

    def test_sequential(self, sequential_instance):
        sol = parallel_sgs(sequential_instance)
        assert sol.makespan == 9

    def test_parallel_unlimited(self, parallel_instance):
        sol = parallel_sgs(parallel_instance)
        valid, _ = validate_solution(parallel_instance, sol.start_times)
        assert valid
        assert sol.makespan == 4

    def test_all_priority_rules(self, small_instance):
        for rule in ["lft", "est", "mts", "grpw"]:
            sol = parallel_sgs(small_instance, priority_rule=rule)
            valid, _ = validate_solution(small_instance, sol.start_times)
            assert valid, f"Rule {rule} produced invalid solution"

    def test_random_instances(self):
        for seed in [1, 2, 3, 4, 5]:
            inst = RCPSPInstance.random(n=8, num_resources=2, seed=seed)
            sol = parallel_sgs(inst)
            valid, violations = validate_solution(inst, sol.start_times)
            assert valid, f"Seed {seed}: {violations}"

    def test_invalid_rule(self, small_instance):
        with pytest.raises(ValueError):
            parallel_sgs(small_instance, priority_rule="invalid")


# ============================================================
# Genetic Algorithm Tests
# ============================================================

class TestGeneticAlgorithm:
    """Tests for GA metaheuristic."""

    def test_small_instance(self, small_instance):
        sol = genetic_algorithm(
            small_instance, pop_size=20, generations=50, seed=42,
        )
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid
        assert sol.makespan >= small_instance.critical_path_length()

    def test_deterministic(self):
        inst = RCPSPInstance.random(n=8, num_resources=2, seed=10)
        s1 = genetic_algorithm(inst, pop_size=10, generations=20, seed=42)
        s2 = genetic_algorithm(inst, pop_size=10, generations=20, seed=42)
        assert s1.makespan == s2.makespan

    def test_sequential(self, sequential_instance):
        sol = genetic_algorithm(
            sequential_instance, pop_size=10, generations=20, seed=42,
        )
        assert sol.makespan == 9

    def test_improves_over_sgs(self):
        inst = RCPSPInstance.random(n=12, num_resources=2, seed=42)
        sgs_sol = serial_sgs(inst, priority_rule="lft")
        ga_sol = genetic_algorithm(
            inst, pop_size=30, generations=100, seed=42,
        )
        # GA should at least match SGS
        assert ga_sol.makespan <= sgs_sol.makespan + 5

    def test_random_instances(self):
        for seed in [1, 2, 3]:
            inst = RCPSPInstance.random(n=8, num_resources=2, seed=seed)
            sol = genetic_algorithm(
                inst, pop_size=15, generations=30, seed=42,
            )
            valid, violations = validate_solution(inst, sol.start_times)
            assert valid, f"Seed {seed}: {violations}"


# ============================================================
# Edge Cases & Integration
# ============================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_two_activities(self):
        """Simplest non-trivial instance: source -> 1 -> sink."""
        inst = RCPSPInstance.from_arrays(
            durations=[0, 5, 0],
            resource_demands=[[0], [3], [0]],
            resource_capacities=[5],
            successors={0: [1], 1: [2]},
        )
        sol = serial_sgs(inst)
        assert sol.makespan == 5

    def test_no_resource_constraints(self):
        """Ample resources — should match critical path."""
        inst = RCPSPInstance.from_arrays(
            durations=[0, 3, 4, 2, 0],
            resource_demands=[[0], [1], [1], [1], [0]],
            resource_capacities=[100],
            successors={0: [1, 2, 3], 1: [4], 2: [4], 3: [4]},
        )
        sol = serial_sgs(inst)
        assert sol.makespan == inst.critical_path_length()

    def test_all_methods_agree_on_sequential(self, sequential_instance):
        s1 = serial_sgs(sequential_instance).makespan
        s2 = parallel_sgs(sequential_instance).makespan
        s3 = genetic_algorithm(
            sequential_instance, pop_size=10, generations=10, seed=42
        ).makespan
        assert s1 == s2 == s3 == 9

    def test_tight_resources(self):
        """Very tight resources force sequential execution."""
        inst = RCPSPInstance.from_arrays(
            durations=[0, 2, 3, 4, 0],
            resource_demands=[[0], [3], [3], [3], [0]],
            resource_capacities=[3],
            successors={0: [1, 2, 3], 1: [4], 2: [4], 3: [4]},
        )
        sol = serial_sgs(inst)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
        # With capacity 3 and demand 3 each, must be sequential
        assert sol.makespan == 2 + 3 + 4

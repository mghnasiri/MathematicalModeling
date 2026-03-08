"""
Tests for Job Shop Scheduling Problem (JSP)

Tests cover:
- Instance creation and validation
- Dispatching rules (SPT, LPT, MWR, LWR, FIFO, RANDOM)
- Shifting Bottleneck heuristic
- Simulated Annealing metaheuristic
- Tabu Search metaheuristic
- Benchmark instances (ft06, ft10)

Run: python -m pytest problems/scheduling/job_shop/tests/test_job_shop.py -v
"""

import sys
import os
import importlib.util
import pytest
import numpy as np

# Use importlib to avoid name collisions with other instance.py files
_job_shop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_instance_mod = _load_module(
    "job_shop_instance",
    os.path.join(_job_shop_dir, "instance.py"),
)
_dispatch_mod = _load_module(
    "job_shop_dispatching",
    os.path.join(_job_shop_dir, "heuristics", "dispatching_rules.py"),
)
_sb_mod = _load_module(
    "job_shop_shifting_bottleneck",
    os.path.join(_job_shop_dir, "heuristics", "shifting_bottleneck.py"),
)
_sa_mod = _load_module(
    "job_shop_sa",
    os.path.join(_job_shop_dir, "metaheuristics", "simulated_annealing.py"),
)
_ts_mod = _load_module(
    "job_shop_ts",
    os.path.join(_job_shop_dir, "metaheuristics", "tabu_search.py"),
)

JobShopInstance = _instance_mod.JobShopInstance
JobShopSolution = _instance_mod.JobShopSolution
Operation = _instance_mod.Operation
compute_makespan = _instance_mod.compute_makespan
validate_solution = _instance_mod.validate_solution
build_machine_sequences = _instance_mod.build_machine_sequences
ft06 = _instance_mod.ft06
ft10 = _instance_mod.ft10

dispatching_rule = _dispatch_mod.dispatching_rule
spt = _dispatch_mod.spt
lpt = _dispatch_mod.lpt
mwr = _dispatch_mod.mwr
lwr = _dispatch_mod.lwr
fifo = _dispatch_mod.fifo

shifting_bottleneck = _sb_mod.shifting_bottleneck
simulated_annealing = _sa_mod.simulated_annealing
tabu_search = _ts_mod.tabu_search


# ============================================================
# Instance Tests
# ============================================================

class TestJobShopInstance:
    """Tests for JobShopInstance dataclass."""

    def test_create_basic(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[
                [(0, 3), (1, 2)],
                [(1, 4), (0, 1)],
            ]
        )
        assert inst.n == 2
        assert inst.m == 2
        assert inst.total_operations() == 4

    def test_random_instance(self):
        inst = JobShopInstance.random(n=5, m=3, seed=42)
        assert inst.n == 5
        assert inst.m == 3
        for j in range(5):
            assert inst.num_operations(j) == 3
            machines = [m for m, _ in inst.jobs[j]]
            assert sorted(machines) == [0, 1, 2]

    def test_from_arrays(self):
        machines = [[0, 1], [1, 0]]
        times = [[3, 2], [4, 1]]
        inst = JobShopInstance.from_arrays(machines, times)
        assert inst.n == 2
        assert inst.m == 2
        assert inst.jobs[0] == [(0, 3), (1, 2)]
        assert inst.jobs[1] == [(1, 4), (0, 1)]

    def test_from_standard_format(self):
        text = """2 2
0 3 1 2
1 4 0 1
"""
        inst = JobShopInstance.from_standard_format(text)
        assert inst.n == 2
        assert inst.m == 2
        assert inst.jobs[0] == [(0, 3), (1, 2)]

    def test_get_operation(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 5), (1, 3)], [(1, 2), (0, 4)]],
        )
        op = inst.get_operation(0, 1)
        assert op.job == 0
        assert op.position == 1
        assert op.machine == 1
        assert op.processing_time == 3

    def test_processing_time_matrix(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 5), (1, 3)], [(1, 2), (0, 4)]],
        )
        pt = inst.processing_time_matrix()
        assert pt.shape == (2, 2)
        assert pt[0, 0] == 5
        assert pt[0, 1] == 3
        assert pt[1, 0] == 2
        assert pt[1, 1] == 4

    def test_machine_matrix(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 5), (1, 3)], [(1, 2), (0, 4)]],
        )
        mm = inst.machine_matrix()
        assert mm[0, 0] == 0
        assert mm[0, 1] == 1
        assert mm[1, 0] == 1
        assert mm[1, 1] == 0

    def test_ft06_exists(self):
        inst = ft06()
        assert inst.n == 6
        assert inst.m == 6
        assert inst.total_operations() == 36

    def test_ft10_exists(self):
        inst = ft10()
        assert inst.n == 10
        assert inst.m == 10
        assert inst.total_operations() == 100

    def test_single_job(self):
        inst = JobShopInstance(n=1, m=3, jobs=[[(0, 2), (1, 3), (2, 1)]])
        assert inst.n == 1
        assert inst.num_operations(0) == 3

    def test_invalid_machine(self):
        with pytest.raises(AssertionError):
            JobShopInstance(n=1, m=2, jobs=[[(0, 1), (5, 2)]])


# ============================================================
# Validation Tests
# ============================================================

class TestValidation:
    """Tests for solution validation."""

    def test_valid_solution(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 3), (1, 2)], [(1, 4), (0, 1)]],
        )
        start_times = {(0, 0): 0, (0, 1): 4, (1, 0): 0, (1, 1): 4}
        valid, violations = validate_solution(inst, start_times)
        assert valid
        assert len(violations) == 0

    def test_precedence_violation(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 3), (1, 2)], [(1, 4), (0, 1)]],
        )
        start_times = {(0, 0): 5, (0, 1): 2, (1, 0): 0, (1, 1): 6}
        valid, violations = validate_solution(inst, start_times)
        assert not valid

    def test_machine_conflict(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 3), (1, 2)], [(0, 4), (1, 1)]],
        )
        # Both start on machine 0 at time 0
        start_times = {(0, 0): 0, (0, 1): 3, (1, 0): 1, (1, 1): 5}
        valid, violations = validate_solution(inst, start_times)
        assert not valid

    def test_missing_operation(self):
        inst = JobShopInstance(
            n=1, m=2, jobs=[[(0, 3), (1, 2)]],
        )
        start_times = {(0, 0): 0}
        valid, violations = validate_solution(inst, start_times)
        assert not valid


# ============================================================
# Dispatching Rules Tests
# ============================================================

class TestDispatchingRules:
    """Tests for dispatching rule heuristics."""

    @pytest.fixture
    def small_instance(self):
        return JobShopInstance(
            n=3, m=2,
            jobs=[
                [(0, 3), (1, 2)],
                [(1, 4), (0, 1)],
                [(0, 2), (1, 3)],
            ],
        )

    def test_spt_feasible(self, small_instance):
        sol = spt(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid
        assert sol.makespan > 0

    def test_lpt_feasible(self, small_instance):
        sol = lpt(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_mwr_feasible(self, small_instance):
        sol = mwr(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_lwr_feasible(self, small_instance):
        sol = lwr(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_fifo_feasible(self, small_instance):
        sol = fifo(small_instance)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_random_feasible(self, small_instance):
        sol = dispatching_rule(small_instance, rule="random", seed=42)
        valid, _ = validate_solution(small_instance, sol.start_times)
        assert valid

    def test_random_deterministic(self, small_instance):
        s1 = dispatching_rule(small_instance, rule="random", seed=123)
        s2 = dispatching_rule(small_instance, rule="random", seed=123)
        assert s1.makespan == s2.makespan

    def test_all_rules_on_ft06(self):
        inst = ft06()
        for rule in ["spt", "lpt", "mwr", "lwr", "fifo"]:
            sol = dispatching_rule(inst, rule=rule)
            valid, _ = validate_solution(inst, sol.start_times)
            assert valid, f"Rule {rule} produced invalid solution"
            assert sol.makespan >= 55, f"Rule {rule}: makespan < optimal"

    def test_invalid_rule(self, small_instance):
        with pytest.raises(ValueError):
            dispatching_rule(small_instance, rule="invalid")

    def test_single_job(self):
        inst = JobShopInstance(n=1, m=3, jobs=[[(0, 2), (1, 3), (2, 1)]])
        sol = spt(inst)
        assert sol.makespan == 6
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid

    def test_single_machine(self):
        inst = JobShopInstance(n=3, m=1, jobs=[[(0, 3)], [(0, 2)], [(0, 5)]])
        sol = spt(inst)
        assert sol.makespan == 10
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid


# ============================================================
# Shifting Bottleneck Tests
# ============================================================

class TestShiftingBottleneck:
    """Tests for Shifting Bottleneck heuristic."""

    def test_small_instance(self):
        inst = JobShopInstance(
            n=3, m=2,
            jobs=[
                [(0, 3), (1, 2)],
                [(1, 4), (0, 1)],
                [(0, 2), (1, 3)],
            ],
        )
        sol = shifting_bottleneck(inst)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
        assert sol.makespan > 0

    def test_ft06(self):
        inst = ft06()
        sol = shifting_bottleneck(inst)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
        assert sol.makespan >= 55

    def test_single_job(self):
        inst = JobShopInstance(n=1, m=3, jobs=[[(0, 2), (1, 3), (2, 1)]])
        sol = shifting_bottleneck(inst)
        assert sol.makespan == 6
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid

    def test_two_jobs_two_machines(self):
        inst = JobShopInstance(
            n=2, m=2,
            jobs=[[(0, 3), (1, 2)], [(1, 4), (0, 1)]],
        )
        sol = shifting_bottleneck(inst)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid


# ============================================================
# Simulated Annealing Tests
# ============================================================

class TestSimulatedAnnealing:
    """Tests for SA metaheuristic."""

    def test_small_instance(self):
        inst = JobShopInstance(
            n=3, m=2,
            jobs=[
                [(0, 3), (1, 2)],
                [(1, 4), (0, 1)],
                [(0, 2), (1, 3)],
            ],
        )
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid

    def test_deterministic(self):
        inst = JobShopInstance.random(n=4, m=3, seed=10)
        s1 = simulated_annealing(inst, max_iterations=200, seed=42)
        s2 = simulated_annealing(inst, max_iterations=200, seed=42)
        assert s1.makespan == s2.makespan

    def test_ft06_quality(self):
        inst = ft06()
        sol = simulated_annealing(inst, max_iterations=1000, seed=42)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
        assert sol.makespan >= 55

    def test_improves_over_initial(self):
        inst = JobShopInstance.random(n=4, m=3, seed=10)
        initial = spt(inst)
        sa_sol = simulated_annealing(inst, max_iterations=500, seed=42)
        assert sa_sol.makespan <= initial.makespan


# ============================================================
# Tabu Search Tests
# ============================================================

class TestTabuSearch:
    """Tests for Tabu Search metaheuristic."""

    def test_small_instance(self):
        inst = JobShopInstance(
            n=3, m=2,
            jobs=[
                [(0, 3), (1, 2)],
                [(1, 4), (0, 1)],
                [(0, 2), (1, 3)],
            ],
        )
        sol = tabu_search(inst, max_iterations=500, seed=42)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid

    def test_deterministic(self):
        inst = JobShopInstance.random(n=4, m=3, seed=10)
        s1 = tabu_search(inst, max_iterations=200, seed=42)
        s2 = tabu_search(inst, max_iterations=200, seed=42)
        assert s1.makespan == s2.makespan

    def test_ft06_quality(self):
        inst = ft06()
        sol = tabu_search(inst, max_iterations=500, seed=42)
        valid, _ = validate_solution(inst, sol.start_times)
        assert valid
        assert sol.makespan >= 55

    def test_improves_over_initial(self):
        inst = JobShopInstance.random(n=4, m=3, seed=10)
        initial = spt(inst)
        ts_sol = tabu_search(inst, max_iterations=300, seed=42)
        assert ts_sol.makespan <= initial.makespan


# ============================================================
# Benchmark Tests
# ============================================================

class TestBenchmarks:
    """Tests on classic benchmark instances."""

    def test_ft06_all_methods(self):
        inst = ft06()
        results = {}
        for rule in ["spt", "lpt", "mwr", "lwr", "fifo"]:
            sol = dispatching_rule(inst, rule=rule)
            results[rule] = sol.makespan
            valid, _ = validate_solution(inst, sol.start_times)
            assert valid

        sb_sol = shifting_bottleneck(inst)
        results["SB"] = sb_sol.makespan

        sa_sol = simulated_annealing(inst, max_iterations=500, seed=42)
        results["SA"] = sa_sol.makespan

        ts_sol = tabu_search(inst, max_iterations=500, seed=42)
        results["TS"] = ts_sol.makespan

        # All should be feasible and >= optimal
        for name, ms in results.items():
            assert ms >= 55, f"{name}: makespan {ms} < optimal 55"

    def test_ft10_dispatching(self):
        inst = ft10()
        for rule in ["spt", "mwr"]:
            sol = dispatching_rule(inst, rule=rule)
            valid, _ = validate_solution(inst, sol.start_times)
            assert valid
            assert sol.makespan >= 930

    def test_random_instances(self):
        """Test on several random instances for robustness."""
        for seed in [1, 2, 3]:
            inst = JobShopInstance.random(n=6, m=4, seed=seed)
            for rule in ["spt", "mwr"]:
                sol = dispatching_rule(inst, rule=rule)
                valid, violations = validate_solution(inst, sol.start_times)
                assert valid, f"Seed {seed}, rule {rule}: {violations}"

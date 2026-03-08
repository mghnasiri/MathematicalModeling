"""
Test suite for VRPTW (Vehicle Routing Problem with Time Windows).

Tests cover:
- Instance creation and validation
- Solomon insertion heuristic and nearest neighbor
- Simulated annealing
- Genetic algorithm
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import importlib.util

# ── Module loading ───────────────────────────────────────────────────────────

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_instance_mod = _load_module(
    "vrptw_instance_test", os.path.join(_base_dir, "instance.py")
)
_si_mod = _load_module(
    "vrptw_si_test",
    os.path.join(_base_dir, "heuristics", "solomon_insertion.py"),
)
_sa_mod = _load_module(
    "vrptw_sa_test",
    os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py"),
)
_ga_mod = _load_module(
    "vrptw_ga_test",
    os.path.join(_base_dir, "metaheuristics", "genetic_algorithm.py"),
)

VRPTWInstance = _instance_mod.VRPTWInstance
VRPTWSolution = _instance_mod.VRPTWSolution
validate_solution = _instance_mod.validate_solution
solomon_c101_mini = _instance_mod.solomon_c101_mini
tight_tw5 = _instance_mod.tight_tw5

solomon_insertion = _si_mod.solomon_insertion
nearest_neighbor_tw = _si_mod.nearest_neighbor_tw
simulated_annealing = _sa_mod.simulated_annealing
genetic_algorithm = _ga_mod.genetic_algorithm


# ── Test fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inst_c101():
    return solomon_c101_mini()


@pytest.fixture
def inst_tw5():
    return tight_tw5()


@pytest.fixture
def random_inst():
    return VRPTWInstance.random(10, capacity=80.0, horizon=400.0, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestVRPTWInstance:
    def test_create_basic(self, inst_c101):
        assert inst_c101.n == 8
        assert inst_c101.capacity == 40.0
        assert inst_c101.demands.shape == (8,)
        assert inst_c101.distance_matrix.shape == (9, 9)
        assert inst_c101.time_windows.shape == (9, 2)
        assert inst_c101.service_times.shape == (9,)

    def test_random_instance(self):
        inst = VRPTWInstance.random(15, capacity=100.0, seed=123)
        assert inst.n == 15
        assert inst.capacity == 100.0
        assert inst.distance_matrix.shape == (16, 16)
        assert inst.time_windows.shape == (16, 2)
        assert np.all(inst.demands <= inst.capacity)

    def test_route_distance(self, inst_tw5):
        d = inst_tw5.route_distance([1])
        expected = inst_tw5.distance_matrix[0][1] + inst_tw5.distance_matrix[1][0]
        assert abs(d - expected) < 1e-10

    def test_route_demand(self, inst_tw5):
        demand = inst_tw5.route_demand([1, 2])
        assert abs(demand - 13.0) < 1e-10  # 5 + 8

    def test_empty_route(self, inst_tw5):
        assert inst_tw5.route_distance([]) == 0.0
        assert inst_tw5.route_feasible([])

    def test_route_schedule(self, inst_tw5):
        schedule = inst_tw5.route_schedule([1])
        assert len(schedule) == 1
        # Travel from depot to customer 1: dist=10, tw=[10,40]
        assert schedule[0] >= inst_tw5.time_windows[1][0]

    def test_route_feasible(self, inst_tw5):
        # Single customer with enough capacity should be feasible
        assert inst_tw5.route_feasible([1])

    def test_invalid_demands_shape(self):
        with pytest.raises(ValueError):
            VRPTWInstance(
                n=3, capacity=10.0,
                demands=np.array([1.0, 2.0]),
                distance_matrix=np.zeros((4, 4)),
                time_windows=np.zeros((4, 2)),
                service_times=np.zeros(4),
            )

    def test_demand_exceeds_capacity(self):
        with pytest.raises(ValueError):
            VRPTWInstance(
                n=2, capacity=5.0,
                demands=np.array([3.0, 10.0]),
                distance_matrix=np.zeros((3, 3)),
                time_windows=np.zeros((3, 2)),
                service_times=np.zeros(3),
            )


class TestValidation:
    def test_valid_solution(self, inst_tw5):
        sol = nearest_neighbor_tw(inst_tw5)
        valid, errors = validate_solution(inst_tw5, sol)
        assert valid, errors

    def test_missing_customer(self, inst_tw5):
        sol = VRPTWSolution(routes=[[1, 2], [3]], distance=0.0)
        valid, errors = validate_solution(inst_tw5, sol)
        assert not valid

    def test_duplicate_customer(self, inst_tw5):
        sol = VRPTWSolution(routes=[[1, 2, 3], [3, 4, 5]], distance=0.0)
        valid, errors = validate_solution(inst_tw5, sol)
        assert not valid


# ── Solomon insertion tests ──────────────────────────────────────────────────


class TestSolomonInsertion:
    def test_feasible_solution(self, inst_c101):
        sol = solomon_insertion(inst_c101)
        valid, errors = validate_solution(inst_c101, sol)
        assert valid, errors

    def test_all_customers_visited(self, inst_c101):
        sol = solomon_insertion(inst_c101)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, inst_c101.n + 1))

    def test_distance_matches(self, inst_c101):
        sol = solomon_insertion(inst_c101)
        expected = inst_c101.total_distance(sol.routes)
        assert abs(sol.distance - expected) < 1e-10

    def test_tight_windows(self, inst_tw5):
        sol = solomon_insertion(inst_tw5)
        valid, errors = validate_solution(inst_tw5, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = solomon_insertion(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_different_parameters(self, inst_c101):
        sol1 = solomon_insertion(inst_c101, alpha1=1.0, alpha2=0.0, lam=1.0)
        sol2 = solomon_insertion(inst_c101, alpha1=0.5, alpha2=0.5, lam=2.0)
        # Both should be feasible
        valid1, _ = validate_solution(inst_c101, sol1)
        valid2, _ = validate_solution(inst_c101, sol2)
        assert valid1
        assert valid2


class TestNearestNeighborTW:
    def test_feasible_solution(self, inst_c101):
        sol = nearest_neighbor_tw(inst_c101)
        valid, errors = validate_solution(inst_c101, sol)
        assert valid, errors

    def test_all_customers_visited(self, inst_tw5):
        sol = nearest_neighbor_tw(inst_tw5)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, inst_tw5.n + 1))

    def test_random_instance(self, random_inst):
        sol = nearest_neighbor_tw(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── Simulated Annealing tests ────────────────────────────────────────────────


class TestSimulatedAnnealing:
    def test_feasible_solution(self, inst_c101):
        sol = simulated_annealing(inst_c101, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst_c101, sol)
        assert valid, errors

    def test_deterministic_with_seed(self, inst_tw5):
        sol1 = simulated_annealing(inst_tw5, max_iterations=3000, seed=42)
        sol2 = simulated_annealing(inst_tw5, max_iterations=3000, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_random_instance(self, random_inst):
        sol = simulated_annealing(random_inst, max_iterations=5000, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_competitive_with_heuristic(self, inst_c101):
        si_sol = solomon_insertion(inst_c101)
        sa_sol = simulated_annealing(inst_c101, max_iterations=10000, seed=42)
        # SA should be within 30% of Solomon (it may use more/fewer vehicles)
        assert sa_sol.distance < si_sol.distance * 1.3


# ── Genetic Algorithm tests ──────────────────────────────────────────────────


class TestGeneticAlgorithm:
    def test_feasible_solution(self, inst_c101):
        sol = genetic_algorithm(inst_c101, pop_size=20, generations=50, seed=42)
        valid, errors = validate_solution(inst_c101, sol)
        assert valid, errors

    def test_deterministic_with_seed(self, inst_tw5):
        sol1 = genetic_algorithm(inst_tw5, pop_size=20, generations=50, seed=42)
        sol2 = genetic_algorithm(inst_tw5, pop_size=20, generations=50, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_random_instance(self, random_inst):
        sol = genetic_algorithm(
            random_inst, pop_size=20, generations=50, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_all_customers_served(self, inst_c101):
        sol = genetic_algorithm(inst_c101, pop_size=20, generations=50, seed=42)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, inst_c101.n + 1))


# ── Cross-method comparison tests ────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_all_methods_valid(self, inst_c101):
        methods = [
            lambda i: solomon_insertion(i),
            lambda i: nearest_neighbor_tw(i),
            lambda i: simulated_annealing(i, max_iterations=3000, seed=42),
            lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42),
        ]
        for method in methods:
            sol = method(inst_c101)
            valid, errors = validate_solution(inst_c101, sol)
            assert valid, errors

    def test_all_methods_reasonable_cost(self, random_inst):
        methods = {
            "solomon": lambda i: solomon_insertion(i),
            "nn_tw": lambda i: nearest_neighbor_tw(i),
            "SA": lambda i: simulated_annealing(i, max_iterations=5000, seed=42),
            "GA": lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42),
        }
        costs = {}
        for name, method in methods.items():
            sol = method(random_inst)
            valid, _ = validate_solution(random_inst, sol)
            assert valid
            costs[name] = sol.distance

        min_cost = min(costs.values())
        max_cost = max(costs.values())
        assert max_cost < min_cost * 3.0  # All within 3x (TW makes it harder)

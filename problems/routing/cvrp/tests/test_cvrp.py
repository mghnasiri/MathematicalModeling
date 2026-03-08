"""
Test suite for CVRP (Capacitated Vehicle Routing Problem) algorithms.

Tests cover:
- Instance creation and validation
- Clarke-Wright savings heuristic
- Sweep algorithm
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
    "cvrp_instance", os.path.join(_base_dir, "instance.py")
)
_cw_mod = _load_module(
    "cvrp_cw", os.path.join(_base_dir, "heuristics", "clarke_wright.py")
)
_sweep_mod = _load_module(
    "cvrp_sweep", os.path.join(_base_dir, "heuristics", "sweep.py")
)
_sa_mod = _load_module(
    "cvrp_sa", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py")
)
_ga_mod = _load_module(
    "cvrp_ga", os.path.join(_base_dir, "metaheuristics", "genetic_algorithm.py")
)

CVRPInstance = _instance_mod.CVRPInstance
CVRPSolution = _instance_mod.CVRPSolution
validate_solution = _instance_mod.validate_solution
small6 = _instance_mod.small6
christofides1 = _instance_mod.christofides1
medium12 = _instance_mod.medium12

clarke_wright_savings = _cw_mod.clarke_wright_savings
sweep = _sweep_mod.sweep
sweep_multistart = _sweep_mod.sweep_multistart
simulated_annealing = _sa_mod.simulated_annealing
genetic_algorithm = _ga_mod.genetic_algorithm


# ── Test fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inst6():
    return small6()


@pytest.fixture
def inst_c1():
    return christofides1()


@pytest.fixture
def inst12():
    return medium12()


@pytest.fixture
def random_inst():
    return CVRPInstance.random(15, capacity=50.0, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestCVRPInstance:
    def test_create_basic(self, inst6):
        assert inst6.n == 6
        assert inst6.capacity == 15.0
        assert inst6.demands.shape == (6,)
        assert inst6.distance_matrix.shape == (7, 7)

    def test_random_instance(self):
        inst = CVRPInstance.random(20, capacity=100.0, seed=123)
        assert inst.n == 20
        assert inst.capacity == 100.0
        assert inst.demands.shape == (20,)
        assert inst.distance_matrix.shape == (21, 21)
        assert inst.coords is not None
        assert np.all(inst.demands <= inst.capacity)

    def test_from_coordinates(self):
        coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
        demands = [3, 4, 5]
        inst = CVRPInstance.from_coordinates(coords, demands, capacity=10.0)
        assert inst.n == 3
        assert inst.capacity == 10.0

    def test_route_distance(self, inst_c1):
        # Single customer route: depot -> 1 -> depot
        d = inst_c1.route_distance([1])
        assert abs(d - 6.0) < 1e-10  # d(0,1) + d(1,0) = 3 + 3

    def test_route_demand(self, inst_c1):
        # Customer 1 -> demands[0]=2, Customer 3 -> demands[2]=2
        demand = inst_c1.route_demand([1, 3])
        assert abs(demand - 4.0) < 1e-10  # 2 + 2

    def test_empty_route_distance(self, inst_c1):
        assert inst_c1.route_distance([]) == 0.0

    def test_total_distance(self, inst_c1):
        routes = [[1], [2, 3]]
        total = inst_c1.total_distance(routes)
        r1 = inst_c1.route_distance([1])
        r2 = inst_c1.route_distance([2, 3])
        assert abs(total - (r1 + r2)) < 1e-10

    def test_demand_exceeds_capacity(self):
        with pytest.raises(ValueError):
            CVRPInstance(
                n=2,
                capacity=5.0,
                demands=np.array([3.0, 10.0]),
                distance_matrix=np.zeros((3, 3)),
            )

    def test_invalid_demands_shape(self):
        with pytest.raises(ValueError):
            CVRPInstance(
                n=3,
                capacity=10.0,
                demands=np.array([1.0, 2.0]),
                distance_matrix=np.zeros((4, 4)),
            )

    def test_invalid_distance_matrix_shape(self):
        with pytest.raises(ValueError):
            CVRPInstance(
                n=3,
                capacity=10.0,
                demands=np.array([1.0, 2.0, 3.0]),
                distance_matrix=np.zeros((3, 3)),
            )


class TestValidation:
    def test_valid_solution(self, inst_c1):
        # demands = [2,3,2,3,2], capacity = 6
        # Route [1,3]: 2+2=4 <= 6, Route [2,4]: 3+3=6 <= 6, Route [5]: 2 <= 6
        routes = [[1, 3], [2, 4], [5]]
        sol = CVRPSolution(
            routes=routes,
            distance=inst_c1.total_distance(routes),
        )
        valid, errors = validate_solution(inst_c1, sol)
        assert valid, errors

    def test_missing_customer(self, inst_c1):
        sol = CVRPSolution(routes=[[1, 2], [3, 4]], distance=0.0)
        valid, errors = validate_solution(inst_c1, sol)
        assert not valid

    def test_duplicate_customer(self, inst_c1):
        sol = CVRPSolution(routes=[[1, 2, 3], [3, 4, 5]], distance=0.0)
        valid, errors = validate_solution(inst_c1, sol)
        assert not valid

    def test_capacity_violation(self, inst_c1):
        # All 5 customers in one route: demand = 2+3+2+3+2 = 12 > 6
        sol = CVRPSolution(routes=[[1, 2, 3, 4, 5]], distance=0.0)
        valid, errors = validate_solution(inst_c1, sol)
        assert not valid
        assert any("capacity" in e.lower() for e in errors)

    def test_num_vehicles(self):
        dist = np.zeros((4, 4))
        inst = CVRPInstance(
            n=3,
            capacity=10.0,
            demands=np.array([5.0, 5.0, 5.0]),
            distance_matrix=dist,
            num_vehicles=2,
        )
        sol = CVRPSolution(routes=[[1], [2], [3]], distance=0.0)
        valid, errors = validate_solution(inst, sol)
        assert not valid
        assert any("vehicles" in e.lower() for e in errors)


# ── Clarke-Wright tests ──────────────────────────────────────────────────────


class TestClarkeWright:
    def test_feasible_solution(self, inst_c1):
        sol = clarke_wright_savings(inst_c1)
        valid, errors = validate_solution(inst_c1, sol)
        assert valid, errors

    def test_capacity_respected(self, inst_c1):
        sol = clarke_wright_savings(inst_c1)
        for route in sol.routes:
            assert inst_c1.route_demand(route) <= inst_c1.capacity + 1e-10

    def test_all_customers_visited(self, inst6):
        sol = clarke_wright_savings(inst6)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, inst6.n + 1))

    def test_distance_matches(self, inst_c1):
        sol = clarke_wright_savings(inst_c1)
        expected = inst_c1.total_distance(sol.routes)
        assert abs(sol.distance - expected) < 1e-10

    def test_medium_instance(self, inst12):
        sol = clarke_wright_savings(inst12)
        valid, errors = validate_solution(inst12, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = clarke_wright_savings(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_single_customer(self):
        dist = np.array([[0, 5], [5, 0]], dtype=float)
        inst = CVRPInstance(
            n=1, capacity=10.0,
            demands=np.array([5.0]),
            distance_matrix=dist,
        )
        sol = clarke_wright_savings(inst)
        valid, _ = validate_solution(inst, sol)
        assert valid
        assert abs(sol.distance - 10.0) < 1e-10


# ── Sweep tests ──────────────────────────────────────────────────────────────


class TestSweep:
    def test_feasible_solution(self, inst6):
        sol = sweep(inst6)
        valid, errors = validate_solution(inst6, sol)
        assert valid, errors

    def test_capacity_respected(self, inst6):
        sol = sweep(inst6)
        for route in sol.routes:
            assert inst6.route_demand(route) <= inst6.capacity + 1e-10

    def test_requires_coords(self, inst_c1):
        # christofides1 has no coords
        with pytest.raises(ValueError):
            sweep(inst_c1)

    def test_multistart_improves(self, inst6):
        sol_single = sweep(inst6)
        sol_multi = sweep_multistart(inst6)
        assert sol_multi.distance <= sol_single.distance + 1e-10

    def test_medium_instance(self, inst12):
        sol = sweep(inst12)
        valid, errors = validate_solution(inst12, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = sweep(random_inst)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── Simulated Annealing tests ────────────────────────────────────────────────


class TestSimulatedAnnealing:
    def test_feasible_solution(self, inst_c1):
        sol = simulated_annealing(inst_c1, max_iterations=5000, seed=42)
        valid, errors = validate_solution(inst_c1, sol)
        assert valid, errors

    def test_deterministic_with_seed(self, inst_c1):
        sol1 = simulated_annealing(inst_c1, max_iterations=5000, seed=42)
        sol2 = simulated_annealing(inst_c1, max_iterations=5000, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_improves_over_cw(self, inst12):
        cw_sol = clarke_wright_savings(inst12)
        sa_sol = simulated_annealing(inst12, max_iterations=20000, seed=42)
        # SA should generally improve or be competitive
        assert sa_sol.distance <= cw_sol.distance * 1.15

    def test_medium_instance(self, inst12):
        sol = simulated_annealing(inst12, max_iterations=10000, seed=42)
        valid, errors = validate_solution(inst12, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = simulated_annealing(random_inst, max_iterations=10000, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors


# ── Genetic Algorithm tests ──────────────────────────────────────────────────


class TestGeneticAlgorithm:
    def test_feasible_solution(self, inst_c1):
        sol = genetic_algorithm(inst_c1, pop_size=20, generations=50, seed=42)
        valid, errors = validate_solution(inst_c1, sol)
        assert valid, errors

    def test_deterministic_with_seed(self, inst_c1):
        sol1 = genetic_algorithm(inst_c1, pop_size=20, generations=50, seed=42)
        sol2 = genetic_algorithm(inst_c1, pop_size=20, generations=50, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_medium_instance(self, inst12):
        sol = genetic_algorithm(inst12, pop_size=30, generations=100, seed=42)
        valid, errors = validate_solution(inst12, sol)
        assert valid, errors

    def test_random_instance(self, random_inst):
        sol = genetic_algorithm(random_inst, pop_size=30, generations=100, seed=42)
        valid, errors = validate_solution(random_inst, sol)
        assert valid, errors

    def test_all_customers_served(self, inst6):
        sol = genetic_algorithm(inst6, pop_size=20, generations=50, seed=42)
        all_customers = [c for r in sol.routes for c in r]
        assert sorted(all_customers) == list(range(1, inst6.n + 1))


# ── Cross-method comparison tests ────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_all_methods_valid(self, inst6):
        methods = [
            lambda i: clarke_wright_savings(i),
            lambda i: sweep(i),
            lambda i: simulated_annealing(i, max_iterations=5000, seed=42),
            lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42),
        ]
        for method in methods:
            sol = method(inst6)
            valid, errors = validate_solution(inst6, sol)
            assert valid, errors

    def test_all_methods_reasonable_cost(self, inst12):
        methods = {
            "CW": lambda i: clarke_wright_savings(i),
            "sweep": lambda i: sweep_multistart(i),
            "SA": lambda i: simulated_annealing(i, max_iterations=10000, seed=42),
            "GA": lambda i: genetic_algorithm(i, pop_size=30, generations=100, seed=42),
        }
        costs = {}
        for name, method in methods.items():
            sol = method(inst12)
            valid, _ = validate_solution(inst12, sol)
            assert valid
            costs[name] = sol.distance

        # All methods should produce reasonable results (within 2x of each other)
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        assert max_cost < min_cost * 2.0

    def test_metaheuristics_competitive(self, random_inst):
        cw_sol = clarke_wright_savings(random_inst)
        sa_sol = simulated_annealing(random_inst, max_iterations=15000, seed=42)
        ga_sol = genetic_algorithm(
            random_inst, pop_size=30, generations=100, seed=42
        )

        # Metaheuristics should be within 30% of CW
        assert sa_sol.distance < cw_sol.distance * 1.3
        assert ga_sol.distance < cw_sol.distance * 1.3

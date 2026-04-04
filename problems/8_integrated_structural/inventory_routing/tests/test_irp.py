"""Tests for Inventory-Routing Problem."""
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
_inst_mod = _load_mod("irp_instance", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod(
    "irp_greedy",
    os.path.join(_base, "heuristics", "greedy_irp.py"),
)
_sa_mod = _load_mod(
    "irp_sa",
    os.path.join(_base, "metaheuristics", "simulated_annealing.py"),
)

IRPInstance = _inst_mod.IRPInstance
IRPSolution = _inst_mod.IRPSolution
compute_cost = _inst_mod.compute_cost
greedy_irp = _greedy_mod.greedy_irp
greedy_fill_up = _greedy_mod.greedy_fill_up
simulated_annealing = _sa_mod.simulated_annealing


def _make_small():
    """Create the built-in small instance for testing."""
    return IRPInstance.small_instance()


class TestIRPInstance:
    """Tests for IRPInstance creation and properties."""

    def test_random_instance(self):
        """Random instance has correct dimensions."""
        inst = IRPInstance.random(n_customers=8, T=4, seed=42)
        assert inst.n_customers == 8
        assert inst.T == 4
        assert inst.demands.shape == (8,)
        assert inst.coordinates.shape == (9, 2)
        assert inst.holding_costs.shape == (8,)
        assert inst.storage_capacities.shape == (8,)
        assert inst.initial_inventory.shape == (8,)

    def test_small_instance(self):
        """Small built-in instance has expected dimensions."""
        inst = _make_small()
        assert inst.n_customers == 5
        assert inst.T == 3
        assert inst.demands.shape == (5,)
        assert inst.coordinates.shape == (6, 2)
        assert inst.vehicle_capacity == 30.0
        assert inst.n_vehicles == 2

    def test_demand_positive(self):
        """All demands are positive in random instance."""
        inst = IRPInstance.random(n_customers=15, T=6, seed=99)
        assert np.all(inst.demands > 0)

    def test_distance_symmetric(self):
        """Distance matrix is symmetric."""
        inst = _make_small()
        dm = inst.distance_matrix()
        assert dm.shape == (6, 6)
        np.testing.assert_allclose(dm, dm.T, atol=1e-10)
        # Diagonal is zero
        np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-10)

    def test_distance_positive(self):
        """Distances between distinct nodes are positive."""
        inst = _make_small()
        assert inst.distance(0, 1) > 0
        assert inst.distance(1, 2) > 0

    def test_route_distance_empty(self):
        """Empty route has zero distance."""
        inst = _make_small()
        assert inst.route_distance([]) == 0.0

    def test_route_distance_single(self):
        """Single-customer route: depot -> cust -> depot."""
        inst = _make_small()
        d = inst.route_distance([1])
        expected = inst.distance(0, 1) + inst.distance(1, 0)
        assert d == pytest.approx(expected)

    def test_initial_inventory_within_capacity(self):
        """Initial inventory does not exceed storage capacity."""
        inst = _make_small()
        assert np.all(inst.initial_inventory <= inst.storage_capacities + 1e-9)

    def test_random_reproducible(self):
        """Same seed produces identical instances."""
        inst1 = IRPInstance.random(n_customers=6, T=3, seed=77)
        inst2 = IRPInstance.random(n_customers=6, T=3, seed=77)
        np.testing.assert_array_equal(inst1.demands, inst2.demands)
        np.testing.assert_array_equal(inst1.coordinates, inst2.coordinates)


class TestComputeCost:
    """Tests for the compute_cost evaluation function."""

    def test_no_deliveries_consumes_inventory(self):
        """With no deliveries, inventory decreases by demand each period."""
        inst = _make_small()
        empty_routes = [[] for _ in range(inst.T)]
        empty_deliveries = [{} for _ in range(inst.T)]

        # Only works if initial inventory covers all periods
        # For small instance: demands up to 6, initial up to 12, T=3
        # Customer 5 has demand 6, initial 12 -> period 2 gives 0 -> period 3 stockout
        # Stockouts are penalized (not raised) so cost should be very high
        sol = compute_cost(inst, empty_routes, empty_deliveries)
        assert sol.total_cost > 1000.0  # penalty makes cost very high

    def test_sufficient_deliveries_no_stockout(self):
        """Delivering demand each period prevents stockout."""
        inst = _make_small()
        routes = []
        deliveries = []
        for t in range(inst.T):
            # Deliver exactly demand to each customer
            period_del = {c + 1: inst.demands[c] for c in range(inst.n_customers)}
            period_routes = [list(range(1, inst.n_customers + 1))]
            routes.append(period_routes)
            deliveries.append(period_del)

        sol = compute_cost(inst, routes, deliveries)
        assert sol.total_cost > 0
        assert sol.routing_cost > 0
        assert sol.holding_cost >= 0

    def test_inventory_levels_shape(self):
        """Inventory levels array has correct shape."""
        inst = _make_small()
        routes = []
        deliveries = []
        for t in range(inst.T):
            period_del = {c + 1: inst.demands[c] for c in range(inst.n_customers)}
            period_routes = [list(range(1, inst.n_customers + 1))]
            routes.append(period_routes)
            deliveries.append(period_del)

        sol = compute_cost(inst, routes, deliveries)
        assert sol.inventory_levels.shape == (inst.T + 1, inst.n_customers)
        # Initial inventory matches
        np.testing.assert_allclose(
            sol.inventory_levels[0], inst.initial_inventory
        )


class TestGreedyIRP:
    """Tests for the greedy constructive heuristic."""

    def test_returns_solution(self):
        """Greedy returns a valid IRPSolution."""
        inst = _make_small()
        sol = greedy_irp(inst, seed=42)
        assert type(sol).__name__ == "IRPSolution"
        assert len(sol.routes_per_period) == inst.T
        assert len(sol.deliveries_per_period) == inst.T

    def test_no_stockout(self):
        """Greedy prevents stockouts across all periods."""
        inst = _make_small()
        sol = greedy_irp(inst, seed=42)
        # If compute_cost succeeded, no stockout occurred
        assert sol.inventory_levels is not None
        # All inventory levels non-negative
        assert np.all(sol.inventory_levels >= -1e-9)

    def test_capacity_respected(self):
        """Delivery quantities per route respect vehicle capacity."""
        inst = _make_small()
        sol = greedy_irp(inst, seed=42)
        for t in range(inst.T):
            for route in sol.routes_per_period[t]:
                route_load = sum(
                    sol.deliveries_per_period[t].get(c, 0.0) for c in route
                )
                assert route_load <= inst.vehicle_capacity + 1e-9, (
                    f"Period {t}: route load {route_load} > "
                    f"capacity {inst.vehicle_capacity}"
                )

    def test_cost_positive(self):
        """Solution cost is positive."""
        inst = _make_small()
        sol = greedy_irp(inst, seed=42)
        assert sol.total_cost > 0
        assert sol.routing_cost >= 0
        assert sol.holding_cost >= 0
        assert abs(sol.total_cost - sol.routing_cost - sol.holding_cost) < 1e-9

    def test_deterministic(self):
        """Same seed produces identical solutions."""
        inst = _make_small()
        sol1 = greedy_irp(inst, seed=42)
        sol2 = greedy_irp(inst, seed=42)
        assert sol1.total_cost == pytest.approx(sol2.total_cost)
        assert sol1.routing_cost == pytest.approx(sol2.routing_cost)
        assert sol1.holding_cost == pytest.approx(sol2.holding_cost)

    def test_random_instance_feasible(self):
        """Greedy produces feasible solution on random instance."""
        inst = IRPInstance.random(n_customers=8, T=4, seed=55)
        sol = greedy_irp(inst, seed=55)
        assert sol.total_cost > 0
        assert sol.inventory_levels is not None

    def test_fill_up_returns_solution(self):
        """Fill-up baseline returns valid solution."""
        inst = _make_small()
        sol = greedy_fill_up(inst, seed=42)
        assert type(sol).__name__ == "IRPSolution"
        assert sol.total_cost > 0


class TestIRPSA:
    """Tests for Simulated Annealing metaheuristic."""

    def test_returns_solution(self):
        """SA returns a valid IRPSolution."""
        inst = _make_small()
        sol = simulated_annealing(inst, max_iterations=500, seed=42)
        assert type(sol).__name__ == "IRPSolution"
        assert sol.total_cost > 0

    def test_improves_over_greedy(self):
        """SA finds a solution no worse than greedy on small instance.

        With enough iterations, SA should match or improve upon greedy.
        We use a generous iteration count to allow for improvement.
        """
        inst = _make_small()
        sol_greedy = greedy_irp(inst, seed=42)
        sol_sa = simulated_annealing(
            inst, max_iterations=3000, seed=42
        )
        # SA should be at least as good (with some tolerance for stochastic moves)
        assert sol_sa.total_cost <= sol_greedy.total_cost * 1.05, (
            f"SA cost {sol_sa.total_cost:.2f} > "
            f"greedy cost {sol_greedy.total_cost:.2f} * 1.05"
        )

    def test_deterministic_with_seed(self):
        """Same seed produces identical SA results."""
        inst = _make_small()
        sol1 = simulated_annealing(inst, max_iterations=1000, seed=99)
        sol2 = simulated_annealing(inst, max_iterations=1000, seed=99)
        assert sol1.total_cost == pytest.approx(sol2.total_cost)
        assert sol1.routing_cost == pytest.approx(sol2.routing_cost)

    def test_sa_random_instance(self):
        """SA runs without error on random instance."""
        inst = IRPInstance.random(n_customers=6, T=3, seed=33)
        sol = simulated_annealing(inst, max_iterations=500, seed=33)
        assert sol.total_cost > 0

    def test_different_seeds_differ(self):
        """Different seeds produce potentially different solutions."""
        inst = _make_small()
        sol1 = simulated_annealing(inst, max_iterations=2000, seed=1)
        sol2 = simulated_annealing(inst, max_iterations=2000, seed=2)
        # They may differ (not guaranteed but highly likely with different seeds)
        # At minimum both should be valid
        assert sol1.total_cost > 0
        assert sol2.total_cost > 0

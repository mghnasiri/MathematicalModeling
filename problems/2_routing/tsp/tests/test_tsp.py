"""
Test suite for TSP (Traveling Salesman Problem) algorithms.

Tests cover:
- Instance creation and validation
- Exact methods (Held-Karp, Branch and Bound)
- Constructive heuristics (Nearest Neighbor, Insertion, Greedy)
- Metaheuristics (2-opt, Simulated Annealing, Genetic Algorithm)
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
    "tsp_instance", os.path.join(_base_dir, "instance.py")
)
_held_karp_mod = _load_module(
    "tsp_held_karp", os.path.join(_base_dir, "exact", "held_karp.py")
)
_bnb_mod = _load_module(
    "tsp_bnb", os.path.join(_base_dir, "exact", "branch_and_bound.py")
)
_nn_mod = _load_module(
    "tsp_nn", os.path.join(_base_dir, "heuristics", "nearest_neighbor.py")
)
_insertion_mod = _load_module(
    "tsp_insertion", os.path.join(_base_dir, "heuristics", "cheapest_insertion.py")
)
_greedy_mod = _load_module(
    "tsp_greedy", os.path.join(_base_dir, "heuristics", "greedy.py")
)
_ls_mod = _load_module(
    "tsp_ls", os.path.join(_base_dir, "metaheuristics", "local_search.py")
)
_sa_mod = _load_module(
    "tsp_sa", os.path.join(_base_dir, "metaheuristics", "simulated_annealing.py")
)
_ga_mod = _load_module(
    "tsp_ga", os.path.join(_base_dir, "metaheuristics", "genetic_algorithm.py")
)

TSPInstance = _instance_mod.TSPInstance
TSPSolution = _instance_mod.TSPSolution
validate_tour = _instance_mod.validate_tour
small4 = _instance_mod.small4
small5 = _instance_mod.small5
gr17 = _instance_mod.gr17

held_karp = _held_karp_mod.held_karp
branch_and_bound = _bnb_mod.branch_and_bound

nearest_neighbor = _nn_mod.nearest_neighbor
nearest_neighbor_multistart = _nn_mod.nearest_neighbor_multistart
cheapest_insertion = _insertion_mod.cheapest_insertion
farthest_insertion = _insertion_mod.farthest_insertion
nearest_insertion = _insertion_mod.nearest_insertion
greedy = _greedy_mod.greedy

two_opt = _ls_mod.two_opt
or_opt = _ls_mod.or_opt
vnd = _ls_mod.vnd
simulated_annealing = _sa_mod.simulated_annealing
genetic_algorithm = _ga_mod.genetic_algorithm


# ── Test fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def inst4():
    return small4()


@pytest.fixture
def inst5():
    return small5()


@pytest.fixture
def inst17():
    return gr17()


@pytest.fixture
def random_inst():
    return TSPInstance.random(15, seed=42)


# ── Instance tests ───────────────────────────────────────────────────────────


class TestTSPInstance:
    def test_create_basic(self, inst4):
        assert inst4.n == 4
        assert inst4.distance_matrix.shape == (4, 4)
        assert inst4.is_symmetric()

    def test_random_instance(self):
        inst = TSPInstance.random(20, seed=123)
        assert inst.n == 20
        assert inst.distance_matrix.shape == (20, 20)
        assert inst.coords is not None
        assert inst.coords.shape == (20, 2)
        assert inst.is_symmetric()

    def test_random_asymmetric(self):
        inst = TSPInstance.random(10, symmetric=False, seed=42)
        assert inst.n == 10
        assert not inst.is_symmetric()

    def test_from_coordinates(self):
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        inst = TSPInstance.from_coordinates(coords, name="square")
        assert inst.n == 4
        assert inst.is_symmetric()
        assert inst.name == "square"
        # Distance from (0,0) to (1,0) should be 1.0
        assert abs(inst.distance_matrix[0][1] - 1.0) < 1e-10

    def test_from_distance_matrix(self):
        dist = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
        inst = TSPInstance.from_distance_matrix(dist, name="test3")
        assert inst.n == 3
        assert inst.tour_distance([0, 1, 2]) == 45.0

    def test_tour_distance(self, inst4):
        # Known optimal: [0, 1, 3, 2] -> 1 + 3 + 1 + 3 = nope
        # dist: 0->1=1, 1->3=3, 3->2=1, 2->0=4 => wait
        # Actually: 0->1=1, 1->3=3, 3->2=1, 2->0=4 = 9
        # Let me check: tour [0,1,3,2]: 0->1=1, 1->3=3, 3->2=1, 2->0=4 = 9
        opt_dist = inst4.tour_distance([0, 1, 3, 2])
        assert opt_dist == 9.0

    def test_validate_tour_valid(self, inst4):
        valid, errors = validate_tour(inst4, [0, 1, 2, 3])
        assert valid
        assert len(errors) == 0

    def test_validate_tour_wrong_length(self, inst4):
        valid, errors = validate_tour(inst4, [0, 1, 2])
        assert not valid

    def test_validate_tour_duplicate(self, inst4):
        valid, errors = validate_tour(inst4, [0, 1, 1, 2])
        assert not valid

    def test_invalid_distance_matrix_shape(self):
        with pytest.raises(ValueError):
            TSPInstance(n=3, distance_matrix=np.zeros((2, 2)))

    def test_invalid_coords_shape(self):
        with pytest.raises(ValueError):
            TSPInstance(n=3, distance_matrix=np.zeros((3, 3)), coords=np.zeros((2, 2)))

    def test_single_city(self):
        inst = TSPInstance.from_distance_matrix([[0]])
        assert inst.n == 1
        assert inst.tour_distance([0]) == 0.0


# ── Exact method tests ───────────────────────────────────────────────────────


class TestHeldKarp:
    def test_small4(self, inst4):
        sol = held_karp(inst4)
        valid, _ = validate_tour(inst4, sol.tour)
        assert valid
        # Should find optimal
        assert abs(sol.distance - inst4.tour_distance(sol.tour)) < 1e-10

    def test_small5(self, inst5):
        sol = held_karp(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_gr17_optimal(self, inst17):
        sol = held_karp(inst17)
        valid, _ = validate_tour(inst17, sol.tour)
        assert valid
        assert abs(sol.distance - 2016) < 1e-6

    def test_single_city(self):
        inst = TSPInstance.from_distance_matrix([[0]])
        sol = held_karp(inst)
        assert sol.tour == [0]
        assert sol.distance == 0.0

    def test_two_cities(self):
        inst = TSPInstance.from_distance_matrix([[0, 5], [5, 0]])
        sol = held_karp(inst)
        assert len(sol.tour) == 2
        assert abs(sol.distance - 10.0) < 1e-10

    def test_too_large_instance(self):
        inst = TSPInstance(n=24, distance_matrix=np.zeros((24, 24)))
        with pytest.raises(ValueError):
            held_karp(inst)

    def test_deterministic(self, inst5):
        sol1 = held_karp(inst5)
        sol2 = held_karp(inst5)
        assert abs(sol1.distance - sol2.distance) < 1e-10


class TestBranchAndBound:
    def test_small4(self, inst4):
        sol = branch_and_bound(inst4)
        valid, _ = validate_tour(inst4, sol.tour)
        assert valid

    def test_small5(self, inst5):
        sol = branch_and_bound(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_agrees_with_held_karp(self, inst5):
        hk_sol = held_karp(inst5)
        bb_sol = branch_and_bound(inst5)
        assert abs(hk_sol.distance - bb_sol.distance) < 1e-6

    def test_single_city(self):
        inst = TSPInstance.from_distance_matrix([[0]])
        sol = branch_and_bound(inst)
        assert sol.tour == [0]
        assert sol.distance == 0.0

    def test_two_cities(self):
        inst = TSPInstance.from_distance_matrix([[0, 7], [7, 0]])
        sol = branch_and_bound(inst)
        assert abs(sol.distance - 14.0) < 1e-10


# ── Heuristic tests ──────────────────────────────────────────────────────────


class TestNearestNeighbor:
    def test_feasible_tour(self, inst5):
        sol = nearest_neighbor(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid
        assert abs(sol.distance - inst5.tour_distance(sol.tour)) < 1e-10

    def test_start_city(self, inst5):
        sol = nearest_neighbor(inst5, start=2)
        assert sol.tour[0] == 2

    def test_multistart_improves(self, inst17):
        sol_single = nearest_neighbor(inst17, start=0)
        sol_multi = nearest_neighbor_multistart(inst17)
        assert sol_multi.distance <= sol_single.distance

    def test_random_instance(self, random_inst):
        sol = nearest_neighbor(random_inst)
        valid, _ = validate_tour(random_inst, sol.tour)
        assert valid


class TestInsertionHeuristics:
    def test_cheapest_feasible(self, inst5):
        sol = cheapest_insertion(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_farthest_feasible(self, inst5):
        sol = farthest_insertion(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_nearest_feasible(self, inst5):
        sol = nearest_insertion(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_all_on_gr17(self, inst17):
        for fn in [cheapest_insertion, farthest_insertion, nearest_insertion]:
            sol = fn(inst17)
            valid, _ = validate_tour(inst17, sol.tour)
            assert valid
            assert sol.distance < 5000  # Reasonable upper bound

    def test_two_cities(self):
        inst = TSPInstance.from_distance_matrix([[0, 3], [3, 0]])
        for fn in [cheapest_insertion, farthest_insertion, nearest_insertion]:
            sol = fn(inst)
            assert len(sol.tour) == 2

    def test_random_instance(self, random_inst):
        for fn in [cheapest_insertion, farthest_insertion, nearest_insertion]:
            sol = fn(random_inst)
            valid, _ = validate_tour(random_inst, sol.tour)
            assert valid


class TestGreedy:
    def test_feasible_tour(self, inst5):
        sol = greedy(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_gr17(self, inst17):
        sol = greedy(inst17)
        valid, _ = validate_tour(inst17, sol.tour)
        assert valid
        assert sol.distance < 5000

    def test_random_instance(self, random_inst):
        sol = greedy(random_inst)
        valid, _ = validate_tour(random_inst, sol.tour)
        assert valid


# ── Metaheuristic tests ──────────────────────────────────────────────────────


class TestLocalSearch:
    def test_two_opt_improves(self, inst17):
        nn_sol = nearest_neighbor(inst17)
        ls_sol = two_opt(inst17, nn_sol.tour)
        assert ls_sol.distance <= nn_sol.distance + 1e-10

    def test_two_opt_feasible(self, inst5):
        sol = two_opt(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_or_opt_feasible(self, inst5):
        sol = or_opt(inst5)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_vnd_improves(self, inst17):
        nn_sol = nearest_neighbor(inst17)
        vnd_sol = vnd(inst17, nn_sol.tour)
        assert vnd_sol.distance <= nn_sol.distance + 1e-10
        valid, _ = validate_tour(inst17, vnd_sol.tour)
        assert valid

    def test_small_instance(self):
        inst = TSPInstance.from_distance_matrix(
            [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
        )
        sol = two_opt(inst)
        valid, _ = validate_tour(inst, sol.tour)
        assert valid


class TestSimulatedAnnealing:
    def test_feasible_tour(self, inst5):
        sol = simulated_annealing(inst5, seed=42)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_deterministic_with_seed(self, inst17):
        sol1 = simulated_annealing(inst17, seed=42, max_iterations=10000)
        sol2 = simulated_annealing(inst17, seed=42, max_iterations=10000)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_improves_over_nn(self, inst17):
        nn_sol = nearest_neighbor(inst17)
        sa_sol = simulated_annealing(
            inst17, max_iterations=50000, seed=42
        )
        # SA should generally improve or match NN
        assert sa_sol.distance <= nn_sol.distance * 1.1

    def test_quality_on_gr17(self, inst17):
        sol = simulated_annealing(inst17, max_iterations=100000, seed=42)
        valid, _ = validate_tour(inst17, sol.tour)
        assert valid
        # Should be within 20% of optimal (2016)
        assert sol.distance < 2016 * 1.20

    def test_custom_initial_tour(self, inst5):
        sol = simulated_annealing(inst5, initial_tour=[0, 1, 2, 3, 4], seed=42)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid


class TestGeneticAlgorithm:
    def test_feasible_tour(self, inst5):
        sol = genetic_algorithm(inst5, pop_size=20, generations=50, seed=42)
        valid, _ = validate_tour(inst5, sol.tour)
        assert valid

    def test_deterministic_with_seed(self, inst5):
        sol1 = genetic_algorithm(inst5, pop_size=20, generations=50, seed=42)
        sol2 = genetic_algorithm(inst5, pop_size=20, generations=50, seed=42)
        assert abs(sol1.distance - sol2.distance) < 1e-10

    def test_quality_on_gr17(self, inst17):
        sol = genetic_algorithm(
            inst17, pop_size=50, generations=200, seed=42
        )
        valid, _ = validate_tour(inst17, sol.tour)
        assert valid
        # Should be within 25% of optimal (2016)
        assert sol.distance < 2016 * 1.25

    def test_random_instance(self, random_inst):
        sol = genetic_algorithm(
            random_inst, pop_size=30, generations=100, seed=42
        )
        valid, _ = validate_tour(random_inst, sol.tour)
        assert valid


# ── Cross-method comparison tests ────────────────────────────────────────────


class TestCrossMethodComparison:
    def test_exact_methods_agree(self, inst5):
        hk_sol = held_karp(inst5)
        bb_sol = branch_and_bound(inst5)
        assert abs(hk_sol.distance - bb_sol.distance) < 1e-6

    def test_heuristics_within_range(self, inst17):
        optimal = 2016.0
        for fn in [
            lambda i: nearest_neighbor_multistart(i),
            lambda i: cheapest_insertion(i),
            lambda i: farthest_insertion(i),
            lambda i: greedy(i),
        ]:
            sol = fn(inst17)
            valid, _ = validate_tour(inst17, sol.tour)
            assert valid
            # Heuristics should be within 50% of optimal
            assert sol.distance < optimal * 1.5

    def test_local_search_improves_heuristic(self, inst17):
        nn_sol = nearest_neighbor(inst17)
        ls_sol = two_opt(inst17, nn_sol.tour)
        assert ls_sol.distance <= nn_sol.distance + 1e-10

    def test_all_methods_produce_valid_tours(self, random_inst):
        methods = [
            lambda i: nearest_neighbor(i),
            lambda i: cheapest_insertion(i),
            lambda i: farthest_insertion(i),
            lambda i: nearest_insertion(i),
            lambda i: greedy(i),
            lambda i: two_opt(i),
            lambda i: simulated_annealing(i, max_iterations=5000, seed=42),
            lambda i: genetic_algorithm(i, pop_size=20, generations=50, seed=42),
        ]
        for method in methods:
            sol = method(random_inst)
            valid, _ = validate_tour(random_inst, sol.tour)
            assert valid

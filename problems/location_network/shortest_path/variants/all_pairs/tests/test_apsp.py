"""Tests for All-Pairs Shortest Path."""

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


_inst = _load_mod("apsp_inst_test", os.path.join(_variant_dir, "instance.py"))
APSPInstance = _inst.APSPInstance
APSPSolution = _inst.APSPSolution
validate_solution = _inst.validate_solution
small_apsp_4 = _inst.small_apsp_4

_heur = _load_mod("apsp_heur_test", os.path.join(_variant_dir, "heuristics.py"))
floyd_warshall = _heur.floyd_warshall
repeated_dijkstra = _heur.repeated_dijkstra


class TestAPSPInstance:
    def test_random(self):
        inst = APSPInstance.random(n=5, seed=42)
        assert inst.n == 5
        assert inst.weight_matrix.shape == (5, 5)

    def test_small(self):
        inst = small_apsp_4()
        assert inst.n == 4

    def test_diagonal_zero(self):
        inst = small_apsp_4()
        for i in range(inst.n):
            assert inst.weight_matrix[i][i] == 0.0


class TestFloydWarshall:
    def test_valid(self):
        inst = small_apsp_4()
        sol = floyd_warshall(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_known_distances(self):
        inst = small_apsp_4()
        sol = floyd_warshall(inst)
        # 0->1: direct = 3
        assert abs(sol.dist_matrix[0][1] - 3) < 1e-6
        # 0->2: 0->1->2 = 3+2 = 5
        assert abs(sol.dist_matrix[0][2] - 5) < 1e-6
        # 0->3: 0->1->2->3 = 3+2+1 = 6
        assert abs(sol.dist_matrix[0][3] - 6) < 1e-6

    def test_path_reconstruction(self):
        inst = small_apsp_4()
        sol = floyd_warshall(inst)
        path = sol.get_path(0, 3)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 3

    def test_random(self):
        inst = APSPInstance.random(n=6, seed=42)
        sol = floyd_warshall(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"


class TestRepeatedDijkstra:
    def test_valid(self):
        inst = small_apsp_4()
        sol = repeated_dijkstra(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_agrees_with_floyd(self):
        inst = APSPInstance.random(n=5, seed=42)
        fw = floyd_warshall(inst)
        rd = repeated_dijkstra(inst)
        # Distances should match for non-negative weights
        for i in range(inst.n):
            for j in range(inst.n):
                if fw.dist_matrix[i][j] < np.inf:
                    assert abs(fw.dist_matrix[i][j] - rd.dist_matrix[i][j]) < 1e-6, \
                        f"Mismatch at ({i},{j}): FW={fw.dist_matrix[i][j]}, RD={rd.dist_matrix[i][j]}"

    def test_random(self):
        inst = APSPInstance.random(n=8, seed=42)
        sol = repeated_dijkstra(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

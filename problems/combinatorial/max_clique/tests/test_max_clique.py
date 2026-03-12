"""Tests for Maximum Clique Problem."""
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
_inst_mod = _load_mod("mc_instance_test", os.path.join(_base, "instance.py"))
_bk_mod = _load_mod(
    "mc_bk_test", os.path.join(_base, "exact", "bron_kerbosch.py")
)

MaxCliqueInstance = _inst_mod.MaxCliqueInstance
MaxCliqueSolution = _inst_mod.MaxCliqueSolution
bron_kerbosch = _bk_mod.bron_kerbosch
greedy_clique = _bk_mod.greedy_clique


class TestInstance:

    def test_complete(self):
        inst = MaxCliqueInstance.complete(5)
        assert inst.n_vertices == 5
        assert inst.n_edges == 10

    def test_is_clique(self):
        inst = MaxCliqueInstance.complete(4)
        assert inst.is_clique([0, 1, 2, 3])
        assert inst.is_clique([0, 1])

    def test_petersen(self):
        inst = MaxCliqueInstance.petersen()
        assert inst.n_vertices == 10

    def test_random(self):
        inst = MaxCliqueInstance.random(n_vertices=8, seed=42)
        assert inst.n_vertices == 8


class TestBronKerbosch:

    def test_complete_graph(self):
        inst = MaxCliqueInstance.complete(5)
        sol = bron_kerbosch(inst)
        assert sol.size == 5
        assert inst.is_clique(sol.clique)

    def test_petersen_clique(self):
        inst = MaxCliqueInstance.petersen()
        sol = bron_kerbosch(inst)
        assert sol.size == 2
        assert inst.is_clique(sol.clique)

    def test_empty_graph(self):
        inst = MaxCliqueInstance(n_vertices=5, edges=[])
        sol = bron_kerbosch(inst)
        assert sol.size == 1  # single vertex is a clique

    def test_random_valid(self):
        inst = MaxCliqueInstance.random(n_vertices=10, density=0.5, seed=7)
        sol = bron_kerbosch(inst)
        assert inst.is_clique(sol.clique)
        assert sol.size >= 1


class TestGreedyClique:

    def test_complete(self):
        inst = MaxCliqueInstance.complete(4)
        sol = greedy_clique(inst)
        assert inst.is_clique(sol.clique)
        assert sol.size == 4

    def test_greedy_vs_exact(self):
        inst = MaxCliqueInstance.random(n_vertices=10, density=0.5, seed=42)
        sol_g = greedy_clique(inst)
        sol_bk = bron_kerbosch(inst)
        assert inst.is_clique(sol_g.clique)
        assert sol_bk.size >= sol_g.size

    def test_repr(self):
        inst = MaxCliqueInstance.complete(3)
        sol = bron_kerbosch(inst)
        assert "size" in repr(sol)

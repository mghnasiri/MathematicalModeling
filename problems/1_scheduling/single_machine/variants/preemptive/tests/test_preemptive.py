"""Tests for Preemptive Single Machine Scheduling."""

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


_inst = _load_mod("preemptive_inst_test", os.path.join(_variant_dir, "instance.py"))
PreemptiveSMInstance = _inst.PreemptiveSMInstance
PreemptiveSMSolution = _inst.PreemptiveSMSolution
validate_solution = _inst.validate_solution
small_preemptive_4 = _inst.small_preemptive_4

_heur = _load_mod("preemptive_heur_test", os.path.join(_variant_dir, "heuristics.py"))
srpt = _heur.srpt
wsrpt = _heur.wsrpt


class TestPreemptiveInstance:
    def test_random(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        assert inst.n == 6

    def test_small(self):
        inst = small_preemptive_4()
        assert inst.n == 4


class TestSRPT:
    def test_valid(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        sol = srpt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_obj(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        sol = srpt(inst)
        assert sol.objective > 0

    def test_small(self):
        inst = small_preemptive_4()
        sol = srpt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_all_completed(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        sol = srpt(inst)
        for j in range(inst.n):
            assert sol.completion_times[j] > 0


class TestWSRPT:
    def test_valid(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        sol = wsrpt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

    def test_positive_obj(self):
        inst = PreemptiveSMInstance.random(n=6, seed=42)
        sol = wsrpt(inst)
        assert sol.objective > 0

    def test_small(self):
        inst = small_preemptive_4()
        sol = wsrpt(inst)
        valid, errors = validate_solution(inst, sol)
        assert valid, f"Errors: {errors}"

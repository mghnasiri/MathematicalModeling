"""Tests for Batch Scheduling Problem."""
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

_instance_mod = _load_mod("batch_inst", os.path.join(_base, "instance.py"))
_greedy_mod = _load_mod("batch_greedy", os.path.join(_base, "heuristics", "greedy_batch.py"))

BatchSchedulingInstance = _instance_mod.BatchSchedulingInstance
BatchSchedulingSolution = _instance_mod.BatchSchedulingSolution
greedy_batch = _greedy_mod.greedy_batch


class TestBatchInstance:
    """Tests for BatchSchedulingInstance."""

    def test_random_creation(self):
        inst = BatchSchedulingInstance.random(n_jobs=10, n_families=3)
        assert inst.n_jobs == 10
        assert inst.n_families == 3
        assert len(inst.processing_times) == 10

    def test_families_in_range(self):
        inst = BatchSchedulingInstance.random()
        assert np.all(inst.families >= 0)
        assert np.all(inst.families < inst.n_families)


class TestGreedyBatch:
    """Tests for greedy_batch heuristic."""

    def _simple_instance(self):
        """3 jobs, 2 families."""
        return BatchSchedulingInstance(
            n_jobs=3,
            processing_times=np.array([2.0, 3.0, 1.0]),
            weights=np.array([1.0, 1.0, 1.0]),
            families=np.array([0, 1, 0]),
            setup_time=2.0,
            n_families=2
        )

    def test_all_jobs_scheduled(self):
        inst = self._simple_instance()
        sol = greedy_batch(inst)
        assert set(sol.schedule) == set(range(inst.n_jobs))

    def test_family_grouping(self):
        """Jobs in same family should be consecutive in schedule."""
        inst = self._simple_instance()
        sol = greedy_batch(inst)
        for batch in sol.batches:
            families = [int(inst.families[j]) for j in batch]
            assert len(set(families)) == 1

    def test_completion_times_increasing(self):
        inst = self._simple_instance()
        sol = greedy_batch(inst)
        times_in_order = [sol.completion_times[j] for j in sol.schedule]
        for i in range(len(times_in_order) - 1):
            assert times_in_order[i] <= times_in_order[i + 1] + 1e-9

    def test_setup_time_included(self):
        inst = self._simple_instance()
        sol = greedy_batch(inst)
        n_batches = len(sol.batches)
        total_processing = float(np.sum(inst.processing_times))
        total_setup = n_batches * inst.setup_time
        last_completion = max(sol.completion_times)
        assert abs(last_completion - (total_processing + total_setup)) < 1e-9

    def test_single_family(self):
        inst = BatchSchedulingInstance(
            n_jobs=3,
            processing_times=np.array([2.0, 3.0, 1.0]),
            weights=np.array([1.0, 1.0, 1.0]),
            families=np.array([0, 0, 0]),
            setup_time=5.0,
            n_families=1
        )
        sol = greedy_batch(inst)
        assert len(sol.batches) == 1

    def test_weighted_completion_correct(self):
        inst = BatchSchedulingInstance(
            n_jobs=2,
            processing_times=np.array([3.0, 2.0]),
            weights=np.array([2.0, 1.0]),
            families=np.array([0, 0]),
            setup_time=1.0,
            n_families=1
        )
        sol = greedy_batch(inst)
        twc = float(np.sum(inst.weights * sol.completion_times))
        assert abs(twc - sol.total_weighted_completion) < 1e-9

    def test_single_job(self):
        inst = BatchSchedulingInstance(
            n_jobs=1,
            processing_times=np.array([5.0]),
            weights=np.array([1.0]),
            families=np.array([0]),
            setup_time=2.0,
            n_families=1
        )
        sol = greedy_batch(inst)
        assert sol.schedule == [0]
        assert abs(sol.completion_times[0] - 7.0) < 1e-9  # setup + processing

    def test_solution_repr(self):
        sol = BatchSchedulingSolution(
            schedule=[0], batches=[[0]],
            completion_times=np.array([5.0]),
            total_weighted_completion=5.0
        )
        assert "n_batches=1" in repr(sol)

    def test_random_instance(self):
        inst = BatchSchedulingInstance.random()
        sol = greedy_batch(inst)
        assert set(sol.schedule) == set(range(inst.n_jobs))
        assert sol.total_weighted_completion > 0

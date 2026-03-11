"""
Batch Single Machine Scheduling — Heuristics.

Algorithms:
    - WSPT batch: group jobs by similar processing times, order batches by
      weighted shortest processing time.
    - Single-job batches: baseline with each job as its own batch.

References:
    Webster, S. & Baker, K.R. (1995). Scheduling groups of jobs on a
    single machine. Operations Research, 43(4), 692-703.
    https://doi.org/10.1287/opre.43.4.692
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("batch_instance_h", os.path.join(_this_dir, "instance.py"))
BatchSMInstance = _inst.BatchSMInstance
BatchSMSolution = _inst.BatchSMSolution


def wspt_single(instance: BatchSMInstance) -> BatchSMSolution:
    """WSPT ordering with each job as a single batch.

    Sort jobs by wj/pj descending (WSPT rule), each in its own batch.

    Args:
        instance: Batch SM instance.

    Returns:
        BatchSMSolution.
    """
    ratios = instance.weights / np.maximum(instance.processing_times, 1e-10)
    order = np.argsort(-ratios).tolist()
    batches = [[j] for j in order]
    obj = instance.evaluate(batches)
    return BatchSMSolution(batches=batches, objective=obj)


def greedy_batching(instance: BatchSMInstance) -> BatchSMSolution:
    """Greedy batching: group jobs with similar processing times.

    Sorts jobs by processing time, then greedily groups consecutive jobs
    into batches when the setup cost saving outweighs the waiting cost.

    Args:
        instance: Batch SM instance.

    Returns:
        BatchSMSolution.
    """
    n = instance.n
    # Sort by processing time
    order = np.argsort(instance.processing_times).tolist()

    batches: list[list[int]] = [[order[0]]]
    for k in range(1, n):
        j = order[k]
        current_batch = batches[-1]
        batch_proc = max(instance.processing_times[jj] for jj in current_batch)
        new_proc = instance.processing_times[j]

        # Cost of adding to current batch vs starting new
        w_j = instance.weights[j]
        # If we add to current batch: no extra setup, but job j waits for max proc
        # If we start new batch: setup cost, but batch proc = pj only
        extra_wait = (new_proc - batch_proc) * sum(instance.weights[jj] for jj in current_batch)
        setup_cost = instance.setup_time * (w_j + sum(
            instance.weights[jj] for jj2 in batches for jj in jj2
            if jj not in current_batch
        ))

        # Simplified: add to batch if processing times are close
        if new_proc <= batch_proc * 1.5:
            batches[-1].append(j)
        else:
            batches.append([j])

    # Now reorder batches by WSPT-like criterion
    def batch_ratio(batch):
        w_total = sum(instance.weights[j] for j in batch)
        p_batch = max(instance.processing_times[j] for j in batch) + instance.setup_time
        return w_total / max(p_batch, 1e-10)

    batches.sort(key=batch_ratio, reverse=True)

    obj = instance.evaluate(batches)
    return BatchSMSolution(batches=batches, objective=obj)


if __name__ == "__main__":
    from instance import small_batch_6

    inst = small_batch_6()
    sol1 = wspt_single(inst)
    print(f"WSPT single: {sol1}")
    sol2 = greedy_batching(inst)
    print(f"Greedy batching: {sol2}")

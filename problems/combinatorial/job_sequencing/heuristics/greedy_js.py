"""
Greedy Heuristics for Job Sequencing with Deadlines.

Problem: 1 | d_j | Sigma w_j U_j
Complexity: O(n^2) for weighted greedy, O(n log n) for unit-weight

Algorithms:
1. greedy_profit: Sort by profit descending, add if feasible.
2. greedy_edd: Sort by earliest deadline, add if feasible.

References:
    Moore, J.M. (1968). An n job, one machine sequencing algorithm for
    minimizing the number of late jobs. Management Science, 15(1), 102-109.
    https://doi.org/10.1287/mnsc.15.1.102
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np


def _load_parent(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "js_instance_greedy",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
JobSequencingInstance = _inst.JobSequencingInstance
JobSequencingSolution = _inst.JobSequencingSolution


def greedy_profit(instance: JobSequencingInstance) -> JobSequencingSolution:
    """Greedy by profit: sort jobs by profit descending, add if feasible.

    Feasibility is checked by EDD ordering of selected jobs.

    Args:
        instance: A JobSequencingInstance.

    Returns:
        JobSequencingSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda j: instance.profits[j], reverse=True)

    selected = []
    for j in order:
        # Try adding j to selected set
        trial = selected + [j]
        # Sort by deadline (EDD) to find best feasible ordering
        trial.sort(key=lambda k: instance.deadlines[k])
        if instance.is_feasible(trial):
            selected = trial

    profit = sum(instance.profits[j] for j in selected)
    return JobSequencingSolution(
        sequence=selected, total_profit=profit, n_selected=len(selected)
    )


def greedy_edd(instance: JobSequencingInstance) -> JobSequencingSolution:
    """Greedy by EDD: sort by deadline, add if feasible.

    Args:
        instance: A JobSequencingInstance.

    Returns:
        JobSequencingSolution.
    """
    n = instance.n
    order = sorted(range(n), key=lambda j: instance.deadlines[j])

    selected = []
    time = 0.0
    for j in order:
        if time + instance.processing_times[j] <= instance.deadlines[j] + 1e-10:
            selected.append(j)
            time += instance.processing_times[j]

    profit = sum(instance.profits[j] for j in selected)
    return JobSequencingSolution(
        sequence=selected, total_profit=profit, n_selected=len(selected)
    )


if __name__ == "__main__":
    inst = JobSequencingInstance.unit_processing(5)
    sol_p = greedy_profit(inst)
    sol_e = greedy_edd(inst)
    print(f"Profit greedy: {sol_p}")
    print(f"EDD greedy: {sol_e}")

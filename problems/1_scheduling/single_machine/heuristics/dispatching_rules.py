"""
Dispatching Rules for Single Machine Scheduling

Optimal polynomial-time rules for tractable single machine objectives.

Rules implemented:
    1. SPT (Shortest Processing Time) — optimal for 1 || ΣCj, O(n log n)
    2. WSPT (Weighted SPT / Smith's Rule) — optimal for 1 || ΣwjCj, O(n log n)
    3. EDD (Earliest Due Date / Jackson's Rule) — optimal for 1 || Lmax, O(n log n)
    4. LPT (Longest Processing Time) — used for parallel machine heuristics

References:
    Smith, W.E. (1956). "Various Optimizers for Single-Stage Production"
    Naval Research Logistics Quarterly, 3(1-2):59-66.
    DOI: 10.1002/nav.3800030106

    Jackson, J.R. (1955). "Scheduling a Production Line to Minimize
    Maximum Tardiness" Management Science Research Project, Research
    Report 43, UCLA.

    Conway, R.W., Maxwell, W.L. & Miller, L.W. (1967). "Theory of Scheduling"
    Addison-Wesley.
"""

from __future__ import annotations
import sys
import os
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(os.path.dirname(_this_dir), "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_total_completion_time = _sm_instance.compute_total_completion_time
compute_weighted_completion_time = _sm_instance.compute_weighted_completion_time
compute_maximum_lateness = _sm_instance.compute_maximum_lateness


def spt(instance: SingleMachineInstance) -> SingleMachineSolution:
    """
    Shortest Processing Time rule — optimal for 1 || ΣCj.

    Sorts jobs in non-decreasing order of processing time. Minimizes
    total (unweighted) completion time. Ties broken by job index.

    Args:
        instance: A SingleMachineInstance.

    Returns:
        SingleMachineSolution with optimal ΣCj.

    Complexity: O(n log n)
    """
    sequence = sorted(range(instance.n), key=lambda j: (instance.processing_times[j], j))
    obj = compute_total_completion_time(instance, sequence)
    return SingleMachineSolution(sequence=sequence, objective_value=obj, objective_name="ΣCj")


def wspt(instance: SingleMachineInstance) -> SingleMachineSolution:
    """
    Weighted Shortest Processing Time rule (Smith's Rule) — optimal for 1 || ΣwjCj.

    Sorts jobs in non-decreasing order of pj/wj ratio (processing time
    divided by weight). Jobs with higher weight-to-processing-time ratio
    are scheduled first.

    Args:
        instance: A SingleMachineInstance (must have weights).

    Returns:
        SingleMachineSolution with optimal ΣwjCj.

    Complexity: O(n log n)
    """
    w = instance.weights
    if w is None:
        return spt(instance)

    p = instance.processing_times
    sequence = sorted(range(instance.n), key=lambda j: (p[j] / w[j], j))
    obj = compute_weighted_completion_time(instance, sequence)
    return SingleMachineSolution(sequence=sequence, objective_value=obj, objective_name="ΣwjCj")


def edd(instance: SingleMachineInstance) -> SingleMachineSolution:
    """
    Earliest Due Date rule (Jackson's Rule) — optimal for 1 || Lmax.

    Sorts jobs in non-decreasing order of due date. Minimizes maximum
    lateness when all jobs are available at time zero.

    Args:
        instance: A SingleMachineInstance (must have due dates).

    Returns:
        SingleMachineSolution with optimal Lmax.

    Complexity: O(n log n)
    """
    assert instance.due_dates is not None, "EDD requires due dates"

    sequence = sorted(range(instance.n), key=lambda j: (instance.due_dates[j], j))
    obj = compute_maximum_lateness(instance, sequence)
    return SingleMachineSolution(sequence=sequence, objective_value=obj, objective_name="Lmax")


def lpt(instance: SingleMachineInstance) -> SingleMachineSolution:
    """
    Longest Processing Time rule.

    Sorts jobs in non-increasing order of processing time. While not
    optimal for any single-machine objective, LPT is a key building
    block for parallel machine scheduling (Pm || Cmax).

    Args:
        instance: A SingleMachineInstance.

    Returns:
        SingleMachineSolution with the LPT sequence.

    Complexity: O(n log n)
    """
    sequence = sorted(
        range(instance.n),
        key=lambda j: (-instance.processing_times[j], j),
    )
    obj = compute_total_completion_time(instance, sequence)
    return SingleMachineSolution(sequence=sequence, objective_value=obj, objective_name="ΣCj (LPT)")


if __name__ == "__main__":
    import numpy as np

    print("=== Single Machine Dispatching Rules ===\n")

    inst = SingleMachineInstance.from_arrays(
        processing_times=[3, 5, 2, 7, 4],
        weights=[2, 1, 3, 1, 2],
        due_dates=[10, 12, 8, 20, 15],
    )
    print(f"Jobs: {inst.n}")
    print(f"p = {inst.processing_times}")
    print(f"w = {inst.weights}")
    print(f"d = {inst.due_dates}")

    sol_spt = spt(inst)
    print(f"\nSPT:  {sol_spt}")

    sol_wspt = wspt(inst)
    print(f"WSPT: {sol_wspt}")

    sol_edd = edd(inst)
    print(f"EDD:  {sol_edd}")

    sol_lpt = lpt(inst)
    print(f"LPT:  {sol_lpt}")

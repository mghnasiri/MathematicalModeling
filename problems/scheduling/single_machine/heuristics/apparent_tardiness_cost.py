"""
Apparent Tardiness Cost (ATC) Rule — 1 || ΣwjTj

A composite dispatching rule for the weighted tardiness objective.
At each scheduling decision point, the job with the highest ATC
priority index is selected. The index balances urgency (due date
proximity) with importance (weight-to-processing-time ratio).

The ATC priority for job j at time t is:
    I_j(t) = (w_j / p_j) * exp(-max(d_j - p_j - t, 0) / (K * p_avg))

where K is a lookahead scaling parameter and p_avg is the average
processing time of remaining jobs.

This is one of the best-known dispatching rules for weighted tardiness,
particularly effective when combined with local search.

Complexity: O(n²)

Reference: Vepsalainen, A.P.J. & Morton, T.E. (1987). "Priority Rules for
           Job Shops with Weighted Tardiness Costs"
           Management Science, 33(8):1035-1047.
           DOI: 10.1287/mnsc.33.8.1035
"""

from __future__ import annotations
import sys
import os
import math
import importlib.util
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(os.path.dirname(_this_dir), "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_weighted_tardiness = _sm_instance.compute_weighted_tardiness


def atc(
    instance: SingleMachineInstance,
    K: float = 2.0,
) -> SingleMachineSolution:
    """
    Apparent Tardiness Cost dispatching rule for 1 || ΣwjTj.

    At each step, selects the unscheduled job with the highest ATC index,
    which combines the WSPT ratio with an exponential urgency term.

    Args:
        instance: A SingleMachineInstance (must have weights and due dates).
        K: Lookahead scaling parameter. Higher K reduces the urgency
           effect, making ATC behave more like WSPT. Lower K increases
           sensitivity to due dates. Default: 2.0.

    Returns:
        SingleMachineSolution with ΣwjTj objective.

    Complexity: O(n²)
    """
    assert instance.due_dates is not None, "ATC requires due dates"

    n = instance.n
    p = instance.processing_times
    d = instance.due_dates
    w = instance.weights if instance.weights is not None else np.ones(n)

    unscheduled = set(range(n))
    sequence = []
    current_time = 0

    for _ in range(n):
        # Average processing time of remaining jobs
        p_avg = np.mean([p[j] for j in unscheduled])
        if p_avg == 0:
            p_avg = 1.0

        best_job = None
        best_priority = -1.0

        for j in unscheduled:
            slack = max(int(d[j]) - int(p[j]) - current_time, 0)
            urgency = math.exp(-slack / (K * p_avg))
            priority = (w[j] / p[j]) * urgency

            if priority > best_priority:
                best_priority = priority
                best_job = j

        sequence.append(best_job)
        current_time += p[best_job]
        unscheduled.remove(best_job)

    obj = compute_weighted_tardiness(instance, sequence)
    return SingleMachineSolution(
        sequence=sequence, objective_value=obj, objective_name="ΣwjTj"
    )


if __name__ == "__main__":
    print("=== ATC Rule Demo ===\n")

    inst = SingleMachineInstance.from_arrays(
        processing_times=[10, 6, 8, 4, 12],
        weights=[3, 1, 4, 2, 1],
        due_dates=[15, 20, 25, 10, 30],
    )
    print(f"p = {inst.processing_times}")
    print(f"w = {inst.weights}")
    print(f"d = {inst.due_dates}")

    for k_val in [0.5, 1.0, 2.0, 5.0]:
        sol = atc(inst, K=k_val)
        print(f"\nATC (K={k_val}): seq={sol.sequence}, ΣwjTj={sol.objective_value}")

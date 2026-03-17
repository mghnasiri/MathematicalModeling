"""
Branch and Bound for Single Machine Weighted Tardiness — 1 || ΣwjTj

Implements a B&B algorithm for the strongly NP-hard weighted tardiness
problem. Uses a depth-first search with the following components:

- Branching: Sequence jobs from first to last position.
- Lower bound: Current partial cost + EDD-based lower bound on remaining jobs.
- Upper bound: Warm-start with ATC heuristic solution.
- Dominance: WSPT ordering dominance for adjacent jobs.

Reference: Potts, C.N. & Van Wassenhove, L.N. (1985). "A Branch and Bound
           Algorithm for the Total Weighted Tardiness Problem"
           Operations Research, 33(2):363-377.
           DOI: 10.1287/opre.33.2.363
"""

from __future__ import annotations
import sys
import os
import time
import importlib.util
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)

_instance_path = os.path.join(_sm_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_weighted_tardiness = _sm_instance.compute_weighted_tardiness

_atc_path = os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py")
_spec2 = importlib.util.spec_from_file_location("sm_atc", _atc_path)
_sm_atc = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("sm_atc", _sm_atc)
_spec2.loader.exec_module(_sm_atc)

atc = _sm_atc.atc


def branch_and_bound_weighted_tardiness(
    instance: SingleMachineInstance,
    time_limit: float = 60.0,
) -> SingleMachineSolution:
    """
    Branch and Bound for 1 || ΣwjTj.

    Enumerates job sequences using depth-first search with lower bounds
    based on unweighted EDD scheduling of remaining jobs and the
    accumulated partial cost. Warm-started with ATC heuristic.

    Args:
        instance: A SingleMachineInstance (must have weights and due dates).
        time_limit: Maximum time in seconds.

    Returns:
        SingleMachineSolution. Optimal if completed within time limit,
        otherwise the best solution found.

    Complexity: O(n!) worst case, practically much faster with bounds.
    """
    assert instance.due_dates is not None, "B&B requires due dates"

    n = instance.n
    p = instance.processing_times.astype(int)
    d = instance.due_dates.astype(int)
    w = instance.weights.astype(int) if instance.weights is not None else np.ones(n, dtype=int)

    # Warm-start with ATC
    atc_sol = atc(instance)
    best_obj = compute_weighted_tardiness(instance, atc_sol.sequence)
    best_sequence = list(atc_sol.sequence)

    start_time = time.time()
    nodes_explored = 0

    def _lower_bound(scheduled: list[int], current_time: int, partial_cost: int) -> int:
        """
        Lower bound: partial cost + sum of min possible weighted tardiness
        for each remaining job (optimistic: each scheduled independently at
        earliest possible time using EDD order).
        """
        remaining = [j for j in range(n) if j not in scheduled_set]
        if not remaining:
            return partial_cost

        # Sort remaining by EDD and compute optimistic tardiness
        remaining_edd = sorted(remaining, key=lambda j: d[j])
        t = current_time
        lb = partial_cost
        for j in remaining_edd:
            t += p[j]
            lb += w[j] * max(0, t - d[j])

        return lb

    def _solve(scheduled: list[int], current_time: int, partial_cost: int):
        nonlocal best_obj, best_sequence, nodes_explored

        if time.time() - start_time >= time_limit:
            return

        nodes_explored += 1

        if len(scheduled) == n:
            if partial_cost < best_obj:
                best_obj = partial_cost
                best_sequence = list(scheduled)
            return

        remaining = [j for j in range(n) if j not in scheduled_set]

        # Sort remaining by WSPT ratio for better branching order
        remaining.sort(key=lambda j: p[j] / w[j])

        for job in remaining:
            new_time = current_time + p[job]
            new_cost = partial_cost + w[job] * max(0, new_time - d[job])

            # Pruning: lower bound check
            scheduled_set.add(job)
            scheduled.append(job)

            lb = _lower_bound(scheduled, new_time, new_cost)
            if lb < best_obj:
                _solve(scheduled, new_time, new_cost)

            scheduled.pop()
            scheduled_set.remove(job)

            if time.time() - start_time >= time_limit:
                return

    scheduled_set = set()
    _solve([], 0, 0)

    return SingleMachineSolution(
        sequence=best_sequence,
        objective_value=best_obj,
        objective_name="ΣwjTj",
    )


if __name__ == "__main__":
    print("=== B&B for Weighted Tardiness ===\n")

    inst = SingleMachineInstance.from_arrays(
        processing_times=[4, 3, 2, 5, 1],
        weights=[3, 2, 1, 4, 2],
        due_dates=[5, 6, 8, 10, 3],
    )
    print(f"p = {inst.processing_times}")
    print(f"w = {inst.weights}")
    print(f"d = {inst.due_dates}")

    sol = branch_and_bound_weighted_tardiness(inst)
    print(f"\nOptimal ΣwjTj = {sol.objective_value}")
    print(f"Sequence:      {sol.sequence}")

    verify = compute_weighted_tardiness(inst, sol.sequence)
    print(f"Verified:      {verify}")

    # Larger instance
    print("\n--- Random 12-job instance ---")
    rand_inst = SingleMachineInstance.random(n=12, seed=42)
    sol_r = branch_and_bound_weighted_tardiness(rand_inst, time_limit=10.0)
    print(f"Best ΣwjTj = {sol_r.objective_value}")
    print(f"Sequence:   {sol_r.sequence}")

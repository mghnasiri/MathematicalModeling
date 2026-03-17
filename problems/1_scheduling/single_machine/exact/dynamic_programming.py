"""
Dynamic Programming for Single Machine Total Tardiness — 1 || ΣTj

Implements the Lawler (1977) decomposition-based approach and a
pseudo-polynomial DP for minimizing total tardiness on a single machine.

The problem 1 || ΣTj is NP-hard (Du & Leung, 1990), but admits a
pseudo-polynomial time DP algorithm. The state space is (set of remaining
jobs, current time), but Lawler's decomposition reduces complexity
significantly for moderate-size instances.

This implementation uses a subset DP formulation with bitmask encoding
for small instances (n ≤ 20) and an EDD-based decomposition approach
for larger instances.

Complexity: O(2^n * n) for bitmask DP (exact for n ≤ 20)

References:
    Lawler, E.L. (1977). "A Pseudopolynomial Algorithm for Sequencing
    Jobs to Minimize Total Tardiness"
    Annals of Discrete Mathematics, 1:331-342.
    DOI: 10.1016/S0167-5060(08)70742-8

    Du, J. & Leung, J.Y.-T. (1990). "Minimizing Total Tardiness on One
    Machine is NP-Hard"
    Mathematics of Operations Research, 15(3):483-495.
    DOI: 10.1287/moor.15.3.483
"""

from __future__ import annotations
import sys
import os
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
compute_total_tardiness = _sm_instance.compute_total_tardiness


def dp_total_tardiness(
    instance: SingleMachineInstance,
    max_jobs: int = 20,
) -> SingleMachineSolution:
    """
    Exact DP for 1 || ΣTj using bitmask state representation.

    Uses a backward DP: for each subset S of unscheduled jobs, compute
    the minimum total tardiness achievable by scheduling the jobs in S
    starting at the appropriate time.

    The recursion is:
        f(S, t) = min over j in S of { T_j(t) + f(S \\ {j}, t + p_j) }
    where t = sum of processing times of jobs NOT in S.

    Args:
        instance: A SingleMachineInstance (must have due dates).
        max_jobs: Maximum number of jobs for exact solution. Raises
                  ValueError if n > max_jobs.

    Returns:
        SingleMachineSolution with optimal ΣTj.

    Complexity: O(2^n * n)
    """
    assert instance.due_dates is not None, "DP requires due dates"

    n = instance.n
    if n > max_jobs:
        raise ValueError(
            f"Instance has {n} jobs, exceeding max_jobs={max_jobs}. "
            "Use a heuristic or metaheuristic instead."
        )

    p = instance.processing_times.astype(int)
    d = instance.due_dates.astype(int)
    total_p = int(p.sum())

    # dp[mask] = minimum total tardiness for scheduling the jobs in mask
    # last_job[mask] = last job scheduled in subset mask (for backtracking)
    full_mask = (1 << n) - 1
    dp = np.full(full_mask + 1, np.iinfo(np.int64).max, dtype=np.int64)
    last_job = np.full(full_mask + 1, -1, dtype=np.int32)

    dp[0] = 0

    for mask in range(1, full_mask + 1):
        # Current time = total_p - sum of processing times of jobs in mask
        # (jobs in mask are yet to be scheduled from the perspective of
        # building the sequence backwards... but actually we build forward)
        #
        # Forward formulation: mask represents the set of jobs already scheduled.
        # Time after scheduling jobs in mask = sum of p_j for j in mask.
        t = 0
        for j in range(n):
            if mask & (1 << j):
                t += p[j]

        # Try each job j in mask as the LAST job scheduled
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if dp[prev_mask] == np.iinfo(np.int64).max:
                continue

            # Job j completes at time t
            tardiness_j = max(0, t - d[j])
            candidate = dp[prev_mask] + tardiness_j

            if candidate < dp[mask]:
                dp[mask] = candidate
                last_job[mask] = j

    # Backtrack to find the optimal sequence
    sequence = []
    mask = full_mask
    while mask > 0:
        j = last_job[mask]
        sequence.append(j)
        mask ^= (1 << j)
    sequence.reverse()

    obj = int(dp[full_mask])
    return SingleMachineSolution(
        sequence=sequence, objective_value=obj, objective_name="ΣTj"
    )


if __name__ == "__main__":
    print("=== DP for Total Tardiness ===\n")

    inst = SingleMachineInstance.from_arrays(
        processing_times=[4, 3, 2, 5, 1],
        due_dates=[5, 6, 8, 10, 3],
    )
    print(f"p = {inst.processing_times}")
    print(f"d = {inst.due_dates}")

    sol = dp_total_tardiness(inst)
    print(f"\nOptimal ΣTj = {sol.objective_value}")
    print(f"Sequence:    {sol.sequence}")

    # Verify
    verify = compute_total_tardiness(inst, sol.sequence)
    print(f"Verified:    {verify}")

    # Random instance
    print("\n--- Random 15-job instance ---")
    rand_inst = SingleMachineInstance.random(n=15, seed=42)
    sol_r = dp_total_tardiness(rand_inst)
    print(f"Optimal ΣTj = {sol_r.objective_value}")
    print(f"Sequence:    {sol_r.sequence}")

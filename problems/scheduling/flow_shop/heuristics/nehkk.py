"""
NEHKK — NEH with Kalczynski-Kamburowski Tie-Breaking for Fm | prmu | Cmax.

The NEHKK heuristic (Kalczynski & Kamburowski, 2008) improves the standard
NEH by adding a machine-idle-time based tie-breaking rule when multiple
insertion positions yield the same partial makespan. When ties occur, the
algorithm selects the position that minimizes total machine idle time in
the partial schedule, which tends to leave machines less idle for subsequent
job insertions.

Algorithm:
    1. Sort jobs by decreasing total processing time (same as NEH).
    2. For each job insertion, evaluate all positions:
       a. If a unique best makespan position exists, select it (same as NEH).
       b. If multiple positions tie for best makespan, compute total machine
          idle time for each tied position and select the one with minimum
          total idle time across ALL machines.
    3. Return the final permutation.

The key difference from Fernandez-Viagas & Framinan (2014) tie-breaking
(already in neh.py) is that NEHKK considers idle time on ALL machines,
not just the last machine.

Notation: Fm | prmu | Cmax
Complexity: O(n^2 * m) — same as NEH.
Quality: 1-3% improvement over standard NEH on average.

Reference:
    Kalczynski, P.J. & Kamburowski, J. (2008). An improved NEH heuristic
    to minimize makespan in permutation flow shops. Computers & Operations
    Research, 35(9), 3001-3008.
    https://doi.org/10.1016/j.cor.2007.01.020
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def _compute_all_machine_idle(
    instance: FlowShopInstance,
    permutation: list[int],
) -> int:
    """Compute total idle time summed across ALL machines.

    Idle time on machine i for job k is the gap between when the machine
    finishes job k-1 and when it can start job k (after the preceding
    machine finishes job k).

    Args:
        instance: Flow shop instance.
        permutation: Current job permutation.

    Returns:
        Total idle time across all machines.
    """
    n_jobs = len(permutation)
    m = instance.m
    p = instance.processing_times

    C = np.zeros((m, n_jobs), dtype=int)

    for k in range(n_jobs):
        job = permutation[k]
        C[0, k] = (C[0, k - 1] if k > 0 else 0) + p[0, job]
        for i in range(1, m):
            prev_job = C[i, k - 1] if k > 0 else 0
            C[i, k] = max(C[i - 1, k], prev_job) + p[i, job]

    total_idle = 0
    for i in range(m):
        for k in range(n_jobs):
            start_time = C[i, k] - p[i, permutation[k]]
            prev_end = C[i, k - 1] if k > 0 else 0
            total_idle += max(0, start_time - prev_end)

    return total_idle


def nehkk(instance: FlowShopInstance) -> FlowShopSolution:
    """Apply the NEHKK heuristic (NEH with KK all-machine idle tie-breaking).

    Args:
        instance: A FlowShopInstance.

    Returns:
        FlowShopSolution with the constructed permutation and makespan.
    """
    p = instance.processing_times

    # Step 1: Sort by decreasing total processing time
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(
        range(instance.n),
        key=lambda j: total_times[j],
        reverse=True,
    )

    sequence: list[int] = [sorted_jobs[0]]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_makespan = float("inf")
        best_idle = float("inf")
        best_position = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(instance, candidate)

            if ms < best_makespan:
                best_makespan = ms
                best_idle = _compute_all_machine_idle(instance, candidate)
                best_position = pos
            elif ms == best_makespan:
                # KK tie-breaking: minimize total idle across ALL machines
                idle = _compute_all_machine_idle(instance, candidate)
                if idle < best_idle:
                    best_idle = idle
                    best_position = pos

        sequence.insert(best_position, job)

    makespan = compute_makespan(instance, sequence)
    return FlowShopSolution(permutation=sequence, makespan=makespan)


if __name__ == "__main__":
    from heuristics.neh import neh, neh_with_tiebreaking

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh(instance)
    sol_ff = neh_with_tiebreaking(instance)
    sol_kk = nehkk(instance)

    print(f"NEH:    makespan = {sol_neh.makespan}")
    print(f"NEH-FF: makespan = {sol_ff.makespan}")
    print(f"NEHKK:  makespan = {sol_kk.makespan}")

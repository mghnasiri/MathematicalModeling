"""
Rajendran-Ziegler (RZ) Heuristic for Fm | prmu | Cmax.

The RZ heuristic (Rajendran & Ziegler, 1997) is a constructive heuristic
with an improvement pass. It differs from NEH in two key ways:
1. Initial ordering: ascending total processing time (shortest jobs first),
   rather than NEH's descending order.
2. After building the initial sequence via insertion, a second pass
   re-inserts each job in the sequence into its best position.

Algorithm:
    Phase 1 (Construction):
        1. Sort jobs by ascending total processing time.
        2. Start with the first job.
        3. For each subsequent job, try all insertion positions,
           keep the one with minimum partial makespan (same as NEH).

    Phase 2 (Improvement):
        4. For each job in the current sequence (in order):
           a. Remove the job from its position.
           b. Try reinserting it at every position.
           c. If a better makespan is found, keep the new position.
        5. Repeat until no improvement in a full pass.

Notation: Fm | prmu | Cmax
Complexity: O(n^2 * m) for construction, O(n^2 * m) per improvement pass.
Quality: Competitive with NEH; the improvement pass often finds better
         solutions, especially on larger instances.

Reference:
    Rajendran, C. & Ziegler, H. (1997). An efficient heuristic for
    scheduling in a flowshop to minimize total weighted flowtime of jobs.
    European Journal of Operational Research, 103(1), 129-138.
    https://doi.org/10.1016/S0377-2217(96)00273-1
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def rajendran_ziegler(instance: FlowShopInstance) -> FlowShopSolution:
    """Apply the Rajendran-Ziegler heuristic.

    Args:
        instance: A FlowShopInstance.

    Returns:
        FlowShopSolution with the constructed permutation and makespan.
    """
    p = instance.processing_times

    # Phase 1: Sort by ASCENDING total processing time (unlike NEH)
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(
        range(instance.n),
        key=lambda j: total_times[j],
    )

    # Construct sequence via best-insertion (same mechanism as NEH)
    sequence: list[int] = [sorted_jobs[0]]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_makespan = float("inf")
        best_position = 0

        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(instance, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_position = pos

        sequence.insert(best_position, job)

    # Phase 2: Improvement by re-insertion
    improved = True
    while improved:
        improved = False
        for idx in range(len(sequence)):
            job = sequence[idx]
            current_ms = compute_makespan(instance, sequence)

            # Remove job and try all positions
            remaining = sequence[:idx] + sequence[idx + 1:]
            best_ms = current_ms
            best_pos = idx

            for pos in range(len(remaining) + 1):
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan(instance, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
                    improved = True

            if best_ms < current_ms:
                sequence = remaining[:best_pos] + [job] + remaining[best_pos:]

    makespan = compute_makespan(instance, sequence)
    return FlowShopSolution(permutation=sequence, makespan=makespan)


if __name__ == "__main__":
    import numpy as np
    from heuristics.neh import neh

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh(instance)
    sol_rz = rajendran_ziegler(instance)

    print(f"NEH: makespan = {sol_neh.makespan}")
    print(f"RZ:  makespan = {sol_rz.makespan}")

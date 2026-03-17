"""
Beam Search Constructive Heuristic for Fm | prmu | Cmax.

Beam Search extends the NEH insertion paradigm by maintaining multiple
partial sequences (a "beam") in parallel. At each step, the next job is
inserted into every position of every beam candidate, and only the top-k
best partial solutions are retained for the next step.

Standard NEH is equivalent to Beam Search with beam width k=1. Increasing
the beam width trades computation time for solution quality, systematically
exploring more of the search space without the randomness of metaheuristics.

Algorithm:
    1. Sort jobs by decreasing total processing time (same as NEH).
    2. Initialize beam with the first job: beam = [{first_job}].
    3. For each subsequent job in sorted order:
       a. For each partial sequence in the beam, try inserting the job
          into every position.
       b. Evaluate all (beam_size * positions) candidates.
       c. Keep the top-k candidates as the new beam.
    4. Return the best solution in the final beam.

Notation: Fm | prmu | Cmax
Complexity: O(k * n^2 * m) where k = beam_width.
Quality: Improves over NEH monotonically with beam width. k=5 typically
         gives 1-3% improvement; k=20 approaches metaheuristic quality.

Reference:
    Libralesso, L., Focke, P.A., Secardin, A. & Jost, V. (2022).
    Iterative beam search algorithms for the permutation flowshop.
    European Journal of Operational Research, 301(1), 217-234.
    https://doi.org/10.1016/j.ejor.2021.10.015

    Fernandez-Viagas, V. & Framinan, J.M. (2017). A beam-search-based
    constructive heuristic for the PFSP to minimise total flowtime.
    Computers & Operations Research, 81, 167-177.
    https://doi.org/10.1016/j.cor.2016.12.020
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def beam_search(
    instance: FlowShopInstance,
    beam_width: int = 5,
) -> FlowShopSolution:
    """Apply Beam Search constructive heuristic for PFSP.

    Args:
        instance: A FlowShopInstance.
        beam_width: Number of partial solutions to retain at each step.
            beam_width=1 is equivalent to standard NEH.

    Returns:
        FlowShopSolution with the best permutation and makespan.
    """
    p = instance.processing_times

    # Sort by decreasing total processing time (same initial ordering as NEH)
    total_times = p.sum(axis=0)
    sorted_jobs = sorted(
        range(instance.n),
        key=lambda j: total_times[j],
        reverse=True,
    )

    # Initialize beam with the first job
    beam: list[tuple[list[int], int]] = [
        ([sorted_jobs[0]], compute_makespan(instance, [sorted_jobs[0]]))
    ]

    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        candidates: list[tuple[list[int], int]] = []

        for seq, _ in beam:
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [job] + seq[pos:]
                ms = compute_makespan(instance, new_seq)
                candidates.append((new_seq, ms))

        # Sort by makespan and keep only top beam_width
        candidates.sort(key=lambda x: x[1])

        # Deduplicate: keep unique permutations
        seen = set()
        new_beam = []
        for seq, ms in candidates:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                new_beam.append((seq, ms))
                if len(new_beam) >= beam_width:
                    break

        beam = new_beam

    # Return the best solution
    best_seq, best_ms = min(beam, key=lambda x: x[1])
    return FlowShopSolution(
        permutation=best_seq,
        makespan=best_ms,
    )


if __name__ == "__main__":
    from heuristics.neh import neh

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    sol_neh = neh(instance)
    print(f"NEH (beam=1): makespan = {sol_neh.makespan}")

    for bw in [3, 5, 10, 20]:
        sol_bs = beam_search(instance, beam_width=bw)
        improvement = (sol_neh.makespan - sol_bs.makespan) / sol_neh.makespan * 100
        print(f"Beam (k={bw:2d}):   makespan = {sol_bs.makespan} "
              f"({improvement:+.1f}%)")

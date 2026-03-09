"""
RA (Ruiz-Allahverdi) Constructive Heuristic for Fm | prmu | Cmax.

The RA heuristic (Rad, Ruiz & Boroojerdian, 2009) is an enhanced
constructive heuristic that extends the NEH insertion principle with
multiple insertion criteria. Instead of sorting jobs only by total
processing time, it uses multiple initial orderings (ascending,
descending, and weighted-sum variants) and selects the best result.

Algorithm:
    1. Generate k candidate initial orderings of jobs:
       a. Descending total processing time (like NEH)
       b. Ascending total processing time
       c. Descending weighted processing time (weighted by machine index)
       d. Ascending weighted processing time
    2. For each ordering, apply the NEH insertion procedure:
       a. Start with the first two jobs in the best of two orders.
       b. Insert each subsequent job at its best position.
    3. Return the best solution across all orderings.

Notation: Fm | prmu | Cmax
Complexity: O(k * n^2 * m) where k = 4 candidate orderings.
Quality: Slightly better than NEH on average due to multi-start approach.

Reference:
    Rad, S.F., Ruiz, R. & Boroojerdian, N. (2009). New high performing
    heuristics for minimizing makespan in permutation flowshops.
    OMEGA: The International Journal of Management Science, 37(2),
    331-345. https://doi.org/10.1016/j.omega.2007.02.002

    Framinan, J.M., Gupta, J.N.D. & Leisten, R. (2004). A review and
    classification of heuristics for permutation flow-shop scheduling
    with makespan objective. Journal of the Operational Research Society,
    55(12), 1243-1255.
    https://doi.org/10.1057/palgrave.jors.2601784
"""

from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance import FlowShopInstance, FlowShopSolution, compute_makespan


def _neh_insertion(
    instance: FlowShopInstance,
    job_order: list[int],
) -> tuple[list[int], int]:
    """Apply NEH insertion procedure with a given initial job ordering.

    Args:
        instance: Flow shop instance.
        job_order: Initial ordering of jobs to insert.

    Returns:
        Tuple of (best permutation, makespan).
    """
    perm = [job_order[0]]

    for k in range(1, len(job_order)):
        job = job_order[k]
        best_pos = 0
        best_ms = float("inf")

        for pos in range(len(perm) + 1):
            perm.insert(pos, job)
            ms = compute_makespan(instance, perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
            perm.pop(pos)

        perm.insert(best_pos, job)

    return perm, compute_makespan(instance, perm)


def ra_heuristic(instance: FlowShopInstance) -> FlowShopSolution:
    """Apply the RA multi-ordering constructive heuristic.

    Generates multiple initial job orderings and applies NEH insertion
    to each, returning the best result.

    Args:
        instance: A FlowShopInstance.

    Returns:
        FlowShopSolution with the best permutation and makespan.
    """
    p = instance.processing_times  # shape (m, n)
    n, m = instance.n, instance.m

    # Total processing time per job
    total_pt = np.sum(p, axis=0)  # shape (n,)

    # Weighted processing time (higher weight to later machines)
    weights = np.arange(1, m + 1, dtype=float)
    weighted_pt = np.dot(weights, p)  # shape (n,)

    # Generate candidate orderings
    orderings = []

    # 1. Descending total processing time (standard NEH order)
    orderings.append(list(np.argsort(-total_pt)))

    # 2. Ascending total processing time
    orderings.append(list(np.argsort(total_pt)))

    # 3. Descending weighted processing time
    orderings.append(list(np.argsort(-weighted_pt)))

    # 4. Ascending weighted processing time
    orderings.append(list(np.argsort(weighted_pt)))

    # Apply NEH insertion to each ordering
    best_perm = None
    best_ms = float("inf")

    for order in orderings:
        perm, ms = _neh_insertion(instance, order)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    return FlowShopSolution(
        permutation=best_perm,
        makespan=int(best_ms),
    )


if __name__ == "__main__":
    from heuristics.neh import neh as neh_original

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    neh_sol = neh_original(instance)
    ra_sol = ra_heuristic(instance)

    print(f"NEH:  makespan = {neh_sol.makespan}")
    print(f"RA:   makespan = {ra_sol.makespan}")
    print(f"Improvement: {(neh_sol.makespan - ra_sol.makespan) / neh_sol.makespan * 100:.1f}%")

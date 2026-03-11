"""
Quadratic Assignment Problem (QAP) — Constructive Heuristics.

Problem notation: QAP

Algorithms:
    - Greedy construction: assign facility-location pairs by highest
      flow × distance benefit, O(n^3).
    - 2-opt local search: pairwise swap improvement, O(n^2) per pass.

References:
    Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). Assignment
    Problems. SIAM. https://doi.org/10.1137/1.9780898717754
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


_inst = _load_mod("qap_instance_h", os.path.join(_this_dir, "instance.py"))
QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution


def greedy_construction(instance: QAPInstance) -> QAPSolution:
    """Greedy QAP construction based on flow-distance interaction.

    Assigns the highest-flow facility to the most central location,
    then greedily assigns remaining facilities to minimize incremental cost.

    Args:
        instance: QAP instance.

    Returns:
        QAPSolution with greedy assignment.
    """
    n = instance.n
    F = instance.flow_matrix
    D = instance.distance_matrix

    # Rank facilities by total flow (descending) and locations by total distance (ascending)
    facility_order = np.argsort(-F.sum(axis=1))
    location_order = np.argsort(D.sum(axis=1))

    perm = [-1] * n
    assigned_locs: set[int] = set()

    for fac in facility_order:
        best_loc = -1
        best_cost = float("inf")
        for loc in location_order:
            if loc in assigned_locs:
                continue
            # Compute incremental cost of assigning fac -> loc
            cost = 0.0
            for f2 in range(n):
                if perm[f2] < 0:
                    continue
                cost += F[fac][f2] * D[loc][perm[f2]]
                cost += F[f2][fac] * D[perm[f2]][loc]
            if best_loc < 0 or cost < best_cost:
                best_cost = cost
                best_loc = loc
        perm[fac] = best_loc
        assigned_locs.add(best_loc)

    total = instance.objective(perm)
    return QAPSolution(assignment=perm, cost=total)


def local_search_2opt(
    instance: QAPInstance,
    initial: QAPSolution | None = None,
) -> QAPSolution:
    """2-opt (pairwise swap) local search for QAP.

    Repeatedly swaps location assignments of two facilities
    if the swap reduces total cost, until no improving swap exists.

    Args:
        instance: QAP instance.
        initial: Starting solution (uses greedy if None).

    Returns:
        Locally optimal QAPSolution.
    """
    if initial is None:
        initial = greedy_construction(instance)

    n = instance.n
    perm = list(initial.assignment)
    cost = instance.objective(perm)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Compute swap delta
                delta = _swap_delta(instance, perm, i, j)
                if delta < -1e-10:
                    perm[i], perm[j] = perm[j], perm[i]
                    cost += delta
                    improved = True

    return QAPSolution(assignment=perm, cost=cost)


def _swap_delta(
    instance: QAPInstance, perm: list[int], i: int, j: int
) -> float:
    """Compute change in objective from swapping facilities i and j."""
    n = instance.n
    F = instance.flow_matrix
    D = instance.distance_matrix
    pi, pj = perm[i], perm[j]

    delta = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        pk = perm[k]
        delta += F[i][k] * (D[pj][pk] - D[pi][pk])
        delta += F[j][k] * (D[pi][pk] - D[pj][pk])
        delta += F[k][i] * (D[pk][pj] - D[pk][pi])
        delta += F[k][j] * (D[pk][pi] - D[pk][pj])

    # Diagonal terms for i<->j interaction
    delta += F[i][j] * (D[pj][pi] - D[pi][pj])
    delta += F[j][i] * (D[pi][pj] - D[pj][pi])

    return delta


if __name__ == "__main__":
    from instance import small_qap_4  # noqa: used in __main__ only

    inst = small_qap_4()
    gr = greedy_construction(inst)
    print(f"Greedy: {gr}")
    ls = local_search_2opt(inst)
    print(f"2-opt LS: {ls}")

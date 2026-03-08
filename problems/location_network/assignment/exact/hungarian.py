"""
Hungarian Algorithm (Kuhn-Munkres) — Optimal assignment in O(n^3).

Problem: Linear Assignment Problem (LAP)
Complexity: O(n^3)

The Hungarian method finds the minimum-cost one-to-one assignment
of agents to tasks. Based on the observation that adding/subtracting
a constant from any row or column doesn't change the optimal
assignment, and that an assignment of zero-cost entries is optimal
if feasible.

This implementation uses the Jonker-Volgenant formulation with
shortest augmenting paths for O(n^3) complexity.

References:
    Kuhn, H.W. (1955). The Hungarian method for the assignment
    problem. Naval Research Logistics Quarterly, 2(1-2), 83-97.
    https://doi.org/10.1002/nav.3800020109

    Munkres, J. (1957). Algorithms for the assignment and
    transportation problems. Journal of the Society for Industrial
    and Applied Mathematics, 5(1), 32-38.
    https://doi.org/10.1137/0105003

    Jonker, R. & Volgenant, A. (1987). A shortest augmenting path
    algorithm for dense and sparse linear assignment problems.
    Computing, 38(4), 325-340.
    https://doi.org/10.1007/BF02278710
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("ap_instance_hu", os.path.join(_parent_dir, "instance.py"))
AssignmentInstance = _inst.AssignmentInstance
AssignmentSolution = _inst.AssignmentSolution


def hungarian(instance: AssignmentInstance) -> AssignmentSolution:
    """Solve LAP using the Hungarian (Kuhn-Munkres) algorithm.

    Uses the shortest augmenting path variant for O(n^3) complexity.

    Args:
        instance: An AssignmentInstance.

    Returns:
        AssignmentSolution with optimal assignment.
    """
    n = instance.n
    cost = instance.cost_matrix.copy()

    INF = float("inf")

    # u[i], v[j]: dual variables (potentials)
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    # p[j]: row assigned to column j (1-indexed, 0 = unassigned)
    p = [0] * (n + 1)
    # way[j]: predecessor column in augmenting path
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0  # virtual column
        min_v = [INF] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < min_v[j]:
                        min_v[j] = cur
                        way[j] = j0
                    if min_v[j] < delta:
                        delta = min_v[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_v[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        # Augment along the path
        while j0 != 0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract assignment (convert from 1-indexed)
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1

    total_cost = instance.total_cost(assignment)

    return AssignmentSolution(
        assignment=assignment,
        cost=total_cost,
    )


if __name__ == "__main__":
    from instance import small_assignment_3, medium_assignment_5

    for fn in [small_assignment_3, medium_assignment_5]:
        inst = fn()
        sol = hungarian(inst)
        print(f"{inst.name}: {sol}")

"""
Generalized Assignment Problem (GAP) — Heuristics.

Algorithms:
    - Greedy by cost-resource ratio.
    - First-fit decreasing (by max resource consumption).

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. Wiley. ISBN 978-0471924203.

    Fisher, M.L., Jaikumar, R. & Van Wassenhove, L.N. (1986). A multiplier
    adjustment method for the generalized assignment problem. Management
    Science, 32(9), 1095-1103. https://doi.org/10.1287/mnsc.32.9.1095
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


_inst = _load_mod("gap_instance_h", os.path.join(_this_dir, "instance.py"))
GAPInstance = _inst.GAPInstance
GAPSolution = _inst.GAPSolution


def greedy_ratio(instance: GAPInstance) -> GAPSolution:
    """Greedy assignment by best cost/resource ratio per job.

    For each job, assign to the agent with lowest cost among those
    with remaining capacity.

    Args:
        instance: GAP instance.

    Returns:
        GAPSolution.
    """
    remaining = instance.capacity.copy()
    assignment = [-1] * instance.n

    # Sort jobs by max resource consumption (hardest first)
    job_order = sorted(range(instance.n),
                       key=lambda j: max(instance.resource[i][j]
                                         for i in range(instance.m)),
                       reverse=True)

    for j in job_order:
        best_agent = -1
        best_ratio = float("inf")
        for i in range(instance.m):
            if instance.resource[i][j] <= remaining[i] + 1e-9:
                ratio = instance.cost[i][j]
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_agent = i
        if best_agent == -1:
            # Fallback: assign to agent with most remaining capacity
            best_agent = int(np.argmax(remaining))
        assignment[j] = best_agent
        remaining[best_agent] -= instance.resource[best_agent][j]

    total = instance.total_cost(assignment)
    return GAPSolution(assignment=assignment, total_cost=total)


def first_fit_decreasing(instance: GAPInstance) -> GAPSolution:
    """First-fit decreasing heuristic for GAP.

    Sort jobs by decreasing average resource consumption,
    assign each to the first agent with capacity.

    Args:
        instance: GAP instance.

    Returns:
        GAPSolution.
    """
    remaining = instance.capacity.copy()
    assignment = [-1] * instance.n

    avg_resource = [float(np.mean(instance.resource[:, j]))
                    for j in range(instance.n)]
    job_order = sorted(range(instance.n), key=lambda j: avg_resource[j],
                       reverse=True)

    for j in job_order:
        assigned = False
        # Try agents sorted by cost for this job
        agent_order = sorted(range(instance.m),
                             key=lambda i: instance.cost[i][j])
        for i in agent_order:
            if instance.resource[i][j] <= remaining[i] + 1e-9:
                assignment[j] = i
                remaining[i] -= instance.resource[i][j]
                assigned = True
                break
        if not assigned:
            # Fallback: agent with most remaining capacity
            assignment[j] = int(np.argmax(remaining))
            remaining[assignment[j]] -= instance.resource[assignment[j]][j]

    total = instance.total_cost(assignment)
    return GAPSolution(assignment=assignment, total_cost=total)


if __name__ == "__main__":
    from instance import small_gap_6x3

    inst = small_gap_6x3()
    sol1 = greedy_ratio(inst)
    print(f"Greedy ratio: {sol1}")

    sol2 = first_fit_decreasing(inst)
    print(f"FFD: {sol2}")

"""
Greedy Heuristic for Linear Assignment Problem.

Problem: Linear Assignment Problem (LAP)
Complexity: O(n^2)

Assigns each agent to the cheapest available task greedily.
Not optimal but provides a fast upper bound.

References:
    Burkard, R.E. & Çela, E. (1999). Linear assignment problems
    and extensions. In: Handbook of Combinatorial Optimization,
    Springer, 75-149.
    https://doi.org/10.1007/978-1-4757-3023-4_2
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


_inst = _load_mod("ap_instance_gr", os.path.join(_parent_dir, "instance.py"))
AssignmentInstance = _inst.AssignmentInstance
AssignmentSolution = _inst.AssignmentSolution


def greedy_assignment(instance: AssignmentInstance) -> AssignmentSolution:
    """Solve LAP using greedy minimum-cost assignment.

    For each agent (in order of cheapest available cost), assign to
    the cheapest available task.

    Args:
        instance: An AssignmentInstance.

    Returns:
        AssignmentSolution.
    """
    n = instance.n
    cost = instance.cost_matrix

    # Collect all (cost, agent, task) triples, sort by cost
    entries = []
    for i in range(n):
        for j in range(n):
            entries.append((cost[i][j], i, j))
    entries.sort()

    assignment = [-1] * n
    used_tasks: set[int] = set()

    for _, agent, task in entries:
        if assignment[agent] == -1 and task not in used_tasks:
            assignment[agent] = task
            used_tasks.add(task)

    total = instance.total_cost(assignment)
    return AssignmentSolution(assignment=assignment, cost=total)


if __name__ == "__main__":
    from instance import small_assignment_3

    inst = small_assignment_3()
    sol = greedy_assignment(inst)
    print(f"Greedy: {sol}")

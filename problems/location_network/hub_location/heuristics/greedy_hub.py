"""Greedy heuristic for the p-Hub Median problem.

Greedy add: iteratively select the hub that most reduces total cost,
then assign each node to its nearest hub.

Complexity: O(p * n^3) — p rounds, each evaluating n candidate hubs with
O(n^2) assignment cost.

References:
    O'Kelly, M. E. (1987). A quadratic integer program for the location of
    interacting hub facilities. European Journal of Operational Research, 32(3),
    393-404.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "hub_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
HubLocationInstance = _inst.HubLocationInstance
HubLocationSolution = _inst.HubLocationSolution


def _assign_to_nearest(instance: HubLocationInstance,
                       hubs: list[int]) -> list[int]:
    """Assign each node to its nearest hub.

    Args:
        instance: Problem instance.
        hubs: List of open hub indices.

    Returns:
        Assignment list where assignments[i] is the hub for node i.
    """
    assignments = []
    for i in range(instance.n):
        best_hub = min(hubs, key=lambda h: instance.distances[i, h])
        assignments.append(best_hub)
    return assignments


def greedy_hub(instance: HubLocationInstance) -> HubLocationSolution:
    """Greedy add heuristic for p-hub median.

    Iteratively selects the hub node that yields the greatest cost reduction
    when added to the current hub set, then assigns all nodes to nearest hubs.

    Args:
        instance: A HubLocationInstance.

    Returns:
        A HubLocationSolution with the greedy hub selection and assignments.
    """
    hubs: list[int] = []
    candidates = set(range(instance.n))

    for _ in range(instance.p):
        best_hub = -1
        best_cost = float("inf")

        for c in candidates:
            trial_hubs = hubs + [c]
            trial_assign = _assign_to_nearest(instance, trial_hubs)
            cost = instance.transport_cost(trial_hubs, trial_assign)
            if cost < best_cost:
                best_cost = cost
                best_hub = c

        hubs.append(best_hub)
        candidates.discard(best_hub)

    assignments = _assign_to_nearest(instance, hubs)
    objective = instance.transport_cost(hubs, assignments)
    return HubLocationSolution(hubs=hubs, assignments=assignments,
                               objective=objective)


def enumeration_hub(instance: HubLocationInstance) -> HubLocationSolution:
    """Exhaustive enumeration for small p-hub median instances.

    Tries all C(n, p) hub combinations and returns the best.
    Only practical for small n and p.

    Args:
        instance: A HubLocationInstance.

    Returns:
        The optimal HubLocationSolution.
    """
    from itertools import combinations

    best_sol = None
    for hub_combo in combinations(range(instance.n), instance.p):
        hubs = list(hub_combo)
        assignments = _assign_to_nearest(instance, hubs)
        cost = instance.transport_cost(hubs, assignments)
        if best_sol is None or cost < best_sol.objective:
            best_sol = HubLocationSolution(hubs=hubs, assignments=assignments,
                                           objective=cost)
    return best_sol


if __name__ == "__main__":
    inst = HubLocationInstance.random(n=8, p=2, alpha=0.75, seed=42)
    print(f"Instance: {inst.n} nodes, {inst.p} hubs, alpha={inst.alpha}")

    sol_greedy = greedy_hub(inst)
    print(f"Greedy: {sol_greedy}")

    sol_enum = enumeration_hub(inst)
    print(f"Optimal: {sol_enum}")

    gap = (sol_greedy.objective - sol_enum.objective) / sol_enum.objective * 100
    print(f"Gap: {gap:.2f}%")

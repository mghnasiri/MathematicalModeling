"""
Greedy and Interchange Heuristics for the p-Median Problem.

Problem: p-Median Problem (PMP)
Complexity: O(m * p * n) for greedy, O(m * p * n * iter) for interchange

Greedy: Iteratively select the facility that reduces total cost the most.
Interchange (Teitz & Bart, 1968): Starting from a greedy solution, try
swapping each open facility with each closed one; accept if cost improves.

References:
    Teitz, M.B. & Bart, P. (1968). Heuristic methods for estimating
    the generalized vertex median of a weighted graph. Operations
    Research, 16(5), 955-961.
    https://doi.org/10.1287/opre.16.5.955

    Resende, M.G.C. & Werneck, R.F. (2004). A hybrid heuristic for
    the p-median problem. Journal of Heuristics, 10(1), 59-88.
    https://doi.org/10.1023/B:HEUR.0000019986.96257.50
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


_inst = _load_mod("pm_instance_gr", os.path.join(_parent_dir, "instance.py"))
PMedianInstance = _inst.PMedianInstance
PMedianSolution = _inst.PMedianSolution


def _assign_and_cost(
    instance: PMedianInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers to nearest open facility, return assignments and cost."""
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.distance_matrix[i][j])
        assignments.append(best)
        total += instance.weights[j] * instance.distance_matrix[best][j]
    return assignments, total


def greedy_pmedian(instance: PMedianInstance) -> PMedianSolution:
    """Solve p-Median using greedy facility selection.

    Iteratively add the facility that reduces total cost the most,
    until p facilities are open.

    Args:
        instance: A PMedianInstance.

    Returns:
        PMedianSolution.
    """
    m, n, p = instance.m, instance.n, instance.p
    open_set: set[int] = set()
    closed = set(range(m))

    # Select p facilities greedily
    for _ in range(p):
        best_fac = -1
        best_cost = float("inf")

        for candidate in closed:
            trial = open_set | {candidate}
            _, cost = _assign_and_cost(instance, trial)
            if cost < best_cost:
                best_cost = cost
                best_fac = candidate

        open_set.add(best_fac)
        closed.remove(best_fac)

    assignments, cost = _assign_and_cost(instance, open_set)
    return PMedianSolution(
        open_facilities=sorted(open_set),
        assignments=assignments,
        cost=cost,
    )


def interchange(
    instance: PMedianInstance,
    initial: PMedianSolution | None = None,
    max_iterations: int = 100,
) -> PMedianSolution:
    """Improve p-Median solution using Teitz-Bart interchange.

    For each open facility, try swapping with each closed facility.
    Accept the best improving swap. Repeat until no improvement.

    Args:
        instance: A PMedianInstance.
        initial: Starting solution. If None, uses greedy.
        max_iterations: Maximum iterations.

    Returns:
        PMedianSolution.
    """
    if initial is None:
        initial = greedy_pmedian(instance)

    m = instance.m
    open_set = set(initial.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

    for _ in range(max_iterations):
        improved = False
        closed = set(range(m)) - open_set

        best_swap = None
        best_cost = current_cost

        for out_fac in list(open_set):
            for in_fac in closed:
                trial = (open_set - {out_fac}) | {in_fac}
                _, cost = _assign_and_cost(instance, trial)
                if cost < best_cost - 1e-10:
                    best_cost = cost
                    best_swap = (out_fac, in_fac)

        if best_swap is not None:
            open_set.remove(best_swap[0])
            open_set.add(best_swap[1])
            current_cost = best_cost
            improved = True

        if not improved:
            break

    assignments, cost = _assign_and_cost(instance, open_set)
    return PMedianSolution(
        open_facilities=sorted(open_set),
        assignments=assignments,
        cost=cost,
    )


if __name__ == "__main__":
    from instance import small_pmedian_6_2

    inst = small_pmedian_6_2()
    gr = greedy_pmedian(inst)
    tb = interchange(inst)
    print(f"greedy: {gr}")
    print(f"interchange: {tb}")

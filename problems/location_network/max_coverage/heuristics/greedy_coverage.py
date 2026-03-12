"""Greedy heuristic for the Maximum Coverage Problem.

At each step, select the subset covering the most uncovered elements.
Achieves (1 - 1/e) approximation ratio for submodular maximization.

Complexity: O(k * m * n) — k rounds, each scanning m subsets of size up to n.

References:
    Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis
    of approximations for maximizing submodular set functions. Mathematical
    Programming, 14(1), 265-294.
"""
from __future__ import annotations

import sys
import os
import importlib.util


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "mc_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MaxCoverageInstance = _inst.MaxCoverageInstance
MaxCoverageSolution = _inst.MaxCoverageSolution


def greedy_coverage(instance: MaxCoverageInstance) -> MaxCoverageSolution:
    """Greedy algorithm for maximum coverage.

    Iteratively selects the subset that covers the most new (uncovered)
    elements until k subsets are selected or all elements are covered.

    Args:
        instance: A MaxCoverageInstance.

    Returns:
        A MaxCoverageSolution.
    """
    covered: set[int] = set()
    selected: list[int] = []
    available = set(range(instance.m))

    for _ in range(instance.k):
        if not available:
            break

        best_j = -1
        best_gain = -1

        for j in available:
            gain = len(instance.subsets[j] - covered)
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            break

        selected.append(best_j)
        covered |= instance.subsets[best_j]
        available.discard(best_j)

    return MaxCoverageSolution(selected=selected, covered=covered,
                               objective=len(covered))


def exhaustive_coverage(instance: MaxCoverageInstance) -> MaxCoverageSolution:
    """Exhaustive enumeration for small instances.

    Args:
        instance: A MaxCoverageInstance.

    Returns:
        The optimal MaxCoverageSolution.
    """
    from itertools import combinations

    best_sol = None
    for r in range(1, instance.k + 1):
        for combo in combinations(range(instance.m), r):
            sel = list(combo)
            cov = instance.coverage(sel)
            if best_sol is None or len(cov) > best_sol.objective:
                best_sol = MaxCoverageSolution(selected=sel, covered=cov,
                                               objective=len(cov))
    return best_sol


if __name__ == "__main__":
    inst = MaxCoverageInstance.random(n=15, m=8, k=3, density=0.3, seed=42)
    print(f"Instance: {inst.n} elements, {inst.m} subsets, budget={inst.k}")
    for j, s in enumerate(inst.subsets):
        print(f"  S{j}: {sorted(s)}")

    sol = greedy_coverage(inst)
    print(f"\nGreedy: {sol}")

    sol_opt = exhaustive_coverage(inst)
    print(f"Optimal: {sol_opt}")

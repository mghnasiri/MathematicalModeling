"""
Greedy Heuristics for the Set Covering Problem

The classic greedy: repeatedly select the subset with minimum cost per
newly covered element. Achieves H(m) = ln(m)+1 approximation ratio.

Complexity: O(m * n) per iteration, O(m * n * min(m,n)) total.

References:
    - Chvátal, V. (1979). A greedy heuristic for the set-covering problem.
      Math. Oper. Res., 4(3), 233-235.
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

_inst = _load_parent("scp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
SetCoveringInstance = _inst.SetCoveringInstance
SetCoveringSolution = _inst.SetCoveringSolution


def greedy_cost_effectiveness(instance: SetCoveringInstance) -> SetCoveringSolution:
    """Greedy: pick subset with minimum cost / new coverage ratio.

    Args:
        instance: SetCoveringInstance.

    Returns:
        SetCoveringSolution.
    """
    uncovered = set(range(instance.m))
    selected = []

    while uncovered:
        best_j = -1
        best_ratio = float("inf")

        for j in range(instance.n):
            if j in selected:
                continue
            new_coverage = len(instance.subsets[j] & uncovered)
            if new_coverage == 0:
                continue
            ratio = instance.costs[j] / new_coverage
            if ratio < best_ratio:
                best_ratio = ratio
                best_j = j

        if best_j == -1:
            break

        selected.append(best_j)
        uncovered -= instance.subsets[best_j]

    return SetCoveringSolution(
        selected=selected,
        total_cost=instance.total_cost(selected),
        n_selected=len(selected),
    )


def greedy_largest_first(instance: SetCoveringInstance) -> SetCoveringSolution:
    """Greedy: pick subset covering the most uncovered elements.

    Args:
        instance: SetCoveringInstance.

    Returns:
        SetCoveringSolution.
    """
    uncovered = set(range(instance.m))
    selected = []

    while uncovered:
        best_j = -1
        best_coverage = 0

        for j in range(instance.n):
            if j in selected:
                continue
            new_coverage = len(instance.subsets[j] & uncovered)
            if new_coverage > best_coverage:
                best_coverage = new_coverage
                best_j = j

        if best_j == -1:
            break

        selected.append(best_j)
        uncovered -= instance.subsets[best_j]

    return SetCoveringSolution(
        selected=selected,
        total_cost=instance.total_cost(selected),
        n_selected=len(selected),
    )

"""
Greedy Heuristic for Maximum Weight Set Packing.

Problem: Maximum Weight Set Packing
Complexity: O(m^2 * k) where k is max set size

Strategy: Sort sets by weight (descending, break ties by size ascending).
Greedily select the next compatible set (no element conflicts).

References:
    Hurkens, C.A.J. & Schrijver, A. (1989). On the size of systems of
    sets every t of which have an SDR. SIAM Journal on Discrete Mathematics.
    https://doi.org/10.1137/0402008
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


_inst = _load_mod("sp_instance_greedy", os.path.join(_parent_dir, "instance.py"))
SetPackingInstance = _inst.SetPackingInstance
SetPackingSolution = _inst.SetPackingSolution


def greedy_weight(instance: SetPackingInstance) -> SetPackingSolution:
    """Greedy set packing by decreasing weight.

    Selects sets in order of decreasing weight, skipping sets that
    conflict with already selected sets.

    Args:
        instance: A SetPackingInstance.

    Returns:
        SetPackingSolution with greedy selection.
    """
    # Sort by weight desc, then size asc
    order = sorted(
        range(instance.m),
        key=lambda i: (-instance.weights[i], len(instance.sets[i])),
    )

    selected = []
    used_elements: set[int] = set()

    for i in order:
        if not (instance.sets[i] & used_elements):
            selected.append(i)
            used_elements |= instance.sets[i]

    selected.sort()
    total_weight = instance.total_weight(selected)

    return SetPackingSolution(selected=selected, total_weight=total_weight)


def greedy_density(instance: SetPackingInstance) -> SetPackingSolution:
    """Greedy set packing by weight density (weight / set size).

    Args:
        instance: A SetPackingInstance.

    Returns:
        SetPackingSolution with density-based greedy selection.
    """
    densities = [
        instance.weights[i] / max(len(instance.sets[i]), 1)
        for i in range(instance.m)
    ]
    order = sorted(range(instance.m), key=lambda i: -densities[i])

    selected = []
    used_elements: set[int] = set()

    for i in order:
        if not (instance.sets[i] & used_elements):
            selected.append(i)
            used_elements |= instance.sets[i]

    selected.sort()
    total_weight = instance.total_weight(selected)

    return SetPackingSolution(selected=selected, total_weight=total_weight)


if __name__ == "__main__":
    inst = _inst.small_sp_3()
    sol = greedy_weight(inst)
    print(f"Greedy weight on {inst.name}: {sol}")
    sol2 = greedy_density(inst)
    print(f"Greedy density on {inst.name}: {sol2}")

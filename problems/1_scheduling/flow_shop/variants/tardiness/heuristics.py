"""
Tardiness Flow Shop — Heuristics.

Algorithms:
    - EDD ordering: Earliest Due Date first.
    - ATC-FS: Apparent Tardiness Cost adapted for flow shop.

References:
    Kim, Y.D. (1993). Heuristics for flowshop scheduling problems
    minimizing mean tardiness. Journal of the Operational Research
    Society, 44(1), 19-28. https://doi.org/10.1057/jors.1993.3
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


_inst = _load_mod("tfs_instance_h", os.path.join(_this_dir, "instance.py"))
TardinessFlowShopInstance = _inst.TardinessFlowShopInstance
TardinessFlowShopSolution = _inst.TardinessFlowShopSolution


def edd_rule(instance: TardinessFlowShopInstance) -> TardinessFlowShopSolution:
    """Earliest Due Date ordering.

    Args:
        instance: TardinessFlowShopInstance.

    Returns:
        TardinessFlowShopSolution.
    """
    perm = np.argsort(instance.due_dates).tolist()
    twt = instance.total_weighted_tardiness(perm)
    return TardinessFlowShopSolution(permutation=perm, total_weighted_tardiness=twt)


def wspt_rule(instance: TardinessFlowShopInstance) -> TardinessFlowShopSolution:
    """Weighted Shortest Processing Time ordering.

    Args:
        instance: TardinessFlowShopInstance.

    Returns:
        TardinessFlowShopSolution.
    """
    total_proc = instance.processing_times.sum(axis=1)
    ratio = instance.weights / np.maximum(total_proc, 1e-10)
    perm = np.argsort(-ratio).tolist()
    twt = instance.total_weighted_tardiness(perm)
    return TardinessFlowShopSolution(permutation=perm, total_weighted_tardiness=twt)


def neh_tardiness(instance: TardinessFlowShopInstance) -> TardinessFlowShopSolution:
    """NEH adapted for total weighted tardiness.

    Args:
        instance: TardinessFlowShopInstance.

    Returns:
        TardinessFlowShopSolution.
    """
    n = instance.n
    total_proc = instance.processing_times.sum(axis=1)
    order = np.argsort(-total_proc).tolist()

    perm = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_twt = float("inf")
        best_pos = 0
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            twt = instance.total_weighted_tardiness(candidate)
            if twt < best_twt:
                best_twt = twt
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]

    twt = instance.total_weighted_tardiness(perm)
    return TardinessFlowShopSolution(permutation=perm, total_weighted_tardiness=twt)


if __name__ == "__main__":
    from instance import small_tfs_4x3

    inst = small_tfs_4x3()
    for name, algo in [("EDD", edd_rule), ("WSPT", wspt_rule), ("NEH-T", neh_tardiness)]:
        sol = algo(inst)
        print(f"{name}: {sol}")

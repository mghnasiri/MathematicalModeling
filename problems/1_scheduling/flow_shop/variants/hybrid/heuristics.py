"""
Hybrid Flow Shop (HFS) — Constructive Heuristics.

Problem notation: HFm | prmu | Cmax

Algorithms:
    - NEH-HFS: Adapted NEH for hybrid flow shop.
    - LPT-HFS: Longest Processing Time ordering.

References:
    Nawaz, M., Enscore, E. & Ham, I. (1983). A heuristic algorithm for
    the m-machine, n-job flow-shop sequencing problem. Omega, 11(1),
    91-95. https://doi.org/10.1016/0305-0483(83)90088-9

    Ruiz, R. & Maroto, C. (2006). A genetic algorithm for hybrid
    flowshops with sequence dependent setup times and machine
    eligibility. European Journal of Operational Research, 169(3),
    781-800. https://doi.org/10.1016/j.ejor.2004.06.038
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


_inst = _load_mod("hfs_instance_h", os.path.join(_this_dir, "instance.py"))
HFSInstance = _inst.HFSInstance
HFSSolution = _inst.HFSSolution


def neh_hfs(instance: HFSInstance) -> HFSSolution:
    """NEH adapted for Hybrid Flow Shop.

    Sort jobs by decreasing total processing time, then insert each job
    at the position yielding the best makespan.

    Args:
        instance: HFS instance.

    Returns:
        HFSSolution.
    """
    n = instance.n
    total_proc = instance.processing_times.sum(axis=1)
    order = np.argsort(-total_proc).tolist()

    perm = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_ms = float("inf")
        best_pos = 0
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = instance.makespan(candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]

    ms = instance.makespan(perm)
    return HFSSolution(permutation=perm, makespan=ms)


def lpt_hfs(instance: HFSInstance) -> HFSSolution:
    """Longest Processing Time ordering for HFS.

    Args:
        instance: HFS instance.

    Returns:
        HFSSolution.
    """
    total_proc = instance.processing_times.sum(axis=1)
    perm = np.argsort(-total_proc).tolist()
    ms = instance.makespan(perm)
    return HFSSolution(permutation=perm, makespan=ms)


def spt_hfs(instance: HFSInstance) -> HFSSolution:
    """Shortest Processing Time ordering for HFS.

    Args:
        instance: HFS instance.

    Returns:
        HFSSolution.
    """
    total_proc = instance.processing_times.sum(axis=1)
    perm = np.argsort(total_proc).tolist()
    ms = instance.makespan(perm)
    return HFSSolution(permutation=perm, makespan=ms)


if __name__ == "__main__":
    from instance import small_hfs_4x3

    inst = small_hfs_4x3()
    sol = neh_hfs(inst)
    print(f"NEH-HFS: {sol}")
    sol2 = lpt_hfs(inst)
    print(f"LPT-HFS: {sol2}")

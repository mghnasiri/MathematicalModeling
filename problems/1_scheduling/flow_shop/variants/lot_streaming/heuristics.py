"""
Lot Streaming Flow Shop — Heuristics.

Algorithms:
    - NEH adapted for lot streaming objective.
    - LPT ordering.

References:
    Trietsch, D. & Baker, K.R. (1993). Basic techniques for lot streaming.
    Operations Research, 41(6), 1065-1076.
    https://doi.org/10.1287/opre.41.6.1065
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


_inst = _load_mod("lotstream_instance_h", os.path.join(_this_dir, "instance.py"))
LotStreamInstance = _inst.LotStreamInstance
LotStreamSolution = _inst.LotStreamSolution


def neh_ls(instance: LotStreamInstance) -> LotStreamSolution:
    """NEH heuristic using lot-streaming makespan evaluation.

    Args:
        instance: LotStreamInstance.

    Returns:
        LotStreamSolution.
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
            ms = instance.makespan_streaming(candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]

    ms = instance.makespan_streaming(perm)
    ms_ns = instance.makespan_no_streaming(perm)
    return LotStreamSolution(permutation=perm, makespan=ms, makespan_no_stream=ms_ns)


def lpt_ls(instance: LotStreamInstance) -> LotStreamSolution:
    """LPT ordering with lot streaming evaluation.

    Args:
        instance: LotStreamInstance.

    Returns:
        LotStreamSolution.
    """
    total_proc = instance.processing_times.sum(axis=1)
    perm = np.argsort(-total_proc).tolist()
    ms = instance.makespan_streaming(perm)
    ms_ns = instance.makespan_no_streaming(perm)
    return LotStreamSolution(permutation=perm, makespan=ms, makespan_no_stream=ms_ns)


if __name__ == "__main__":
    from instance import small_ls_4x3

    inst = small_ls_4x3()
    sol1 = neh_ls(inst)
    print(f"NEH-LS: {sol1}")
    sol2 = lpt_ls(inst)
    print(f"LPT-LS: {sol2}")

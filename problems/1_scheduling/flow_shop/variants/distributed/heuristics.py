"""
Distributed Permutation Flow Shop — Heuristics.

Algorithms:
    - NEH-DPFSP: Assign jobs to factory with minimum makespan increase.
    - Round-robin NEH: Distribute jobs round-robin by NEH order.

References:
    Naderi, B. & Ruiz, R. (2010). The distributed permutation flowshop
    scheduling problem. Computers & Operations Research, 37(4), 754-768.
    https://doi.org/10.1016/j.cor.2009.06.019
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


_inst = _load_mod("dpfsp_instance_h", os.path.join(_this_dir, "instance.py"))
DPFSPInstance = _inst.DPFSPInstance
DPFSPSolution = _inst.DPFSPSolution


def neh_dpfsp(instance: DPFSPInstance) -> DPFSPSolution:
    """NEH-based heuristic for DPFSP.

    Sort jobs by total processing time (descending). For each job, try
    inserting at every position in every factory; pick the one giving
    the minimum overall makespan.

    Args:
        instance: DPFSP instance.

    Returns:
        DPFSPSolution.
    """
    n = instance.n
    total_proc = instance.processing_times.sum(axis=1)
    order = np.argsort(-total_proc).tolist()

    assignment: list[list[int]] = [[] for _ in range(instance.f)]

    for job in order:
        best_ms = float("inf")
        best_factory = 0
        best_pos = 0

        for fac in range(instance.f):
            for pos in range(len(assignment[fac]) + 1):
                candidate = assignment[fac][:pos] + [job] + assignment[fac][pos:]
                ms = instance.factory_makespan(candidate)
                overall = max(ms, max(
                    (instance.factory_makespan(assignment[g])
                     for g in range(instance.f) if g != fac),
                    default=0.0
                ))
                if overall < best_ms:
                    best_ms = overall
                    best_factory = fac
                    best_pos = pos

        assignment[best_factory].insert(best_pos, job)

    ms = instance.makespan(assignment)
    return DPFSPSolution(assignment=assignment, makespan=ms)


def round_robin(instance: DPFSPInstance) -> DPFSPSolution:
    """Round-robin assignment by total processing time.

    Sort by descending total processing time, assign to factories
    round-robin.

    Args:
        instance: DPFSP instance.

    Returns:
        DPFSPSolution.
    """
    total_proc = instance.processing_times.sum(axis=1)
    order = np.argsort(-total_proc).tolist()

    assignment: list[list[int]] = [[] for _ in range(instance.f)]
    for i, job in enumerate(order):
        assignment[i % instance.f].append(job)

    ms = instance.makespan(assignment)
    return DPFSPSolution(assignment=assignment, makespan=ms)


if __name__ == "__main__":
    from instance import small_dpfsp_6x3x2

    inst = small_dpfsp_6x3x2()
    sol1 = neh_dpfsp(inst)
    print(f"NEH-DPFSP: {sol1}")
    sol2 = round_robin(inst)
    print(f"Round-robin: {sol2}")

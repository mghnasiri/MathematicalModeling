"""
Distributed Permutation Flow Shop — Metaheuristics.

Algorithms:
    - Simulated Annealing with inter/intra-factory moves.

References:
    Naderi, B. & Ruiz, R. (2010). The distributed permutation flowshop
    scheduling problem. Computers & Operations Research, 37(4), 754-768.
    https://doi.org/10.1016/j.cor.2009.06.019
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("dpfsp_instance_m", os.path.join(_this_dir, "instance.py"))
DPFSPInstance = _inst.DPFSPInstance
DPFSPSolution = _inst.DPFSPSolution

_heur = _load_mod("dpfsp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
neh_dpfsp = _heur.neh_dpfsp


def simulated_annealing(
    instance: DPFSPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> DPFSPSolution:
    """Simulated Annealing for DPFSP.

    Moves:
    - Intra-factory swap/insertion
    - Inter-factory job transfer

    Args:
        instance: DPFSP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best DPFSPSolution found.
    """
    rng = np.random.default_rng(seed)
    f = instance.f

    init = neh_dpfsp(instance)
    assignment = [list(a) for a in init.assignment]
    cost = init.makespan

    best_assignment = [list(a) for a in assignment]
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_assign = [list(a) for a in assignment]
        move = rng.integers(0, 3)

        if move == 0:
            # Intra-factory swap
            fac = rng.integers(0, f)
            if len(new_assign[fac]) >= 2:
                i, j = rng.choice(len(new_assign[fac]), 2, replace=False)
                new_assign[fac][i], new_assign[fac][j] = (
                    new_assign[fac][j], new_assign[fac][i]
                )

        elif move == 1:
            # Intra-factory insertion
            fac = rng.integers(0, f)
            if len(new_assign[fac]) >= 2:
                i = rng.integers(0, len(new_assign[fac]))
                job = new_assign[fac].pop(i)
                j = rng.integers(0, len(new_assign[fac]) + 1)
                new_assign[fac].insert(j, job)

        elif move == 2 and f > 1:
            # Inter-factory transfer
            non_empty = [g for g in range(f) if new_assign[g]]
            if non_empty:
                src = rng.choice(non_empty)
                dst = rng.integers(0, f)
                while dst == src and f > 1:
                    dst = rng.integers(0, f)
                idx = rng.integers(0, len(new_assign[src]))
                job = new_assign[src].pop(idx)
                pos = rng.integers(0, len(new_assign[dst]) + 1)
                new_assign[dst].insert(pos, job)

        new_cost = instance.makespan(new_assign)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            assignment = new_assign
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_assignment = [list(a) for a in assignment]

        temp *= cooling_rate

    return DPFSPSolution(assignment=best_assignment, makespan=best_cost)


if __name__ == "__main__":
    from instance import small_dpfsp_6x3x2

    inst = small_dpfsp_6x3x2()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

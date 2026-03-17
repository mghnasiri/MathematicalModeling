"""
Unrelated Parallel Machine with Tardiness — Metaheuristics.

Algorithms:
    - Simulated Annealing with reassign and swap moves.

References:
    Weng, M.X., Lu, J. & Ren, H. (2001). Unrelated parallel machine
    scheduling with setup consideration and a total weighted completion
    time objective. International Journal of Production Economics, 70(3),
    215-226. https://doi.org/10.1016/S0925-5273(00)00066-9
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


_inst = _load_mod("rm_tard_instance_m", os.path.join(_this_dir, "instance.py"))
RmTardinessInstance = _inst.RmTardinessInstance
RmTardinessSolution = _inst.RmTardinessSolution

_heur = _load_mod("rm_tard_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
edd_ect = _heur.edd_ect


def simulated_annealing(
    instance: RmTardinessInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> RmTardinessSolution:
    """SA for Rm||ΣTj.

    Moves: reassign job to different machine, swap jobs between machines,
    swap adjacent jobs on same machine.

    Args:
        instance: RmTardiness instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        RmTardinessSolution.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m

    init = edd_ect(instance)
    assignment = list(init.assignment)
    sequence = [list(s) for s in init.sequence]
    cost = init.total_tardiness

    best_assignment = list(assignment)
    best_sequence = [list(s) for s in sequence]
    best_cost = cost

    temp = max(1.0, best_cost * 0.2) if best_cost > 0 else 5.0
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_assign = list(assignment)
        new_seq = [list(s) for s in sequence]
        move = rng.integers(0, 3)

        if move == 0:
            # Reassign job to different machine
            j = int(rng.integers(0, n))
            old_m = new_assign[j]
            new_m = int(rng.integers(0, m - 1))
            if new_m >= old_m:
                new_m += 1
            new_seq[old_m].remove(j)
            pos = int(rng.integers(0, len(new_seq[new_m]) + 1))
            new_seq[new_m].insert(pos, j)
            new_assign[j] = new_m

        elif move == 1:
            # Swap two jobs between different machines
            if all(len(s) > 0 for s in new_seq):
                i1 = int(rng.integers(0, m))
                i2 = int(rng.integers(0, m - 1))
                if i2 >= i1:
                    i2 += 1
                if new_seq[i1] and new_seq[i2]:
                    idx1 = int(rng.integers(0, len(new_seq[i1])))
                    idx2 = int(rng.integers(0, len(new_seq[i2])))
                    j1 = new_seq[i1][idx1]
                    j2 = new_seq[i2][idx2]
                    new_seq[i1][idx1] = j2
                    new_seq[i2][idx2] = j1
                    new_assign[j1] = i2
                    new_assign[j2] = i1

        elif move == 2:
            # Swap adjacent jobs on same machine
            i = int(rng.integers(0, m))
            if len(new_seq[i]) >= 2:
                idx = int(rng.integers(0, len(new_seq[i]) - 1))
                new_seq[i][idx], new_seq[i][idx + 1] = \
                    new_seq[i][idx + 1], new_seq[i][idx]

        new_cost = instance.total_tardiness(new_assign, new_seq)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            assignment = new_assign
            sequence = new_seq
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_assignment = list(assignment)
                best_sequence = [list(s) for s in sequence]

        temp *= cooling_rate

    return RmTardinessSolution(assignment=best_assignment,
                                sequence=best_sequence,
                                total_tardiness=best_cost)


if __name__ == "__main__":
    from instance import small_rm_tard_6x2

    inst = small_rm_tard_6x2()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

"""
Batch Single Machine Scheduling — Metaheuristics.

Algorithms:
    - Simulated Annealing with move/merge/split batch neighborhood.

References:
    Potts, C.N. & Kovalyov, M.Y. (2000). Scheduling with batching:
    a review. European Journal of Operational Research, 120(2), 228-249.
    https://doi.org/10.1016/S0377-2217(99)00153-8
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


_inst = _load_mod("batch_instance_m", os.path.join(_this_dir, "instance.py"))
BatchSMInstance = _inst.BatchSMInstance
BatchSMSolution = _inst.BatchSMSolution

_heur = _load_mod("batch_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
wspt_single = _heur.wspt_single


def simulated_annealing(
    instance: BatchSMInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> BatchSMSolution:
    """Simulated Annealing for Batch SM.

    Neighborhood moves:
    - Move: transfer a job from one batch to another.
    - Merge: combine two batches.
    - Split: move a random subset of a batch to a new batch.
    - Swap batches: reorder two batches.

    Args:
        instance: Batch SM instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best BatchSMSolution found.
    """
    rng = np.random.default_rng(seed)

    init = wspt_single(instance)
    batches = [list(b) for b in init.batches]
    cost = init.objective

    best_batches = [list(b) for b in batches]
    best_cost = cost

    temp = best_cost * 0.05
    start_time = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start_time > time_limit:
            break

        new_batches = [list(b) for b in batches]
        move = rng.integers(0, 4)

        if move == 0 and len(new_batches) > 1:
            # Move a job from one batch to another
            r1 = rng.integers(0, len(new_batches))
            if len(new_batches[r1]) > 1:
                idx = rng.integers(0, len(new_batches[r1]))
                job = new_batches[r1].pop(idx)
                r2 = rng.integers(0, len(new_batches))
                new_batches[r2].append(job)

        elif move == 1 and len(new_batches) > 1:
            # Merge two batches
            r1, r2 = rng.choice(len(new_batches), 2, replace=False)
            r1, r2 = min(r1, r2), max(r1, r2)
            new_batches[r1].extend(new_batches[r2])
            new_batches.pop(r2)

        elif move == 2:
            # Split a batch
            r = rng.integers(0, len(new_batches))
            if len(new_batches[r]) >= 2:
                k = rng.integers(1, len(new_batches[r]))
                rng.shuffle(new_batches[r])
                new_batch = new_batches[r][k:]
                new_batches[r] = new_batches[r][:k]
                new_batches.append(new_batch)

        elif move == 3 and len(new_batches) >= 2:
            # Swap two batch positions
            i, j = rng.choice(len(new_batches), 2, replace=False)
            new_batches[i], new_batches[j] = new_batches[j], new_batches[i]

        # Remove empty batches
        new_batches = [b for b in new_batches if b]
        if not new_batches:
            continue

        new_cost = instance.evaluate(new_batches)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            batches = new_batches
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_batches = [list(b) for b in batches]

        temp *= cooling_rate

    return BatchSMSolution(batches=best_batches, objective=best_cost)


if __name__ == "__main__":
    from instance import small_batch_6

    inst = small_batch_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

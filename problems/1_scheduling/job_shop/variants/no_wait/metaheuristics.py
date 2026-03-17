"""
Simulated Annealing for No-Wait Job Shop.

Problem: Jm | no-wait | Cmax

Permutation-based: job ordering determines start times using greedy
insertion. SA explores permutation neighborhood (swap, insert moves).

Warm-started with greedy insertion heuristic.

Complexity: O(iterations * n^2 * m) per run.

References:
    Mascis, A. & Pacciarelli, D. (2002). Job-shop scheduling with
    blocking and no-wait constraints. EJOR, 143(3), 498-517.
    https://doi.org/10.1016/S0377-2217(01)00338-1
"""

from __future__ import annotations

import sys
import os
import math
import time
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


_inst = _load_mod("nwjsp_instance_meta", os.path.join(_this_dir, "instance.py"))
NWJSPInstance = _inst.NWJSPInstance
NWJSPSolution = _inst.NWJSPSolution

_heur = _load_mod("nwjsp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
greedy_insertion = _heur.greedy_insertion
_find_earliest_feasible_start = _heur._find_earliest_feasible_start


def _decode_permutation(instance: NWJSPInstance, perm: list[int]) -> NWJSPSolution:
    """Decode a job permutation into a schedule."""
    n = instance.n
    starts = [0.0] * n
    scheduled: list[int] = []

    for j in perm:
        starts[j] = _find_earliest_feasible_start(instance, starts, scheduled, j)
        scheduled.append(j)

    ms = instance.makespan(starts)
    return NWJSPSolution(job_start_times=starts, makespan=ms)


def simulated_annealing(
    instance: NWJSPInstance,
    max_iterations: int = 20000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> NWJSPSolution:
    """Solve NW-JSP using Simulated Annealing.

    Args:
        instance: A NWJSPInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        NWJSPSolution.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    # Initial permutation from greedy
    init_sol = greedy_insertion(instance)
    # Reconstruct insertion order by sorting start times
    perm = sorted(range(n), key=lambda j: init_sol.job_start_times[j])
    current_sol = _decode_permutation(instance, perm)
    current_ms = current_sol.makespan

    best_perm = perm[:]
    best_sol = current_sol
    best_ms = current_ms

    if initial_temp is None:
        initial_temp = current_ms * 0.2

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_perm = perm[:]
        move = rng.integers(0, 2)

        if move == 0 and n > 1:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        elif n > 1:
            # Insert
            i = rng.integers(0, n)
            item = new_perm.pop(i)
            j = rng.integers(0, len(new_perm) + 1)
            new_perm.insert(j, item)
        else:
            temp *= cooling_rate
            continue

        new_sol = _decode_permutation(instance, new_perm)
        new_ms = new_sol.makespan

        delta = new_ms - current_ms
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            perm = new_perm
            current_ms = new_ms

            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_perm = perm[:]
                best_sol = new_sol

        temp *= cooling_rate

    return best_sol


if __name__ == "__main__":
    inst = NWJSPInstance.random(n=6, m=3, seed=42)
    print(f"NW-JSP: {inst.n} jobs, {inst.m} machines")

    gr_sol = greedy_insertion(inst)
    print(f"Greedy: makespan={gr_sol.makespan:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: makespan={sa_sol.makespan:.1f}")

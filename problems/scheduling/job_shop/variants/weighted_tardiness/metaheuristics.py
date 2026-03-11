"""
Simulated Annealing for Job Shop with Weighted Tardiness.

Problem: Jm || ΣwjTj

Permutation-based: job priority ordering decoded via dispatching.
SA explores priority order neighborhood.

Warm-started with ATC dispatching.

References:
    Singer, M. & Pinedo, M. (1998). A computational study of branch
    and bound for minimizing total weighted tardiness in job shops.
    IIE Transactions, 30(2), 109-118.
    https://doi.org/10.1080/07408179808966441
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


_inst = _load_mod("wtjsp_instance_meta", os.path.join(_this_dir, "instance.py"))
WTJSPInstance = _inst.WTJSPInstance
WTJSPSolution = _inst.WTJSPSolution
schedule_from_sequences = _inst.schedule_from_sequences

_heur = _load_mod("wtjsp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
atc_dispatch = _heur.atc_dispatch


def _decode_priority(instance: WTJSPInstance, job_priority: list[int]) -> WTJSPSolution:
    """Decode job priority into a schedule using dispatching."""
    n, m = instance.n, instance.m
    priority_map = {j: idx for idx, j in enumerate(job_priority)}
    next_op = [0] * n
    machine_time = [0.0] * m
    job_time = [0.0] * n
    machine_sequences: list[list[tuple[int, int]]] = [[] for _ in range(m)]

    total_ops = sum(len(instance.operations[j]) for j in range(n))
    for _ in range(total_ops):
        candidates = []
        for j in range(n):
            if next_op[j] >= len(instance.operations[j]):
                continue
            op_idx = next_op[j]
            mach, dur = instance.operations[j][op_idx]
            start = max(machine_time[mach], job_time[j])
            candidates.append((j, op_idx, mach, dur, start))

        if not candidates:
            break

        best = min(candidates, key=lambda c: priority_map[c[0]])
        j, op_idx, mach, dur, start = best

        machine_sequences[mach].append((j, op_idx))
        machine_time[mach] = start + dur
        job_time[j] = start + dur
        next_op[j] += 1

    ct, _ = schedule_from_sequences(instance, machine_sequences)
    wt = instance.weighted_tardiness(ct)
    return WTJSPSolution(
        machine_sequences=machine_sequences,
        completion_times=ct,
        weighted_tardiness=wt,
    )


def simulated_annealing(
    instance: WTJSPInstance,
    max_iterations: int = 20000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> WTJSPSolution:
    """Solve Jm||ΣwjTj using SA."""
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time_clock = time.time()

    # Warm-start with ATC
    init_sol = atc_dispatch(instance)
    # Reconstruct priority from ATC result
    priority = list(range(n))
    rng.shuffle(priority)
    current_sol = _decode_priority(instance, priority)

    # Try ATC ordering
    atc_sol = init_sol
    if atc_sol.weighted_tardiness < current_sol.weighted_tardiness:
        current_sol = atc_sol

    best_sol = current_sol
    current_wt = current_sol.weighted_tardiness
    best_wt = current_wt

    if initial_temp is None:
        initial_temp = max(best_wt * 0.3, 10.0)

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time_clock >= time_limit:
            break

        new_priority = priority[:]
        if rng.random() < 0.5 and n > 1:
            i, j = rng.choice(n, 2, replace=False)
            new_priority[i], new_priority[j] = new_priority[j], new_priority[i]
        elif n > 1:
            i = rng.integers(0, n)
            item = new_priority.pop(i)
            j = rng.integers(0, len(new_priority) + 1)
            new_priority.insert(j, item)
        else:
            temp *= cooling_rate
            continue

        new_sol = _decode_priority(instance, new_priority)
        new_wt = new_sol.weighted_tardiness

        delta = new_wt - current_wt
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            priority = new_priority
            current_wt = new_wt

            if current_wt < best_wt - 1e-10:
                best_wt = current_wt
                best_sol = new_sol

        temp *= cooling_rate

    return best_sol


if __name__ == "__main__":
    inst = WTJSPInstance.random(n=6, m=3, seed=42)
    atc_sol = atc_dispatch(inst)
    print(f"ATC: wt={atc_sol.weighted_tardiness:.1f}")
    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: wt={sa_sol.weighted_tardiness:.1f}")

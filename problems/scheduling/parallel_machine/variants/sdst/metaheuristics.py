"""
Simulated Annealing for Parallel Machine with SDST.

Problem: Rm | Ssd | Cmax

Neighborhoods:
- Move: transfer a job from the bottleneck machine to another
- Swap: exchange jobs between two machines
- Reorder: swap adjacent jobs on the bottleneck machine

Warm-started with ECT-SDST heuristic.

Complexity: O(iterations * n * m) per run.

References:
    Rabadi, G., Moraga, R.J. & Al-Salem, A. (2006). Heuristics for the
    unrelated parallel machine scheduling problem with setup times.
    Journal of Intelligent Manufacturing, 17(2), 199-207.
    https://doi.org/10.1007/s10845-005-6636-x

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import sys
import os
import math
import time
import copy
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


_inst = _load_mod("pmsdst_instance_meta", os.path.join(_this_dir, "instance.py"))
PMSDSTInstance = _inst.PMSDSTInstance
PMSDSTSolution = _inst.PMSDSTSolution

_heur = _load_mod("pmsdst_heur_meta", os.path.join(_this_dir, "heuristics.py"))
greedy_ect_sdst = _heur.greedy_ect_sdst


def _machine_completion(instance: PMSDSTInstance, jobs: list[int], k: int) -> float:
    """Compute completion time for a sequence of jobs on machine k."""
    t = 0.0
    for idx, j in enumerate(jobs):
        if idx == 0:
            t += instance.setup_times[j][j][k]
        else:
            t += instance.setup_times[jobs[idx - 1]][j][k]
        t += instance.processing_times[j][k]
    return t


def simulated_annealing(
    instance: PMSDSTInstance,
    max_iterations: int = 30000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PMSDSTSolution:
    """Solve Rm|Ssd|Cmax using Simulated Annealing.

    Args:
        instance: A PMSDSTInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        PMSDSTSolution.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m
    start_time = time.time()

    init_sol = greedy_ect_sdst(instance)
    schedule = [s[:] for s in init_sol.schedule]
    current_ms = init_sol.makespan

    best_schedule = [s[:] for s in schedule]
    best_ms = current_ms

    if initial_temp is None:
        initial_temp = best_ms * 0.1

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_schedule = [s[:] for s in schedule]
        move = rng.integers(0, 3)

        # Find bottleneck machine
        machine_loads = [_machine_completion(instance, new_schedule[k], k) for k in range(m)]
        bottleneck = int(np.argmax(machine_loads))

        if move == 0 and len(new_schedule[bottleneck]) > 0:
            # Move: transfer random job from bottleneck to another machine
            idx = rng.integers(0, len(new_schedule[bottleneck]))
            job = new_schedule[bottleneck].pop(idx)
            target = rng.integers(0, m)
            while target == bottleneck and m > 1:
                target = rng.integers(0, m)
            pos = rng.integers(0, len(new_schedule[target]) + 1)
            new_schedule[target].insert(pos, job)

        elif move == 1 and m > 1:
            # Swap: exchange jobs between two machines
            k1 = rng.integers(0, m)
            k2 = rng.integers(0, m)
            while k2 == k1 and m > 1:
                k2 = rng.integers(0, m)
            if new_schedule[k1] and new_schedule[k2]:
                i1 = rng.integers(0, len(new_schedule[k1]))
                i2 = rng.integers(0, len(new_schedule[k2]))
                new_schedule[k1][i1], new_schedule[k2][i2] = (
                    new_schedule[k2][i2], new_schedule[k1][i1]
                )
            else:
                temp *= cooling_rate
                continue

        elif move == 2 and len(new_schedule[bottleneck]) > 1:
            # Reorder: swap adjacent on bottleneck
            idx = rng.integers(0, len(new_schedule[bottleneck]) - 1)
            new_schedule[bottleneck][idx], new_schedule[bottleneck][idx + 1] = (
                new_schedule[bottleneck][idx + 1], new_schedule[bottleneck][idx]
            )
        else:
            temp *= cooling_rate
            continue

        new_ms = instance.makespan(new_schedule)
        delta = new_ms - current_ms
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            schedule = new_schedule
            current_ms = new_ms

            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_schedule = [s[:] for s in schedule]

        temp *= cooling_rate

    return PMSDSTSolution(
        schedule=best_schedule,
        makespan=best_ms,
    )


if __name__ == "__main__":
    inst = PMSDSTInstance.random(n=10, m=3, seed=42)
    print(f"PM-SDST: {inst.n} jobs, {inst.m} machines")

    ect_sol = greedy_ect_sdst(inst)
    print(f"ECT-SDST: makespan={ect_sol.makespan:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: makespan={sa_sol.makespan:.1f}")

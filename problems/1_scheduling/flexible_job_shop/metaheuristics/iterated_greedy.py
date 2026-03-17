"""
Iterated Greedy for Flexible Job Shop Scheduling (FJm || Cmax).

Iteratively destroys the current solution by randomly reassigning a subset
of operations, then reconstructs by greedily reassigning them to the best
machine. Uses Boltzmann acceptance criterion.

Warm-started with best dispatching rule solution.

Complexity: O(iterations * d * total_ops * m) where d = destruction size.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Brandimarte, P. (1993). Routing and scheduling in a flexible job
    shop by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("fjsp_instance_ig", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "fjsp_dispatching_ig",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def iterated_greedy(
    instance: FlexibleJobShopInstance,
    max_iterations: int = 3000,
    d: int | None = None,
    temperature_factor: float = 0.3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """Solve FJSP using Iterated Greedy.

    Args:
        instance: A FlexibleJobShopInstance.
        max_iterations: Maximum number of iterations.
        d: Number of operations to destroy. Default: max(2, total_ops//4).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FlexibleJobShopSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Collect all operations
    all_ops = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            all_ops.append((j, k))
    total_ops = len(all_ops)

    if d is None:
        d = max(2, total_ops // 4)
    d = min(d, total_ops)

    # Warm-start
    best_sol = None
    for rule in ["spt", "lpt", "mwr"]:
        try:
            sol = dispatching_rule(instance, priority_rule=rule)
            if best_sol is None or sol.makespan < best_sol.makespan:
                best_sol = sol
        except Exception:
            continue

    if best_sol is None:
        best_sol = dispatching_rule(instance, priority_rule="spt")

    current_assign = dict(best_sol.assignments)
    current_st, current_ms = _decode(instance, current_assign)
    best_ms = current_ms
    best_assign = dict(current_assign)
    best_st = dict(current_st)

    # Temperature
    temperature = temperature_factor * current_ms / max(instance.n, 1)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: randomly select d operations and clear their assignments
        indices = rng.choice(total_ops, size=d, replace=False)
        destroyed_ops = [all_ops[i] for i in indices]

        new_assign = dict(current_assign)

        # Reconstruction: greedily reassign each destroyed operation
        for j, k in destroyed_ops:
            eligible = instance.jobs[j][k]
            best_mach = None
            best_trial_ms = float("inf")

            for mach in eligible:
                new_assign[(j, k)] = mach
                _, trial_ms = _decode(instance, new_assign)
                if trial_ms < best_trial_ms:
                    best_trial_ms = trial_ms
                    best_mach = mach

            new_assign[(j, k)] = best_mach

        new_st, new_ms = _decode(instance, new_assign)

        # Acceptance
        delta = new_ms - current_ms
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / max(temperature, 1e-10))):
            current_assign = new_assign
            current_st = new_st
            current_ms = new_ms

            if current_ms < best_ms:
                best_ms = current_ms
                best_assign = dict(current_assign)
                best_st = dict(current_st)

    return FlexibleJobShopSolution(
        assignments=best_assign,
        start_times=best_st,
        makespan=best_ms,
    )


def _decode(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], int]:
    """Decode assignments into start times using greedy left-shift."""
    n = instance.n
    m = instance.m
    job_completion = [0] * n
    machine_completion = [0] * m
    start_times: dict[tuple[int, int], int] = {}

    for j in range(n):
        for k in range(len(instance.jobs[j])):
            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            es = max(job_completion[j], machine_completion[mach])
            start_times[(j, k)] = es
            job_completion[j] = es + pt
            machine_completion[mach] = es + pt

    makespan = max(job_completion) if job_completion else 0
    return start_times, makespan


if __name__ == "__main__":
    inst = FlexibleJobShopInstance.random(n=6, m=4, seed=42)
    print(f"FJSP: {inst.n} jobs, {inst.m} machines, {inst.total_operations()} ops")

    disp_sol = dispatching_rule(inst, priority_rule="spt")
    print(f"Dispatching (SPT): makespan={disp_sol.makespan}")

    ig_sol = iterated_greedy(inst, seed=42)
    print(f"IG: makespan={ig_sol.makespan}")

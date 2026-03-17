"""
Variable Neighborhood Search for Flexible Job Shop Scheduling (FJm || Cmax).

VNS uses multiple neighborhood structures to escape local optima:
    N1: Reassign — change machine of one operation (smallest perturbation)
    N2: Swap — exchange machines of two operations on the same machine
    N3: Multi-reassign — reassign k random operations simultaneously

Local search uses best-improvement reassignment.
Warm-started with best dispatching rule solution.

Complexity: O(iterations * k_max * total_ops * m) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Amiri, M., Zandieh, M., Vahdani, B., Soltani, R. & Roshanaei, V.
    (2010). An integrated eigenvector–DEA–TOPSIS methodology for portfolio
    risk evaluation in the FJSP environment. Expert Systems with
    Applications, 37(10), 6940-6948.
    https://doi.org/10.1016/j.eswa.2010.03.010
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("fjsp_instance_vns", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "fjsp_dispatching_vns",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def vns(
    instance: FlexibleJobShopInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """Solve FJSP using Variable Neighborhood Search.

    Args:
        instance: A FlexibleJobShopInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FlexibleJobShopSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

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

    # Collect flexible operations (those with >1 eligible machine)
    flex_ops = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            if len(instance.jobs[j][k]) > 1:
                flex_ops.append((j, k))

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking: perturb k operations
            neighbor = dict(current_assign)
            _shake(instance, neighbor, flex_ops, k, rng)

            # Local search: best-improvement reassignment
            neighbor, ls_st, ls_ms = _local_search(instance, neighbor, flex_ops)

            if ls_ms < current_ms - 1e-10:
                current_assign = neighbor
                current_st = ls_st
                current_ms = ls_ms
                k = 1  # Reset

                if current_ms < best_ms - 1e-10:
                    best_ms = current_ms
                    best_assign = dict(current_assign)
                    best_st = dict(current_st)
            else:
                k += 1

    return FlexibleJobShopSolution(
        assignments=best_assign,
        start_times=best_st,
        makespan=best_ms,
    )


def _shake(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
    flex_ops: list[tuple[int, int]],
    k: int,
    rng: np.random.Generator,
) -> None:
    """Shaking: randomly reassign k operations."""
    if not flex_ops:
        return
    n_perturb = min(k, len(flex_ops))
    indices = rng.choice(len(flex_ops), size=n_perturb, replace=False)
    for idx in indices:
        j, op_k = flex_ops[idx]
        eligible = list(instance.jobs[j][op_k].keys())
        assignments[(j, op_k)] = eligible[rng.integers(len(eligible))]


def _local_search(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
    flex_ops: list[tuple[int, int]],
) -> tuple[dict, dict, int]:
    """Best-improvement local search on machine assignments."""
    improved = True
    current_st, current_ms = _decode(instance, assignments)

    while improved:
        improved = False
        best_delta = 0
        best_op = None
        best_mach = None

        for j, k in flex_ops:
            current_mach = assignments[(j, k)]
            eligible = list(instance.jobs[j][k].keys())
            for new_mach in eligible:
                if new_mach == current_mach:
                    continue
                trial = dict(assignments)
                trial[(j, k)] = new_mach
                _, trial_ms = _decode(instance, trial)
                delta = trial_ms - current_ms
                if delta < best_delta:
                    best_delta = delta
                    best_op = (j, k)
                    best_mach = new_mach

        if best_op is not None:
            assignments[best_op] = best_mach
            current_st, current_ms = _decode(instance, assignments)
            improved = True

    return assignments, current_st, current_ms


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

    vns_sol = vns(inst, seed=42)
    print(f"VNS: makespan={vns_sol.makespan}")

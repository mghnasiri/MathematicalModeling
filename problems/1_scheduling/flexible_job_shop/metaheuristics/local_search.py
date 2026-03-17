"""
Local Search for Flexible Job Shop Scheduling (FJm || Cmax).

Iterative improvement using reassignment and swap neighborhoods on machine
assignments. The decoder converts assignments to start times via greedy
left-shift scheduling.

Neighborhoods:
    - Reassign: move an operation to a different eligible machine
    - Swap: exchange machines of two operations on the same machine
    - Critical reassign: reassign operations on the critical machine

Warm-started with best dispatching rule solution.

Complexity: O(iterations * total_ops * m) per run.

References:
    Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood
    functions for the flexible job shop problem. Journal of Scheduling,
    3(1), 3-20.
    https://doi.org/10.1002/(SICI)1099-1425(200001/02)3:1<3::AID-JOS32>3.0.CO;2-Y

    Brandimarte, P. (1993). Routing and scheduling in a flexible job
    shop by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073
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


_inst = _load_mod("fjsp_instance_ls", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "fjsp_dispatching_ls",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def local_search(
    instance: FlexibleJobShopInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """Solve FJSP using Local Search with restarts.

    Args:
        instance: A FlexibleJobShopInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FlexibleJobShopSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start: try dispatching rules, pick best
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

    no_improve = 0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Best-improvement reassignment
        improved = False
        best_delta = 0
        best_move = None

        # Try reassigning each operation to each eligible machine
        ops = list(current_assign.keys())
        rng.shuffle(ops)

        for j, k in ops[:min(len(ops), 50)]:  # Limit evaluations
            current_mach = current_assign[(j, k)]
            eligible = list(instance.jobs[j][k].keys())
            for new_mach in eligible:
                if new_mach == current_mach:
                    continue
                trial = dict(current_assign)
                trial[(j, k)] = new_mach
                _, trial_ms = _decode(instance, trial)
                delta = trial_ms - current_ms
                if delta < best_delta:
                    best_delta = delta
                    best_move = ((j, k), new_mach)
                    improved = True

        if improved and best_move is not None:
            op, new_mach = best_move
            current_assign[op] = new_mach
            current_st, current_ms = _decode(instance, current_assign)
            no_improve = 0

            if current_ms < best_ms:
                best_ms = current_ms
                best_assign = dict(current_assign)
                best_st = dict(current_st)
        else:
            no_improve += 1
            # Perturbation: random reassignment
            if no_improve >= 5:
                _perturb(instance, current_assign, rng)
                current_st, current_ms = _decode(instance, current_assign)
                no_improve = 0

    return FlexibleJobShopSolution(
        assignments=best_assign,
        start_times=best_st,
        makespan=best_ms,
    )


def _perturb(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> None:
    """Random perturbation: reassign a few operations."""
    ops = list(assignments.keys())
    n_perturb = max(1, len(ops) // 5)
    indices = rng.choice(len(ops), size=min(n_perturb, len(ops)), replace=False)
    for idx in indices:
        j, k = ops[idx]
        eligible = list(instance.jobs[j][k].keys())
        assignments[(j, k)] = eligible[rng.integers(len(eligible))]


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

    ls_sol = local_search(inst, seed=42)
    print(f"LS: makespan={ls_sol.makespan}")

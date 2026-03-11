"""
Tabu Search for Flexible Job Shop Scheduling (FJm || Cmax).

Problem: FJm || Cmax (Flexible Job Shop Makespan)

Neighborhoods:
- Reassign: change the machine assignment of a random operation
- Swap machines: swap machine assignments of two operations

Uses short-term memory preventing recently reassigned operations from
being reassigned again. Aspiration criterion overrides tabu when a
move yields a new global best.

Warm-started with SPT-ECT dispatching rule.

Complexity: O(iterations * n * m) per run.

References:
    Brandimarte, P. (1993). Routing and scheduling in a flexible job
    shop by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073

    Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood
    functions for the flexible job shop problem. Journal of Scheduling,
    3(1), 3-20.
    https://doi.org/10.1002/(SICI)1099-1425(200001/02)3:1<3::AID-JOS32>3.0.CO;2-Y
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("fjsp_instance_ts", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "fjsp_dispatching_ts",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def _decode(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], int]:
    """Decode machine assignments into start times using greedy scheduling."""
    machine_end: dict[int, int] = {}
    job_end = [0] * instance.n
    start_times: dict[tuple[int, int], int] = {}
    makespan = 0

    next_op = [0] * instance.n
    scheduled = 0
    total = sum(len(instance.jobs[j]) for j in range(instance.n))

    while scheduled < total:
        best_op = None
        best_start = float("inf")

        for j in range(instance.n):
            k = next_op[j]
            if k >= len(instance.jobs[j]):
                continue
            mach = assignments[(j, k)]
            earliest = max(machine_end.get(mach, 0), job_end[j])
            if earliest < best_start:
                best_start = earliest
                pt = instance.jobs[j][k][mach]
                best_op = (j, k, mach, pt)

        if best_op is None:
            break

        j, k, mach, pt = best_op
        start = int(best_start)
        end = start + pt
        start_times[(j, k)] = start
        machine_end[mach] = end
        job_end[j] = end
        makespan = max(makespan, end)
        next_op[j] += 1
        scheduled += 1

    return start_times, makespan


def tabu_search(
    instance: FlexibleJobShopInstance,
    max_iterations: int = 3000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """Solve FJSP using Tabu Search.

    Args:
        instance: Flexible job shop instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(n_ops).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FlexibleJobShopSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    all_ops: list[tuple[int, int]] = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            all_ops.append((j, k))

    n_ops = len(all_ops)
    if n_ops == 0:
        return FlexibleJobShopSolution(
            assignments={}, start_times={}, makespan=0
        )

    if tabu_tenure is None:
        tabu_tenure = max(3, int(n_ops ** 0.5))

    # Initialize with dispatching rule
    init_sol = dispatching_rule(instance, priority_rule="spt", machine_rule="ect")
    current_assign = dict(init_sol.assignments)
    current_st, current_ms = _decode(instance, current_assign)

    best_assign = dict(current_assign)
    best_st = dict(current_st)
    best_ms = current_ms

    # Tabu list: (job, op) -> iteration when tabu expires
    tabu_dict: dict[tuple[int, int], int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_delta = float("inf")
        best_move = None

        # ── Reassign neighborhood ────────────────────────────────────────
        for idx in range(n_ops):
            j, k = all_ops[idx]
            eligible = list(instance.jobs[j][k].keys())
            if len(eligible) <= 1:
                continue

            is_tabu = (
                (j, k) in tabu_dict and tabu_dict[(j, k)] > iteration
            )
            current_mach = current_assign[(j, k)]

            for new_mach in eligible:
                if new_mach == current_mach:
                    continue

                test_assign = dict(current_assign)
                test_assign[(j, k)] = new_mach
                _, new_ms = _decode(instance, test_assign)
                delta = new_ms - current_ms

                if is_tabu and current_ms + delta >= best_ms:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("reassign", j, k, new_mach)

        # ── Swap neighborhood (sample) ───────────────────────────────────
        n_swap_candidates = min(n_ops * 3, n_ops * n_ops)
        for _ in range(n_swap_candidates):
            if n_ops < 2:
                break
            i1, i2 = rng.choice(n_ops, size=2, replace=False)
            j1, k1 = all_ops[i1]
            j2, k2 = all_ops[i2]
            m1 = current_assign[(j1, k1)]
            m2 = current_assign[(j2, k2)]

            if m1 == m2:
                continue
            if m2 not in instance.jobs[j1][k1]:
                continue
            if m1 not in instance.jobs[j2][k2]:
                continue

            is_tabu = (
                ((j1, k1) in tabu_dict and tabu_dict[(j1, k1)] > iteration)
                or ((j2, k2) in tabu_dict and tabu_dict[(j2, k2)] > iteration)
            )

            test_assign = dict(current_assign)
            test_assign[(j1, k1)] = m2
            test_assign[(j2, k2)] = m1
            _, new_ms = _decode(instance, test_assign)
            delta = new_ms - current_ms

            if is_tabu and current_ms + delta >= best_ms:
                continue

            if delta < best_delta:
                best_delta = delta
                best_move = ("swap", j1, k1, j2, k2, m2, m1)

        if best_move is None:
            tabu_dict.clear()
            continue

        # ── Apply move ───────────────────────────────────────────────────
        if best_move[0] == "reassign":
            _, j, k, new_mach = best_move
            current_assign[(j, k)] = new_mach
            tabu_dict[(j, k)] = iteration + tabu_tenure

        elif best_move[0] == "swap":
            _, j1, k1, j2, k2, m_new1, m_new2 = best_move
            current_assign[(j1, k1)] = m_new1
            current_assign[(j2, k2)] = m_new2
            tabu_dict[(j1, k1)] = iteration + tabu_tenure
            tabu_dict[(j2, k2)] = iteration + tabu_tenure

        current_st, current_ms = _decode(instance, current_assign)

        if current_ms < best_ms:
            best_ms = current_ms
            best_assign = dict(current_assign)
            best_st = dict(current_st)

    return FlexibleJobShopSolution(
        assignments=best_assign,
        start_times=best_st,
        makespan=best_ms,
    )


if __name__ == "__main__":
    inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=42)
    print(f"FJSP: {inst.n} jobs, {inst.m} machines, {inst.total_operations()} ops")

    sol_disp = dispatching_rule(inst, priority_rule="spt", machine_rule="ect")
    print(f"SPT-ECT: makespan = {sol_disp.makespan}")

    sol_ts = tabu_search(inst, seed=42)
    print(f"TS:      makespan = {sol_ts.makespan}")

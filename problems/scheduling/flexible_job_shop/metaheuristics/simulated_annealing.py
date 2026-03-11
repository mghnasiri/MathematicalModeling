"""
Simulated Annealing for Flexible Job Shop Scheduling (FJm || Cmax).

Problem: FJm || Cmax (Flexible Job Shop Makespan)

Representation: Two vectors per solution:
  - Machine assignment: assigns each operation (j, k) to an eligible machine
  - Operation sequence: determines processing order on each machine

Neighborhoods:
- Reassign: change the machine assignment of a random operation
- Swap: swap two operations on the same machine
- Combined: reassign + local resequence

Warm-started with SPT dispatching rule.

Complexity: O(iterations * n * m) per run.

References:
    Kacem, I., Hammadi, S. & Borne, P. (2002). Approach by localization
    and multiobjective evolutionary optimization for flexible job-shop
    scheduling problems. IEEE Transactions on Systems, Man, and
    Cybernetics, Part C, 32(1), 1-13.
    https://doi.org/10.1109/TSMCC.2002.1009117

    Brandimarte, P. (1993). Routing and scheduling in a flexible job
    shop by tabu search. Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("fjsp_instance_sa", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "fjsp_dispatching_sa",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def _decode(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], int]:
    """Decode machine assignments into start times using greedy scheduling.

    Schedule operations respecting job precedence and machine availability.
    Operations are processed in a priority order based on earliest possible
    start time (ECT-like decoding).

    Args:
        instance: FJSP instance.
        assignments: Dict (job, op) -> machine.

    Returns:
        (start_times dict, makespan)
    """
    # Collect all operations
    all_ops = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            all_ops.append((j, k))

    # Schedule using topological + machine-aware greedy
    machine_end: dict[int, int] = {}
    job_end = [0] * instance.n
    start_times: dict[tuple[int, int], int] = {}
    makespan = 0

    # Process operations in job order, respecting precedence
    next_op = [0] * instance.n  # next unscheduled operation per job
    scheduled = 0
    total = len(all_ops)

    while scheduled < total:
        # Find eligible operations (whose predecessors in the same job are done)
        best_op = None
        best_start = float("inf")

        for j in range(instance.n):
            k = next_op[j]
            if k >= len(instance.jobs[j]):
                continue
            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            earliest = max(machine_end.get(mach, 0), job_end[j])
            if earliest < best_start:
                best_start = earliest
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


def simulated_annealing(
    instance: FlexibleJobShopInstance,
    max_iterations: int = 5000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """Solve FJSP using Simulated Annealing.

    Args:
        instance: Flexible job shop instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FlexibleJobShopSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Collect all operations
    all_ops: list[tuple[int, int]] = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            all_ops.append((j, k))

    n_ops = len(all_ops)
    if n_ops == 0:
        return FlexibleJobShopSolution(
            assignments={}, start_times={}, makespan=0
        )

    # Initialize with dispatching rule
    init_sol = dispatching_rule(instance, priority_rule="spt", machine_rule="ect")
    current_assign = dict(init_sol.assignments)
    current_st, current_ms = _decode(instance, current_assign)

    best_assign = dict(current_assign)
    best_st = dict(current_st)
    best_ms = current_ms

    if initial_temp is None:
        initial_temp = max(1.0, n_ops * 2.0)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_assign = dict(current_assign)

        # Choose neighborhood
        r = rng.random()
        if r < 0.7:
            # Reassign: change machine of a random operation
            idx = rng.integers(0, n_ops)
            j, k = all_ops[idx]
            eligible = list(instance.jobs[j][k].keys())
            if len(eligible) > 1:
                current_mach = new_assign[(j, k)]
                other = [m for m in eligible if m != current_mach]
                new_assign[(j, k)] = other[rng.integers(0, len(other))]
            else:
                temp *= cooling_rate
                continue
        else:
            # Swap machines of two random operations
            if n_ops < 2:
                temp *= cooling_rate
                continue
            i1, i2 = rng.choice(n_ops, size=2, replace=False)
            j1, k1 = all_ops[i1]
            j2, k2 = all_ops[i2]
            m1 = new_assign[(j1, k1)]
            m2 = new_assign[(j2, k2)]
            # Only swap if both machines are eligible for the other operation
            if m2 in instance.jobs[j1][k1] and m1 in instance.jobs[j2][k2]:
                new_assign[(j1, k1)] = m2
                new_assign[(j2, k2)] = m1
            else:
                temp *= cooling_rate
                continue

        new_st, new_ms = _decode(instance, new_assign)

        delta = new_ms - current_ms
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_assign = new_assign
            current_st = new_st
            current_ms = new_ms

            if current_ms < best_ms:
                best_ms = current_ms
                best_assign = dict(current_assign)
                best_st = dict(current_st)

        temp *= cooling_rate

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

    sol_sa = simulated_annealing(inst, seed=42)
    print(f"SA:      makespan = {sol_sa.makespan}")

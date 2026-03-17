"""
Shifting Bottleneck Heuristic for Job Shop Scheduling (Jm || Cmax)

The shifting bottleneck procedure identifies the most critical (bottleneck)
machine at each iteration and solves a single-machine sub-problem to
sequence operations on that machine. After each insertion, previously
scheduled machines are re-optimized.

Simplified version: uses longest-path (critical path) analysis and
EDD-based single machine scheduling for the 1|rj|Lmax sub-problems.

Complexity: O(m^2 * n^2) approximately.

Reference:
    Adams, J., Balas, E. & Zawack, D. (1988).
    "The Shifting Bottleneck Procedure for Job Shop Scheduling."
    Management Science, 34(3), 391-401.
    https://doi.org/10.1287/mnsc.34.3.391
"""

from __future__ import annotations
import sys
import os
import importlib.util
from collections import defaultdict
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_parent(name, filename):
    fp = os.path.join(_parent_dir, filename)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("job_shop_instance_mod", "instance.py")
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences


def shifting_bottleneck(instance: JobShopInstance) -> JobShopSolution:
    """
    Shifting Bottleneck heuristic for JSP.

    Algorithm:
    1. Compute release times and tails via longest-path on job DAG.
    2. For each unscheduled machine, solve 1|rj|Lmax sub-problem.
    3. Schedule the machine with maximum Lmax (bottleneck).
    4. Update release times and repeat.

    Args:
        instance: JSP instance.

    Returns:
        A JobShopSolution.
    """
    n, m = instance.n, instance.m

    # Machine sequences determined so far: machine -> list of (job, pos)
    machine_order: dict[int, list[tuple[int, int]]] = {}
    scheduled_machines: set[int] = set()

    # Build operation info
    ops_on_machine: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for j in range(n):
        for k, (mach, _) in enumerate(instance.jobs[j]):
            ops_on_machine[mach].append((j, k))

    while len(scheduled_machines) < m:
        # Compute release times and tails using current partial schedule
        release, tail = _compute_heads_tails(instance, machine_order)

        # For each unscheduled machine, solve 1|rj|Lmax
        best_machine = -1
        best_lmax = -float('inf')
        best_sequence: list[tuple[int, int]] = []

        for mach in range(m):
            if mach in scheduled_machines:
                continue

            ops = ops_on_machine[mach]
            if not ops:
                scheduled_machines.add(mach)
                machine_order[mach] = []
                continue

            # Solve 1|rj|Lmax by EDD (modified Jackson's rule)
            seq, lmax = _solve_single_machine(instance, ops, release, tail)

            if lmax > best_lmax:
                best_lmax = lmax
                best_machine = mach
                best_sequence = seq

        if best_machine == -1:
            break

        machine_order[best_machine] = best_sequence
        scheduled_machines.add(best_machine)

    # Build start times from machine sequences
    start_times = _build_start_times(instance, machine_order)
    makespan = compute_makespan(instance, start_times)
    machine_seqs = build_machine_sequences(instance, start_times)

    return JobShopSolution(
        start_times=start_times,
        makespan=makespan,
        machine_sequences=machine_seqs,
    )


def _compute_heads_tails(
    instance: JobShopInstance,
    machine_order: dict[int, list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """
    Compute release times (heads) and tails for all operations.

    Heads: earliest start time considering job precedence and machine order.
    Tails: minimum time from end of operation to end of schedule.

    Args:
        instance: JSP instance.
        machine_order: Current machine sequences.

    Returns:
        (release_times, tail_times) as dicts of (job, pos) -> int.
    """
    n, m = instance.n, instance.m
    release: dict[tuple[int, int], int] = {}
    tail: dict[tuple[int, int], int] = {}

    # Build machine predecessor mapping
    machine_pred: dict[tuple[int, int], tuple[int, int]] = {}
    machine_succ: dict[tuple[int, int], tuple[int, int]] = {}
    for mach, seq in machine_order.items():
        for i in range(len(seq) - 1):
            machine_succ[seq[i]] = seq[i + 1]
            machine_pred[seq[i + 1]] = seq[i]

    # Forward pass: compute release times (heads)
    changed = True
    # Initialize
    for j in range(n):
        for k in range(len(instance.jobs[j])):
            release[(j, k)] = 0

    while changed:
        changed = False
        for j in range(n):
            for k in range(len(instance.jobs[j])):
                _, pt_prev = (0, 0)
                new_r = 0

                # Job predecessor
                if k > 0:
                    _, pt_prev_op = instance.jobs[j][k - 1]
                    new_r = max(new_r, release[(j, k - 1)] + pt_prev_op)

                # Machine predecessor
                if (j, k) in machine_pred:
                    pj, pk = machine_pred[(j, k)]
                    _, pt_mp = instance.jobs[pj][pk]
                    new_r = max(new_r, release[(pj, pk)] + pt_mp)

                if new_r > release[(j, k)]:
                    release[(j, k)] = new_r
                    changed = True

    # Backward pass: compute tails
    for j in range(n):
        for k in range(len(instance.jobs[j])):
            tail[(j, k)] = 0

    changed = True
    while changed:
        changed = False
        for j in range(n):
            for k in range(len(instance.jobs[j]) - 1, -1, -1):
                _, pt_cur = instance.jobs[j][k]
                new_t = 0

                # Job successor
                if k < len(instance.jobs[j]) - 1:
                    _, pt_next = instance.jobs[j][k + 1]
                    new_t = max(new_t, tail[(j, k + 1)] + pt_next)

                # Machine successor
                if (j, k) in machine_succ:
                    sj, sk = machine_succ[(j, k)]
                    _, pt_ms = instance.jobs[sj][sk]
                    new_t = max(new_t, tail[(sj, sk)] + pt_ms)

                if new_t > tail[(j, k)]:
                    tail[(j, k)] = new_t
                    changed = True

    return release, tail


def _solve_single_machine(
    instance: JobShopInstance,
    ops: list[tuple[int, int]],
    release: dict[tuple[int, int], int],
    tail: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int]:
    """
    Solve the 1|rj|Lmax sub-problem using modified EDD.

    The due date for each operation is derived from the tail:
        d_op = Cmax_LB - tail[op]
    where Cmax_LB is a lower bound on the makespan.

    Args:
        instance: JSP instance.
        ops: Operations on this machine.
        release: Release times.
        tail: Tail times.

    Returns:
        (sequence, Lmax) for this machine.
    """
    # Compute due dates from tails
    # LB for makespan
    cmax_lb = 0
    for j, k in ops:
        _, pt = instance.jobs[j][k]
        cmax_lb = max(cmax_lb, release[(j, k)] + pt + tail[(j, k)])

    op_info = []
    for j, k in ops:
        _, pt = instance.jobs[j][k]
        r = release[(j, k)]
        d = cmax_lb - tail[(j, k)]
        op_info.append((j, k, pt, r, d))

    # Sort by EDD (earliest due date), break ties by earliest release
    op_info.sort(key=lambda x: (x[4], x[3]))

    # Simulate schedule and compute Lmax
    time = 0
    lmax = -float('inf')
    sequence = []

    remaining = list(op_info)
    scheduled_ops: list[tuple[int, int]] = []

    while remaining:
        # Find available operations
        available = [op for op in remaining if op[3] <= time]

        if not available:
            # Jump to next release time
            time = min(op[3] for op in remaining)
            available = [op for op in remaining if op[3] <= time]

        # Pick by EDD
        available.sort(key=lambda x: (x[4], x[3]))
        selected = available[0]

        j, k, pt, r, d = selected
        start = max(time, r)
        end = start + pt
        lateness = end - d
        lmax = max(lmax, lateness)

        sequence.append((j, k))
        remaining.remove(selected)
        time = end

    return sequence, int(lmax)


def _build_start_times(
    instance: JobShopInstance,
    machine_order: dict[int, list[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    """
    Build feasible start times from complete machine orderings.

    Uses iterative forward pass until convergence.

    Args:
        instance: JSP instance.
        machine_order: Machine -> ordered list of (job, pos).

    Returns:
        Dict (job, pos) -> start time.
    """
    start_times: dict[tuple[int, int], int] = {}

    # Initialize all to 0
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            start_times[(j, k)] = 0

    # Build machine predecessor
    machine_pred: dict[tuple[int, int], tuple[int, int]] = {}
    for mach, seq in machine_order.items():
        for i in range(len(seq) - 1):
            machine_pred[seq[i + 1]] = seq[i]

    # Iterate until stable
    changed = True
    while changed:
        changed = False
        for j in range(instance.n):
            for k in range(len(instance.jobs[j])):
                new_start = 0

                # Job predecessor
                if k > 0:
                    _, pt_prev = instance.jobs[j][k - 1]
                    new_start = max(new_start, start_times[(j, k - 1)] + pt_prev)

                # Machine predecessor
                if (j, k) in machine_pred:
                    pj, pk = machine_pred[(j, k)]
                    _, pt_mp = instance.jobs[pj][pk]
                    new_start = max(new_start, start_times[(pj, pk)] + pt_mp)

                if new_start > start_times[(j, k)]:
                    start_times[(j, k)] = new_start
                    changed = True

    return start_times


if __name__ == "__main__":
    from instance import ft06

    print("=== Shifting Bottleneck on ft06 (optimal=55) ===\n")
    inst = ft06()
    sol = shifting_bottleneck(inst)
    print(f"Makespan: {sol.makespan}")
    print(f"\nMachine sequences:")
    if sol.machine_sequences:
        for m in sorted(sol.machine_sequences):
            jobs = [f"J{j}" for j, _ in sol.machine_sequences[m]]
            print(f"  M{m}: {' -> '.join(jobs)}")

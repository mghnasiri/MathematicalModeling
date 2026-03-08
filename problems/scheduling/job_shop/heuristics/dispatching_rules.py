"""
Dispatching Rules for Job Shop Scheduling (Jm || Cmax)

Implements priority-based dispatching rules using the Giffler & Thompson (1960)
active schedule generation procedure. Available rules:

- SPT (Shortest Processing Time): prioritize shortest next operation
- LPT (Longest Processing Time): prioritize longest next operation
- MWR (Most Work Remaining): prioritize job with most remaining processing
- LWR (Least Work Remaining): prioritize job with least remaining processing
- FIFO (First In First Out): prioritize job that arrived earliest
- RANDOM: random selection among eligible operations

Complexity: O(n * m * n) per schedule generation.

Reference:
    Giffler, B. & Thompson, G.L. (1960).
    "Algorithms for Solving Production-Scheduling Problems."
    Operations Research, 8(4), 487-503.
    https://doi.org/10.1287/opre.8.4.487
"""

from __future__ import annotations
import sys
import os
import importlib.util
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_parent_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_parent_dir, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent_module("job_shop_instance_mod", "instance.py")
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences


def dispatching_rule(
    instance: JobShopInstance,
    rule: str = "spt",
    seed: int | None = None,
) -> JobShopSolution:
    """
    Generate an active schedule using a dispatching rule.

    Uses the Giffler & Thompson (1960) procedure: at each step, find the
    operation with earliest possible completion, then among all operations
    that could start before that completion on the same machine, select
    one by the dispatching rule.

    Args:
        instance: JSP instance.
        rule: One of "spt", "lpt", "mwr", "lwr", "fifo", "random".
        seed: Random seed (only used for "random" rule).

    Returns:
        A JobShopSolution.
    """
    rng = np.random.default_rng(seed) if rule == "random" else None

    # Track next operation index for each job
    next_op = [0] * instance.n
    # Track earliest available time for each machine and each job
    machine_available = [0] * instance.m
    job_available = [0] * instance.n
    # Start times
    start_times: dict[tuple[int, int], int] = {}

    total_ops = instance.total_operations()
    scheduled = 0

    while scheduled < total_ops:
        # Collect schedulable operations (next unscheduled op of each job)
        candidates = []
        for j in range(instance.n):
            k = next_op[j]
            if k < len(instance.jobs[j]):
                mach, pt = instance.jobs[j][k]
                earliest_start = max(machine_available[mach], job_available[j])
                earliest_end = earliest_start + pt
                candidates.append((j, k, mach, pt, earliest_start, earliest_end))

        if not candidates:
            break

        # Find minimum earliest completion time
        min_end = min(c[5] for c in candidates)
        # Find the machine of that operation
        bottleneck_op = min(candidates, key=lambda c: c[5])
        bottleneck_machine = bottleneck_op[2]

        # Filter: only operations on bottleneck machine that can start
        # before min_end (Giffler-Thompson conflict set)
        conflict_set = [
            c for c in candidates
            if c[2] == bottleneck_machine and c[4] < min_end
        ]

        # Apply dispatching rule to select from conflict set
        selected = _apply_rule(instance, conflict_set, rule, next_op, rng)

        j, k, mach, pt, earliest_start, _ = selected
        start_times[(j, k)] = earliest_start
        machine_available[mach] = earliest_start + pt
        job_available[j] = earliest_start + pt
        next_op[j] += 1
        scheduled += 1

    makespan = compute_makespan(instance, start_times)
    machine_seqs = build_machine_sequences(instance, start_times)

    return JobShopSolution(
        start_times=start_times,
        makespan=makespan,
        machine_sequences=machine_seqs,
    )


def _apply_rule(
    instance: JobShopInstance,
    conflict_set: list[tuple[int, int, int, int, int, int]],
    rule: str,
    next_op: list[int],
    rng: np.random.Generator | None,
) -> tuple[int, int, int, int, int, int]:
    """Apply a dispatching rule to select from a conflict set."""
    rule = rule.lower()

    if rule == "spt":
        return min(conflict_set, key=lambda c: c[3])
    elif rule == "lpt":
        return max(conflict_set, key=lambda c: c[3])
    elif rule == "mwr":
        def work_remaining(c):
            j = c[0]
            return sum(
                pt for _, pt in instance.jobs[j][next_op[j]:]
            )
        return max(conflict_set, key=work_remaining)
    elif rule == "lwr":
        def work_remaining(c):
            j = c[0]
            return sum(
                pt for _, pt in instance.jobs[j][next_op[j]:]
            )
        return min(conflict_set, key=work_remaining)
    elif rule == "fifo":
        return min(conflict_set, key=lambda c: c[4])
    elif rule == "random":
        idx = rng.integers(0, len(conflict_set))
        return conflict_set[idx]
    else:
        raise ValueError(f"Unknown rule: {rule}. Use spt/lpt/mwr/lwr/fifo/random.")


def spt(instance: JobShopInstance) -> JobShopSolution:
    """Shortest Processing Time dispatching rule."""
    return dispatching_rule(instance, rule="spt")


def lpt(instance: JobShopInstance) -> JobShopSolution:
    """Longest Processing Time dispatching rule."""
    return dispatching_rule(instance, rule="lpt")


def mwr(instance: JobShopInstance) -> JobShopSolution:
    """Most Work Remaining dispatching rule."""
    return dispatching_rule(instance, rule="mwr")


def lwr(instance: JobShopInstance) -> JobShopSolution:
    """Least Work Remaining dispatching rule."""
    return dispatching_rule(instance, rule="lwr")


def fifo(instance: JobShopInstance) -> JobShopSolution:
    """First In First Out dispatching rule."""
    return dispatching_rule(instance, rule="fifo")


if __name__ == "__main__":
    from instance import ft06

    print("=== Dispatching Rules on ft06 (optimal=55) ===\n")
    inst = ft06()

    for rule_name in ["spt", "lpt", "mwr", "lwr", "fifo", "random"]:
        sol = dispatching_rule(inst, rule=rule_name, seed=42)
        print(f"  {rule_name.upper():6s}: makespan = {sol.makespan}")

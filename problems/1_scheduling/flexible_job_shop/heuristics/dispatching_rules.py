"""
Dispatching Rules for Flexible Job Shop Scheduling (FJm || Cmax)

Adapted dispatching rules for FJSP. Each operation must be both assigned
to an eligible machine and sequenced. Rules combine machine selection
(routing) with operation priority (sequencing).

Machine selection strategies:
- ECT (Earliest Completion Time): assign to machine finishing earliest
- LPT (Least Processing Time): assign to machine with shortest processing time

Operation priority rules:
- SPT, LPT, MWR, LWR (same as JSP but with machine flexibility)

Complexity: O(n * max_ops * m) per schedule generation.

Reference:
    Brandimarte, P. (1993).
    "Routing and scheduling in a flexible job shop by tabu search."
    Annals of Operations Research, 41(3), 157-183.
    https://doi.org/10.1007/BF02023073
"""

from __future__ import annotations
import sys
import os
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

_inst = _load_mod("fjsp_instance_mod", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution


def dispatching_rule(
    instance: FlexibleJobShopInstance,
    priority_rule: str = "spt",
    machine_rule: str = "ect",
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """
    Generate a schedule using combined dispatching and machine assignment rules.

    At each step:
    1. Select the highest-priority schedulable operation.
    2. Assign it to the best machine according to machine_rule.
    3. Schedule it at the earliest feasible time on that machine.

    Args:
        instance: FJSP instance.
        priority_rule: "spt", "lpt", "mwr", "lwr", "random".
        machine_rule: "ect" (earliest completion), "spt" (shortest processing).
        seed: Random seed (for "random" rule).

    Returns:
        A FlexibleJobShopSolution.
    """
    rng = np.random.default_rng(seed) if priority_rule == "random" else None

    next_op = [0] * instance.n
    machine_available = [0] * instance.m
    job_available = [0] * instance.n

    assignments: dict[tuple[int, int], int] = {}
    start_times: dict[tuple[int, int], int] = {}

    total_ops = instance.total_operations()
    scheduled = 0

    while scheduled < total_ops:
        # Collect schedulable operations
        candidates = []
        for j in range(instance.n):
            k = next_op[j]
            if k < len(instance.jobs[j]):
                candidates.append((j, k))

        if not candidates:
            break

        # Select operation by priority rule
        selected_jk = _select_operation(
            instance, candidates, priority_rule, next_op, job_available, rng
        )
        j, k = selected_jk
        eligible = instance.jobs[j][k]

        # Select machine
        best_machine = _select_machine(
            eligible, machine_available, job_available[j], machine_rule
        )
        pt = eligible[best_machine]
        start = max(machine_available[best_machine], job_available[j])

        assignments[(j, k)] = best_machine
        start_times[(j, k)] = start
        machine_available[best_machine] = start + pt
        job_available[j] = start + pt
        next_op[j] += 1
        scheduled += 1

    makespan = compute_makespan(instance, assignments, start_times)

    return FlexibleJobShopSolution(
        assignments=assignments,
        start_times=start_times,
        makespan=makespan,
    )


def _select_operation(
    instance: FlexibleJobShopInstance,
    candidates: list[tuple[int, int]],
    rule: str,
    next_op: list[int],
    job_available: list[int],
    rng: np.random.Generator | None,
) -> tuple[int, int]:
    """Select an operation based on the priority rule."""
    rule = rule.lower()

    if rule == "spt":
        return min(candidates, key=lambda jk: min(
            instance.jobs[jk[0]][jk[1]].values()
        ))
    elif rule == "lpt":
        return max(candidates, key=lambda jk: max(
            instance.jobs[jk[0]][jk[1]].values()
        ))
    elif rule == "mwr":
        def work_remaining(jk):
            j = jk[0]
            total = 0
            for k2 in range(next_op[j], len(instance.jobs[j])):
                total += min(instance.jobs[j][k2].values())
            return total
        return max(candidates, key=work_remaining)
    elif rule == "lwr":
        def work_remaining(jk):
            j = jk[0]
            total = 0
            for k2 in range(next_op[j], len(instance.jobs[j])):
                total += min(instance.jobs[j][k2].values())
            return total
        return min(candidates, key=work_remaining)
    elif rule == "random":
        idx = rng.integers(0, len(candidates))
        return candidates[idx]
    else:
        raise ValueError(f"Unknown priority rule: {rule}")


def _select_machine(
    eligible: dict[int, int],
    machine_available: list[int],
    job_ready: int,
    rule: str,
) -> int:
    """Select a machine for an operation."""
    rule = rule.lower()

    if rule == "ect":
        # Earliest completion time
        return min(
            eligible.keys(),
            key=lambda m: max(machine_available[m], job_ready) + eligible[m]
        )
    elif rule == "spt":
        # Shortest processing time on this operation
        return min(eligible.keys(), key=lambda m: eligible[m])
    else:
        raise ValueError(f"Unknown machine rule: {rule}")


if __name__ == "__main__":
    print("=== FJSP Dispatching Rules ===\n")

    inst = FlexibleJobShopInstance.random(n=5, m=3, flexibility=0.6, seed=42)
    for pr in ["spt", "lpt", "mwr", "lwr"]:
        sol = dispatching_rule(inst, priority_rule=pr, machine_rule="ect")
        print(f"  {pr.upper():4s}/ECT: makespan = {sol.makespan}")

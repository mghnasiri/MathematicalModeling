"""
Hierarchical (Route-Then-Sequence) Heuristic for FJSP (FJm || Cmax)

Decomposes the FJSP into two phases:
1. Routing: Assign each operation to a machine (load balancing).
2. Sequencing: Schedule operations on each machine (dispatching).

Phase 1 uses a load-balancing greedy assignment. Phase 2 uses
earliest-start scheduling with job precedence.

Complexity: O(n * max_ops * m) for routing + O(total_ops^2) for sequencing.

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


def hierarchical(
    instance: FlexibleJobShopInstance,
    routing_strategy: str = "min_load",
) -> FlexibleJobShopSolution:
    """
    Hierarchical route-then-sequence heuristic.

    Args:
        instance: FJSP instance.
        routing_strategy: "min_load" (balance machine loads) or
                         "min_pt" (shortest processing time).

    Returns:
        A FlexibleJobShopSolution.
    """
    # Phase 1: Routing — assign each operation to a machine
    assignments = _route_operations(instance, routing_strategy)

    # Phase 2: Sequencing — build feasible schedule
    start_times = _sequence_operations(instance, assignments)

    makespan = compute_makespan(instance, assignments, start_times)

    return FlexibleJobShopSolution(
        assignments=assignments,
        start_times=start_times,
        makespan=makespan,
    )


def _route_operations(
    instance: FlexibleJobShopInstance,
    strategy: str,
) -> dict[tuple[int, int], int]:
    """
    Assign operations to machines.

    Args:
        instance: FJSP instance.
        strategy: "min_load" or "min_pt".

    Returns:
        Dict (job, pos) -> machine.
    """
    assignments: dict[tuple[int, int], int] = {}
    machine_loads = [0.0] * instance.m

    # Process operations in order of decreasing total work
    all_ops = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            avg_pt = np.mean(list(instance.jobs[j][k].values()))
            all_ops.append((avg_pt, j, k))

    # Sort by descending average processing time (LPT-style)
    all_ops.sort(reverse=True)

    for _, j, k in all_ops:
        eligible = instance.jobs[j][k]

        if strategy == "min_load":
            # Assign to eligible machine with minimum current load
            best_machine = min(
                eligible.keys(),
                key=lambda m: machine_loads[m] + eligible[m]
            )
        elif strategy == "min_pt":
            best_machine = min(eligible.keys(), key=lambda m: eligible[m])
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        assignments[(j, k)] = best_machine
        machine_loads[best_machine] += eligible[best_machine]

    return assignments


def _sequence_operations(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """
    Sequence operations on their assigned machines.

    Uses a greedy earliest-start approach respecting job precedence.
    """
    start_times: dict[tuple[int, int], int] = {}
    next_op = [0] * instance.n
    machine_available = [0] * instance.m
    job_available = [0] * instance.n

    total_ops = instance.total_operations()
    scheduled = 0

    while scheduled < total_ops:
        # Find the operation that can start earliest
        best_start = float('inf')
        best_jk = None

        for j in range(instance.n):
            k = next_op[j]
            if k >= len(instance.jobs[j]):
                continue
            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            start = max(machine_available[mach], job_available[j])
            if start < best_start:
                best_start = start
                best_jk = (j, k)

        if best_jk is None:
            break

        j, k = best_jk
        mach = assignments[(j, k)]
        pt = instance.jobs[j][k][mach]
        start = max(machine_available[mach], job_available[j])

        start_times[(j, k)] = start
        machine_available[mach] = start + pt
        job_available[j] = start + pt
        next_op[j] += 1
        scheduled += 1

    return start_times


if __name__ == "__main__":
    print("=== Hierarchical Heuristic for FJSP ===\n")

    inst = FlexibleJobShopInstance.random(n=5, m=3, flexibility=0.6, seed=42)
    for strategy in ["min_load", "min_pt"]:
        sol = hierarchical(inst, routing_strategy=strategy)
        valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
        print(f"  {strategy}: makespan = {sol.makespan}, valid = {valid}")

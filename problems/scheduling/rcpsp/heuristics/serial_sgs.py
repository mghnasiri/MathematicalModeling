"""
Serial Schedule Generation Scheme (Serial SGS) for RCPSP

The Serial SGS schedules one activity at a time in a given priority order.
For each activity, it finds the earliest feasible start time that respects
both precedence and resource constraints.

The Serial SGS generates active schedules.

Complexity: O(n^2 * K * T) where T is the time horizon.

Reference:
    Kolisch, R. (1996).
    "Serial and parallel resource-constrained project scheduling methods revisited."
    European Journal of Operational Research, 90(2), 320-333.
    https://doi.org/10.1016/0377-2217(95)00357-6
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

_inst = _load_mod("rcpsp_instance_mod", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution
validate_solution = _inst.validate_solution


def serial_sgs(
    instance: RCPSPInstance,
    priority_list: list[int] | None = None,
    priority_rule: str = "lft",
) -> RCPSPSolution:
    """
    Serial Schedule Generation Scheme.

    Args:
        instance: RCPSP instance.
        priority_list: Ordered list of activities (0..n+1) to schedule.
                      If None, generated using priority_rule.
        priority_rule: "lft" (Latest Finish Time), "est" (Earliest Start Time),
                      "mts" (Most Total Successors), "grpw" (Greatest Rank
                      Positional Weight). Used only if priority_list is None.

    Returns:
        An RCPSPSolution.
    """
    total = instance.n + 2

    if priority_list is None:
        priority_list = _generate_priority_list(instance, priority_rule)

    # Upper bound on time horizon
    time_horizon = int(instance.durations.sum()) + 1

    # Resource usage profile
    resource_usage = np.zeros((time_horizon, instance.num_resources), dtype=int)

    start_times = np.zeros(total, dtype=int)
    scheduled = set()

    for act in priority_list:
        if act in scheduled:
            continue

        # Earliest start from precedence
        es = 0
        for pred in instance.predecessors.get(act, []):
            if pred in scheduled:
                es = max(es, start_times[pred] + instance.durations[pred])

        # Find earliest feasible start respecting resources
        duration = instance.durations[act]
        if duration == 0:
            start_times[act] = es
            scheduled.add(act)
            continue

        demands = instance.resource_demands[act]

        t = es
        while t + duration <= time_horizon:
            feasible = True
            for tau in range(t, t + duration):
                for k in range(instance.num_resources):
                    if (resource_usage[tau, k] + demands[k] >
                            instance.resource_capacities[k]):
                        feasible = False
                        break
                if not feasible:
                    break

            if feasible:
                # Schedule activity at time t
                start_times[act] = t
                for tau in range(t, t + duration):
                    resource_usage[tau] += demands
                scheduled.add(act)
                break

            t += 1

    makespan = int(start_times[instance.n + 1])

    return RCPSPSolution(start_times=start_times, makespan=makespan)


def _generate_priority_list(
    instance: RCPSPInstance,
    rule: str,
) -> list[int]:
    """Generate a precedence-feasible priority list using a priority rule."""
    total = instance.n + 2

    if rule == "lft":
        ls = instance.latest_start_times()
        lf = ls + instance.durations
        priorities = {i: lf[i] for i in range(total)}
    elif rule == "est":
        es = instance.earliest_start_times()
        priorities = {i: es[i] for i in range(total)}
    elif rule == "mts":
        priorities = {i: -_count_successors(instance, i) for i in range(total)}
    elif rule == "grpw":
        priorities = {i: -_grpw(instance, i) for i in range(total)}
    else:
        raise ValueError(f"Unknown priority rule: {rule}")

    # Build precedence-feasible list using topological sort with priority
    in_degree = {i: len(instance.predecessors.get(i, [])) for i in range(total)}
    eligible = [i for i in range(total) if in_degree[i] == 0]
    result = []

    while eligible:
        eligible.sort(key=lambda x: priorities[x])
        act = eligible.pop(0)
        result.append(act)
        for succ in instance.successors.get(act, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.append(succ)

    return result


def _count_successors(instance: RCPSPInstance, act: int) -> int:
    """Count total successors (transitive) of an activity."""
    visited = set()
    stack = [act]
    while stack:
        node = stack.pop()
        for succ in instance.successors.get(node, []):
            if succ not in visited:
                visited.add(succ)
                stack.append(succ)
    return len(visited)


def _grpw(instance: RCPSPInstance, act: int) -> int:
    """Greatest Rank Positional Weight = duration + sum of successor durations."""
    weight = int(instance.durations[act])
    visited = set()
    stack = list(instance.successors.get(act, []))
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        weight += int(instance.durations[node])
        stack.extend(instance.successors.get(node, []))
    return weight


if __name__ == "__main__":
    print("=== Serial SGS for RCPSP ===\n")

    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"Critical path LB: {inst.critical_path_length()}")

    for rule in ["lft", "est", "mts", "grpw"]:
        sol = serial_sgs(inst, priority_rule=rule)
        valid, _ = validate_solution(inst, sol.start_times)
        print(f"  {rule.upper()}: makespan = {sol.makespan}, valid = {valid}")

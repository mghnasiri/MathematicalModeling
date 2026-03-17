"""Ranked Positional Weight (RPW) heuristic for SALBP-1.

Algorithm: Compute positional weight for each task (own time + sum of all
successor times). Sort tasks by decreasing positional weight. Assign tasks
to stations in order, opening new stations when cycle time would be exceeded
or precedence is violated.

Complexity: O(n^2) for positional weight computation + O(n log n) for sorting.

References:
    Helgeson, W. B., & Birnie, D. P. (1961). Assembly line balancing using
    the ranked positional weight technique. Journal of Industrial Engineering,
    12(6), 394-398.
"""
import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "salbp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

SALBPInstance = _inst.SALBPInstance
SALBPSolution = _inst.SALBPSolution


def _compute_positional_weights(instance: SALBPInstance) -> np.ndarray:
    """Compute ranked positional weight for each task.

    RPW(i) = t_i + sum of RPW of all successors of i.
    Computed via reverse topological order.

    Args:
        instance: SALBPInstance.

    Returns:
        Array of positional weights for each task.
    """
    n = instance.n_tasks
    succ = instance.successors()

    # Compute all-successors (transitive closure)
    all_succ: dict[int, set[int]] = {i: set() for i in range(n)}
    # Build in reverse: if i -> j, then all_succ[i] includes j and all_succ[j]
    # Use topological sort
    in_degree = np.zeros(n, dtype=int)
    for p, s in instance.precedences:
        in_degree[s] += 1

    # Reverse topological order
    pred_map = instance.predecessors()
    out_degree = np.zeros(n, dtype=int)
    for p, s in instance.precedences:
        out_degree[p] += 1

    queue = [i for i in range(n) if out_degree[i] == 0]
    rpw = np.copy(instance.processing_times).astype(float)

    processed = set()
    while queue:
        task = queue.pop(0)
        processed.add(task)
        # RPW of task already includes its own time
        # Add RPW contribution to predecessors
        for pred in pred_map[task]:
            rpw[pred] += instance.processing_times[task]
            # Check if all successors of pred are processed
            if all(s in processed for s in succ[pred]):
                queue.append(pred)

    # Simpler approach: RPW = own time + sum of all successor times
    # Compute transitive successors
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            s = stack.pop()
            if s not in visited:
                visited.add(s)
                stack.extend(succ[s])
        all_succ[i] = visited

    rpw = np.array([
        instance.processing_times[i] + sum(instance.processing_times[s] for s in all_succ[i])
        for i in range(n)
    ])

    return rpw


def rpw_heuristic(instance: SALBPInstance) -> SALBPSolution:
    """Ranked Positional Weight heuristic for SALBP-1.

    Args:
        instance: A SALBPInstance.

    Returns:
        SALBPSolution with task-to-station assignments.
    """
    n = instance.n_tasks
    rpw = _compute_positional_weights(instance)
    pred_map = instance.predecessors()

    # Sort tasks by decreasing RPW
    task_order = np.argsort(-rpw)

    assignment: dict[int, int] = {}
    station_times: list[float] = []
    assigned: set[int] = set()

    # Track which station each task is in
    current_station = 0
    station_times.append(0.0)

    # Assign tasks in RPW order
    unassigned = list(task_order)

    while unassigned:
        assigned_this_pass = []
        for task in unassigned:
            task = int(task)
            # Check precedence: all predecessors must be assigned
            preds_ok = all(p in assigned for p in pred_map[task])
            if not preds_ok:
                continue

            # Check if fits in current station
            if station_times[current_station] + instance.processing_times[task] <= instance.cycle_time + 1e-9:
                assignment[task] = current_station
                station_times[current_station] += instance.processing_times[task]
                assigned.add(task)
                assigned_this_pass.append(task)

        for task in assigned_this_pass:
            unassigned.remove(task)

        if unassigned and not assigned_this_pass:
            # Open a new station
            current_station += 1
            station_times.append(0.0)

    n_stations = current_station + 1

    # Check feasibility
    feasible = True
    # Check precedence
    for pred, succ in instance.precedences:
        if assignment[pred] > assignment[succ]:
            feasible = False
            break
    # Check cycle time
    for t in station_times:
        if t > instance.cycle_time + 1e-9:
            feasible = False
            break

    return SALBPSolution(
        assignment=assignment, n_stations=n_stations,
        station_times=station_times, feasible=feasible
    )


if __name__ == "__main__":
    inst = SALBPInstance.random()
    sol = rpw_heuristic(inst)
    print(f"Instance: {inst.n_tasks} tasks, cycle time = {inst.cycle_time}")
    print(sol)
    print(f"Assignment: {sol.assignment}")

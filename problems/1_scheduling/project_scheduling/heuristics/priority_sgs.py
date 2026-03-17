"""Priority-based Serial Schedule Generation Scheme for Multi-Project Scheduling.

Extends the RCPSP serial SGS to multiple projects with shared resources.
Activities across all projects are merged into a single priority list and
scheduled one at a time at the earliest feasible start time.

Priority rules: SPT (shortest processing time), LFT (latest finish time
from critical path), or project-weight based.

Complexity: O(N^2 * K) where N = total activities, K = resources.

References:
    Lova, A., Tormos, P., Cervantes, M., & Barber, F. (2009). An efficient
    hybrid genetic algorithm for scheduling projects with resource constraints.
    International Journal of Production Economics, 117(2), 302-316.
"""
from __future__ import annotations

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
    "mps_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MultiProjectInstance = _inst.MultiProjectInstance
MultiProjectSolution = _inst.MultiProjectSolution
Project = _inst.Project


def _compute_earliest_start(project: Project) -> np.ndarray:
    """Compute earliest start times based on precedence only.

    Args:
        project: A Project.

    Returns:
        Array of earliest start times.
    """
    est = np.zeros(project.n_activities, dtype=int)
    for j in range(project.n_activities):
        for pred in project.predecessors[j]:
            est[j] = max(est[j], est[pred] + project.durations[pred])
    return est


def priority_sgs(instance: MultiProjectInstance,
                 rule: str = "spt") -> MultiProjectSolution:
    """Priority-based serial SGS for multi-project scheduling.

    Merges all activities from all projects, sorts by priority rule,
    and schedules each at the earliest feasible time respecting
    precedence and resource constraints.

    Args:
        instance: A MultiProjectInstance.
        rule: Priority rule — "spt" (shortest processing time),
              "est" (earliest start time), or "weight" (project weight).

    Returns:
        A MultiProjectSolution.
    """
    # Build activity list: (project_idx, activity_idx)
    activities = []
    est_all = {}
    for p_idx, proj in enumerate(instance.projects):
        est = _compute_earliest_start(proj)
        for j in range(proj.n_activities):
            activities.append((p_idx, j))
            est_all[(p_idx, j)] = est[j]

    # Sort by priority rule
    if rule == "spt":
        activities.sort(key=lambda x: instance.projects[x[0]].durations[x[1]])
    elif rule == "est":
        activities.sort(key=lambda x: est_all[x])
    elif rule == "weight":
        activities.sort(
            key=lambda x: -instance.projects[x[0]].weight
        )
    else:
        raise ValueError(f"Unknown rule: {rule}")

    # Initialize
    start_times = [np.full(proj.n_activities, -1, dtype=int)
                   for proj in instance.projects]
    # Track resource usage: resource_usage[t][k] = units used at time t
    max_time = sum(int(proj.durations.sum()) for proj in instance.projects) + 100
    resource_usage = np.zeros((max_time, instance.n_resources), dtype=int)

    scheduled = set()

    # Schedule activities respecting topological order
    # We may need multiple passes since priority order may violate precedence
    remaining = list(activities)
    max_iters = len(remaining) * 2

    iteration = 0
    while remaining and iteration < max_iters:
        iteration += 1
        progress = False
        next_remaining = []

        for p_idx, j in remaining:
            proj = instance.projects[p_idx]
            # Check precedence
            preds_done = all((p_idx, pred) in scheduled
                             for pred in proj.predecessors[j])
            if not preds_done:
                next_remaining.append((p_idx, j))
                continue

            # Earliest start from precedence
            es = 0
            for pred in proj.predecessors[j]:
                es = max(es, start_times[p_idx][pred] + proj.durations[pred])

            dur = proj.durations[j]
            req = proj.resource_requirements[j]

            # Find earliest feasible start
            t = es
            while t + dur < max_time:
                feasible = True
                for tt in range(t, t + dur):
                    for k in range(instance.n_resources):
                        if (resource_usage[tt, k] + req[k]
                                > instance.resource_capacities[k]):
                            feasible = False
                            break
                    if not feasible:
                        break
                if feasible:
                    break
                t += 1

            # Schedule the activity
            start_times[p_idx][j] = t
            for tt in range(t, t + dur):
                resource_usage[tt] += req
            scheduled.add((p_idx, j))
            progress = True

        remaining = next_remaining
        if not progress and remaining:
            # Force schedule remaining (should not happen with valid instances)
            break

    # Compute makespans and objective
    makespans = []
    total_obj = 0.0
    for p_idx, proj in enumerate(instance.projects):
        sink = proj.n_activities - 1
        ms = int(start_times[p_idx][sink] + proj.durations[sink])
        makespans.append(ms)
        tardiness = max(0, ms - proj.deadline)
        total_obj += proj.weight * tardiness

    return MultiProjectSolution(
        start_times=start_times,
        project_makespans=makespans,
        objective=total_obj,
    )


if __name__ == "__main__":
    inst = MultiProjectInstance.random(n_projects=3, n_activities=6,
                                        n_resources=2, seed=42)
    print(f"Instance: {inst.n_projects} projects, {inst.n_resources} resources")
    for p in inst.projects:
        print(f"  Project {p.project_id}: {p.n_activities} activities, "
              f"deadline={p.deadline}, weight={p.weight:.2f}")

    for rule in ["spt", "est", "weight"]:
        sol = priority_sgs(inst, rule=rule)
        print(f"\n{rule.upper()} rule: {sol}")

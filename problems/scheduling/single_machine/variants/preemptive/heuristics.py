"""
Optimal Algorithms for Preemptive Single Machine Scheduling.

Problem: 1 | pmtn, rj | ΣCj
Complexity: O(n^2) for SRPT

SRPT (Shortest Remaining Processing Time): at each event (release or
completion), preempt to the job with shortest remaining processing time.
Optimal for 1|pmtn,rj|ΣCj.

References:
    Schrage, L. (1968). A proof of the optimality of the shortest remaining
    processing time discipline. Operations Research, 16(3), 687-690.
    https://doi.org/10.1287/opre.16.3.687
"""

from __future__ import annotations

import sys
import os
import heapq
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("preemptive_instance_h", os.path.join(_this_dir, "instance.py"))
PreemptiveSMInstance = _inst.PreemptiveSMInstance
PreemptiveSMSolution = _inst.PreemptiveSMSolution


def srpt(instance: PreemptiveSMInstance) -> PreemptiveSMSolution:
    """SRPT: optimal for 1|pmtn,rj|ΣCj.

    At each time point, process the available job with shortest
    remaining processing time. Preempt when a new job arrives with
    shorter remaining time.

    Args:
        instance: A PreemptiveSMInstance.

    Returns:
        PreemptiveSMSolution.
    """
    n = instance.n
    remaining = instance.processing_times.copy()
    completion_times = [0.0] * n
    segments: list[tuple[int, float, float]] = []

    # Events: release dates
    events = sorted(set(instance.release_dates))
    events_set = set(events)

    t = 0.0
    available: list[tuple[float, int]] = []  # (remaining_time, job)

    event_idx = 0

    while True:
        # Add newly released jobs
        while event_idx < len(events) and events[event_idx] <= t + 1e-10:
            for j in range(n):
                if abs(instance.release_dates[j] - events[event_idx]) < 1e-10 and remaining[j] > 1e-10:
                    heapq.heappush(available, (remaining[j], j))
            event_idx += 1

        # Remove completed jobs
        available = [(r, j) for r, j in available if remaining[j] > 1e-10]
        heapq.heapify(available)

        if not available:
            if event_idx < len(events):
                t = events[event_idx]
                continue
            else:
                break

        _, current_job = available[0]

        # Next event: either next release or completion of current job
        next_release = events[event_idx] if event_idx < len(events) else float("inf")
        completion_at = t + remaining[current_job]
        next_event = min(next_release, completion_at)

        duration = next_event - t
        segments.append((current_job, t, next_event))
        remaining[current_job] -= duration

        if remaining[current_job] < 1e-10:
            completion_times[current_job] = next_event

        t = next_event

    obj = instance.total_completion(completion_times)
    return PreemptiveSMSolution(
        segments=segments,
        completion_times=completion_times,
        objective=obj,
    )


def wsrpt(instance: PreemptiveSMInstance) -> PreemptiveSMSolution:
    """Weighted SRPT heuristic for 1|pmtn,rj|ΣwjCj.

    At each time point, process the available job with highest wj/pj_remaining.

    Args:
        instance: A PreemptiveSMInstance.

    Returns:
        PreemptiveSMSolution.
    """
    n = instance.n
    remaining = instance.processing_times.copy()
    completion_times = [0.0] * n
    segments: list[tuple[int, float, float]] = []

    events = sorted(set(instance.release_dates))

    t = 0.0
    event_idx = 0

    while True:
        # Collect available jobs
        avail = []
        for j in range(n):
            if instance.release_dates[j] <= t + 1e-10 and remaining[j] > 1e-10:
                priority = instance.weights[j] / max(remaining[j], 1e-10)
                avail.append((priority, j))

        if not avail:
            if event_idx < len(events) and events[event_idx] > t + 1e-10:
                t = events[event_idx]
                event_idx += 1
                continue
            elif event_idx < len(events):
                event_idx += 1
                continue
            else:
                break

        avail.sort(reverse=True)
        current_job = avail[0][1]

        # Next event
        next_release = float("inf")
        for e in events:
            if e > t + 1e-10:
                next_release = e
                break

        completion_at = t + remaining[current_job]
        next_event = min(next_release, completion_at)

        duration = next_event - t
        segments.append((current_job, t, next_event))
        remaining[current_job] -= duration

        if remaining[current_job] < 1e-10:
            completion_times[current_job] = next_event

        t = next_event

    obj = instance.total_weighted_completion(completion_times)
    return PreemptiveSMSolution(
        segments=segments,
        completion_times=completion_times,
        objective=obj,
    )


if __name__ == "__main__":
    inst = _inst.small_preemptive_4()
    sol1 = srpt(inst)
    print(f"SRPT: ΣCj={sol1.objective:.1f}")
    sol2 = wsrpt(inst)
    print(f"WSRPT: ΣwjCj={sol2.objective:.1f}")

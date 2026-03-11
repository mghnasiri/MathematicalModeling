"""
Dispatching Heuristics for Job Shop with Weighted Tardiness.

Problem: Jm || ΣwjTj
Complexity: O(n * m * n) for Giffler-Thompson based dispatch

1. ATC (Apparent Tardiness Cost): composite priority combining
   processing time and due date urgency, adapted for JSP.
2. WSPT: weighted shortest processing time priority.

References:
    Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for
    job shops with weighted tardiness costs. Management Science,
    33(8), 1035-1047. https://doi.org/10.1287/mnsc.33.8.1035
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("wtjsp_instance_h", os.path.join(_this_dir, "instance.py"))
WTJSPInstance = _inst.WTJSPInstance
WTJSPSolution = _inst.WTJSPSolution
schedule_from_sequences = _inst.schedule_from_sequences


def _dispatch(instance: WTJSPInstance, priority_fn) -> WTJSPSolution:
    """Generic Giffler-Thompson active schedule builder.

    Args:
        instance: A WTJSPInstance.
        priority_fn: (job, op_idx, time) -> priority (lower is better).

    Returns:
        WTJSPSolution.
    """
    n, m = instance.n, instance.m
    next_op = [0] * n  # next operation index for each job
    machine_time = [0.0] * m
    job_time = [0.0] * n
    machine_sequences: list[list[tuple[int, int]]] = [[] for _ in range(m)]

    total_ops = sum(len(instance.operations[j]) for j in range(n))
    for _ in range(total_ops):
        # Find eligible operations
        candidates = []
        for j in range(n):
            if next_op[j] >= len(instance.operations[j]):
                continue
            op_idx = next_op[j]
            mach, dur = instance.operations[j][op_idx]
            start = max(machine_time[mach], job_time[j])
            candidates.append((j, op_idx, mach, dur, start))

        if not candidates:
            break

        # Select by priority
        best = min(candidates, key=lambda c: priority_fn(c[0], c[1], c[4]))
        j, op_idx, mach, dur, start = best

        machine_sequences[mach].append((j, op_idx))
        machine_time[mach] = start + dur
        job_time[j] = start + dur
        next_op[j] += 1

    ct, _ = schedule_from_sequences(instance, machine_sequences)
    wt = instance.weighted_tardiness(ct)
    return WTJSPSolution(
        machine_sequences=machine_sequences,
        completion_times=ct,
        weighted_tardiness=wt,
    )


def atc_dispatch(instance: WTJSPInstance, k_param: float = 2.0) -> WTJSPSolution:
    """ATC dispatching rule for Jm||ΣwjTj.

    Priority: wj/pj * exp(-max(0, dj - pj_remaining - t) / (k * p_avg))

    Args:
        instance: A WTJSPInstance.
        k_param: Lookahead parameter.

    Returns:
        WTJSPSolution.
    """
    p_avg = np.mean([d for j in range(instance.n)
                     for _, d in instance.operations[j]])

    remaining = [sum(d for _, d in instance.operations[j]) for j in range(instance.n)]

    def priority(j, op_idx, t):
        _, dur = instance.operations[j][op_idx]
        rem = sum(d for _, d in instance.operations[j][op_idx:])
        slack = max(0.0, instance.due_dates[j] - rem - t)
        urgency = np.exp(-slack / max(k_param * p_avg, 1e-10))
        return -(instance.weights[j] / max(dur, 1e-10)) * urgency

    return _dispatch(instance, priority)


def wspt_dispatch(instance: WTJSPInstance) -> WTJSPSolution:
    """WSPT dispatching: priority = pj/wj (lower is better).

    Args:
        instance: A WTJSPInstance.

    Returns:
        WTJSPSolution.
    """
    def priority(j, op_idx, t):
        _, dur = instance.operations[j][op_idx]
        return dur / max(instance.weights[j], 1e-10)

    return _dispatch(instance, priority)


if __name__ == "__main__":
    inst = _inst.small_wtjsp_3_3()
    sol1 = atc_dispatch(inst)
    print(f"ATC: wt={sol1.weighted_tardiness:.1f}")
    sol2 = wspt_dispatch(inst)
    print(f"WSPT: wt={sol2.weighted_tardiness:.1f}")

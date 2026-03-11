"""
Variable Neighborhood Search for RCPSP.

Problem notation: PS | prec | Cmax

VNS uses multiple neighborhood structures on precedence-feasible
activity lists:
    N1: Adjacent swap — swap two adjacent activities (if precedence-feasible)
    N2: Shift — move an activity to another precedence-feasible position
    N3: Multi-swap — swap k random pairs simultaneously

Local search uses best-improvement adjacent swap with Serial SGS decoder.
Warm-started with best SGS priority rule.

Complexity: O(iterations * k_max * n^2 * K * T) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Kolisch, R. & Hartmann, S. (2006). Experimental investigation of
    heuristics for resource-constrained project scheduling: An update.
    European Journal of Operational Research, 174(1), 23-37.
    https://doi.org/10.1016/j.ejor.2005.01.065
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("rcpsp_instance_vns", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution

_sgs = _load_mod(
    "rcpsp_sgs_vns",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def vns(
    instance: RCPSPInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> RCPSPSolution:
    """Solve RCPSP using Variable Neighborhood Search.

    Args:
        instance: An RCPSPInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        RCPSPSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    total = n + 2
    start_time = time.time()

    # Build transitive predecessor/successor sets for feasibility checks
    all_preds = {}
    all_succs = {}
    for a in range(total):
        all_preds[a] = _transitive(instance.predecessors, a)
        all_succs[a] = _transitive(instance.successors, a)

    # Warm-start: try all rules
    best_sol = None
    best_list = None
    for rule in ["lft", "est", "mts", "grpw"]:
        sol = serial_sgs(instance, priority_rule=rule)
        if best_sol is None or sol.makespan < best_sol.makespan:
            best_sol = sol
            activities = list(range(total))
            activities.sort(key=lambda a: (sol.start_times[a], a))
            best_list = activities

    current_list = best_list[:]
    current_ms = best_sol.makespan
    best_ms = current_ms
    best_result = best_sol

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = current_list[:]
            _shake(instance, shaken, k, all_succs, rng)

            # Local search
            shaken_sol = _best_swap_ls(instance, shaken, all_succs)
            shaken_ms = shaken_sol.makespan

            if shaken_ms < current_ms:
                current_list = shaken
                current_ms = shaken_ms
                k = 1

                if current_ms < best_ms:
                    best_ms = current_ms
                    best_result = shaken_sol
                    best_list = current_list[:]
            else:
                k += 1

    return best_result


def _transitive(graph: dict, node: int) -> set[int]:
    """Compute transitive closure from a node."""
    visited = set()
    stack = list(graph.get(node, []))
    while stack:
        n = stack.pop()
        if n not in visited:
            visited.add(n)
            stack.extend(graph.get(n, []))
    return visited


def _can_swap(
    instance: RCPSPInstance,
    plist: list[int],
    i: int,
    j: int,
    all_succs: dict[int, set[int]],
) -> bool:
    """Check if swapping plist[i] and plist[j] maintains precedence."""
    a, b = plist[i], plist[j]
    if i < j:
        # a before b currently; after swap b before a
        # Invalid if a must come before b (a is predecessor of b)
        return b not in all_succs[a]
    else:
        return a not in all_succs[b]


def _shake(
    instance: RCPSPInstance,
    plist: list[int],
    k: int,
    all_succs: dict[int, set[int]],
    rng: np.random.Generator,
) -> None:
    """Shake: perform k random precedence-feasible swaps."""
    n = len(plist)
    for _ in range(k * 3):  # Try multiple times
        i = rng.integers(1, n - 1)  # Skip dummy activities at ends
        j = i + 1
        if j >= n - 1:
            continue
        if _can_swap(instance, plist, i, j, all_succs):
            plist[i], plist[j] = plist[j], plist[i]


def _best_swap_ls(
    instance: RCPSPInstance,
    plist: list[int],
    all_succs: dict[int, set[int]],
) -> RCPSPSolution:
    """Best-improvement adjacent swap local search."""
    improved = True
    current_sol = serial_sgs(instance, priority_list=plist)
    current_ms = current_sol.makespan

    while improved:
        improved = False
        best_delta = 0
        best_idx = -1

        for i in range(1, len(plist) - 2):  # Skip dummies
            if not _can_swap(instance, plist, i, i + 1, all_succs):
                continue
            plist[i], plist[i + 1] = plist[i + 1], plist[i]
            sol = serial_sgs(instance, priority_list=plist)
            delta = sol.makespan - current_ms
            if delta < best_delta:
                best_delta = delta
                best_idx = i
            plist[i], plist[i + 1] = plist[i + 1], plist[i]

        if best_idx >= 0:
            plist[best_idx], plist[best_idx + 1] = plist[best_idx + 1], plist[best_idx]
            current_sol = serial_sgs(instance, priority_list=plist)
            current_ms = current_sol.makespan
            improved = True

    return current_sol


if __name__ == "__main__":
    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"RCPSP: {inst.n} activities, {inst.num_resources} resources")
    print(f"Critical path: {inst.critical_path_length()}")

    sol = vns(inst, seed=42)
    print(f"VNS: makespan={sol.makespan}")

    sgs_sol = serial_sgs(inst, priority_rule="lft")
    print(f"SGS (LFT): makespan={sgs_sol.makespan}")

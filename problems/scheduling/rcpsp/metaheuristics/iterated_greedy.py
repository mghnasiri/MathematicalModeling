"""
Iterated Greedy for RCPSP.

Problem notation: PS | prec | Cmax

Iterated Greedy repeatedly removes a subset of activities from the schedule
and reinserts them using the Serial SGS decoder. A Boltzmann acceptance
criterion allows escaping local optima.

Warm-started with best SGS priority rule (LFT/EST/MTS/GRPW).

Complexity: O(iterations * d * n^2 * K * T) where d = destruction size.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Valls, V., Ballestín, F. & Quintanilla, S. (2008). A hybrid genetic
    algorithm for the resource-constrained project scheduling problem.
    European Journal of Operational Research, 185(2), 495-508.
    https://doi.org/10.1016/j.ejor.2006.12.033
"""

from __future__ import annotations

import sys
import os
import math
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


_inst = _load_mod("rcpsp_instance_ig", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution

_sgs = _load_mod(
    "rcpsp_sgs_ig",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def iterated_greedy(
    instance: RCPSPInstance,
    max_iterations: int = 2000,
    d: int | None = None,
    temperature_factor: float = 0.3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> RCPSPSolution:
    """Solve RCPSP using Iterated Greedy.

    Args:
        instance: An RCPSPInstance.
        max_iterations: Maximum number of iterations.
        d: Number of activities to remove. Default: max(2, n//4).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        RCPSPSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    total = n + 2
    start_time = time.time()

    if d is None:
        d = max(2, n // 4)
    d = min(d, n)

    # Warm-start: try all rules, pick best
    best_sol = None
    best_list = None
    for rule in ["lft", "est", "mts", "grpw"]:
        sol = serial_sgs(instance, priority_rule=rule)
        if best_sol is None or sol.makespan < best_sol.makespan:
            best_sol = sol
            # Extract priority list from solution
            activities = list(range(total))
            activities.sort(key=lambda a: (sol.start_times[a], a))
            best_list = activities

    current_list = best_list[:]
    current_ms = best_sol.makespan
    best_ms = current_ms
    best_result = best_sol

    # Temperature
    avg_dur = float(np.mean(instance.durations[1:n + 1])) if n > 0 else 1.0
    temperature = temperature_factor * avg_dur

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: remove d random non-dummy activities from the list
        non_dummy = [a for a in current_list if 0 < a < n + 1]
        if len(non_dummy) <= d:
            removed = non_dummy[:]
        else:
            indices = rng.choice(len(non_dummy), size=d, replace=False)
            removed = [non_dummy[i] for i in indices]

        removed_set = set(removed)
        partial = [a for a in current_list if a not in removed_set]

        # Sort removed activities in topological order so predecessors
        # are reinserted before their successors
        removed = _topo_sort(instance, removed)

        # Reconstruction: greedily reinsert each removed activity
        # in the position that yields the best makespan
        for act in removed:
            best_pos = _find_best_insertion(instance, partial, act)
            partial.insert(best_pos, act)

        sol = serial_sgs(instance, priority_list=partial)
        new_ms = sol.makespan

        # Acceptance
        delta = new_ms - current_ms
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / max(temperature, 1e-10))):
            current_list = partial
            current_ms = new_ms

            if current_ms < best_ms:
                best_ms = current_ms
                best_result = sol
                best_list = current_list[:]

    return best_result


def _all_ancestors(instance: RCPSPInstance, act: int) -> set[int]:
    """Return all transitive predecessors of act."""
    visited = set()
    stack = list(instance.predecessors.get(act, []))
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(instance.predecessors.get(node, []))
    return visited


def _all_descendants(instance: RCPSPInstance, act: int) -> set[int]:
    """Return all transitive successors of act."""
    visited = set()
    stack = list(instance.successors.get(act, []))
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(instance.successors.get(node, []))
    return visited


def _topo_sort(instance: RCPSPInstance, activities: list[int]) -> list[int]:
    """Sort activities in topological order respecting precedence."""
    act_set = set(activities)
    in_deg = {}
    for a in activities:
        in_deg[a] = 0
        for p in instance.predecessors.get(a, []):
            if p in act_set:
                in_deg[a] += 1

    queue = [a for a in activities if in_deg[a] == 0]
    result = []
    while queue:
        a = queue.pop(0)
        result.append(a)
        for s in instance.successors.get(a, []):
            if s in act_set:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    queue.append(s)
    # If cycle or missing, append remaining
    for a in activities:
        if a not in set(result):
            result.append(a)
    return result


def _find_best_insertion(
    instance: RCPSPInstance,
    plist: list[int],
    act: int,
) -> int:
    """Find the best precedence-feasible position to insert activity."""
    pos_map = {a: i for i, a in enumerate(plist)}

    # Use transitive predecessors/successors to find valid range
    all_preds = _all_ancestors(instance, act)
    all_succs = _all_descendants(instance, act)

    earliest = 0
    for p in all_preds:
        if p in pos_map:
            earliest = max(earliest, pos_map[p] + 1)

    latest = len(plist)
    for s in all_succs:
        if s in pos_map:
            latest = min(latest, pos_map[s])

    if latest < earliest:
        latest = earliest

    # Try a few positions and pick the best
    best_pos = earliest
    best_ms = float("inf")

    # Sample positions if range is large
    positions = list(range(earliest, latest + 1))
    if len(positions) > 10:
        sampled = [earliest, latest]
        sampled.extend(
            [positions[i] for i in np.linspace(0, len(positions) - 1, 8, dtype=int)]
        )
        positions = sorted(set(sampled))

    for pos in positions:
        trial = plist[:pos] + [act] + plist[pos:]
        sol = serial_sgs(instance, priority_list=trial)
        if sol.makespan < best_ms:
            best_ms = sol.makespan
            best_pos = pos

    return best_pos


if __name__ == "__main__":
    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"RCPSP: {inst.n} activities, {inst.num_resources} resources")
    print(f"Critical path: {inst.critical_path_length()}")

    sol = iterated_greedy(inst, seed=42)
    print(f"IG: makespan={sol.makespan}")

    sgs_sol = serial_sgs(inst, priority_rule="lft")
    print(f"SGS (LFT): makespan={sgs_sol.makespan}")

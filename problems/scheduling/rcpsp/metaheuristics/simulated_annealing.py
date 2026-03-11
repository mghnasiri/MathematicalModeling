"""
Simulated Annealing for Resource-Constrained Project Scheduling (RCPSP).

Problem: PS | prec | Cmax

Representation: A precedence-feasible activity list (permutation of
activities 0..n+1). Decoded into a schedule using the Serial SGS.

Neighborhoods:
- Swap: swap two adjacent non-dummy activities if precedence allows
- Shift: move an activity to a different precedence-feasible position

Warm-started with LFT priority rule via Serial SGS.

Complexity: O(iterations * n^2 * K) per run.

References:
    Bouleimen, K. & Lecocq, H. (2003). A new efficient simulated
    annealing algorithm for the resource-constrained project scheduling
    problem and its multiple mode version. European Journal of
    Operational Research, 149(2), 268-281.
    https://doi.org/10.1016/S0377-2217(02)00761-0

    Kolisch, R. & Hartmann, S. (2006). Experimental investigation of
    heuristics for resource-constrained project scheduling. European
    Journal of Operational Research, 169(1), 16-37.
    https://doi.org/10.1016/j.ejor.2004.01.035
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("rcpsp_instance_sa", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution
validate_solution = _inst.validate_solution

_sgs = _load_mod(
    "rcpsp_serial_sgs_sa",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def _is_precedence_feasible(
    instance: RCPSPInstance,
    activity_list: list[int],
) -> bool:
    """Check if an activity list respects all precedence constraints."""
    pos = {act: i for i, act in enumerate(activity_list)}
    total = instance.n + 2
    for act in range(total):
        for succ in instance.successors.get(act, []):
            if pos.get(act, 0) >= pos.get(succ, 0):
                return False
    return True


def _can_swap(
    instance: RCPSPInstance,
    activity_list: list[int],
    i: int,
    j: int,
) -> bool:
    """Check if swapping positions i and j maintains precedence feasibility."""
    act_i = activity_list[i]
    act_j = activity_list[j]

    # act_j would be at position i, act_i at position j (i < j)
    if i > j:
        i, j = j, i
        act_i, act_j = act_j, act_i

    # act_j moves to earlier position i: check act_j's predecessors are before i
    for pred in instance.predecessors.get(act_j, []):
        if pred == act_i:
            return False  # act_i is a predecessor of act_j
        # pred must appear before position i in the modified list
        # Since we're only swapping i and j, pred's position is unchanged
        # unless pred == act_i (handled above)

    # act_i moves to later position j: check act_i's successors are after j
    for succ in instance.successors.get(act_i, []):
        if succ == act_j:
            return False  # act_j is a successor of act_i
        # succ position unchanged, must be after j

    # More thorough check: rebuild position map
    test = list(activity_list)
    test[i], test[j] = test[j], test[i]
    pos = {act: idx for idx, act in enumerate(test)}
    for pred in instance.predecessors.get(act_j, []):
        if pos.get(pred, 0) >= i:
            return False
    for succ in instance.successors.get(act_i, []):
        if pos.get(succ, len(test)) <= j:
            return False

    return True


def simulated_annealing(
    instance: RCPSPInstance,
    max_iterations: int = 5000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> RCPSPSolution:
    """Solve RCPSP using Simulated Annealing with Serial SGS decoder.

    Args:
        instance: RCPSP instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best RCPSPSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    n = instance.n
    total = n + 2

    # Initialize with LFT-based Serial SGS
    init_sol = serial_sgs(instance, priority_rule="lft")
    best_makespan = init_sol.makespan
    best_start_times = init_sol.start_times.copy()

    # Build initial activity list from LFT ordering
    activity_list = instance.topological_order()
    current_sol = serial_sgs(instance, priority_list=activity_list)
    current_makespan = current_sol.makespan

    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_start_times = current_sol.start_times.copy()

    if initial_temp is None:
        initial_temp = max(1.0, n * 0.5)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if n < 2:
            break

        new_list = list(activity_list)

        # Choose a random non-dummy activity and try to move it
        if rng.random() < 0.6:
            # Swap two adjacent non-dummy activities
            # Pick a random position in the non-dummy range
            attempts = 0
            swapped = False
            while attempts < 10 and not swapped:
                i = rng.integers(1, total - 1)
                j = rng.integers(1, total - 1)
                if i != j and _can_swap(instance, new_list, i, j):
                    new_list[i], new_list[j] = new_list[j], new_list[i]
                    swapped = True
                attempts += 1
            if not swapped:
                temp *= cooling_rate
                continue
        else:
            # Shift: remove a non-dummy activity and reinsert at valid position
            idx = rng.integers(1, total - 1)
            act = new_list.pop(idx)

            # Find valid range for reinsertion
            # Must be after all predecessors and before all successors
            pos_map = {a: p for p, a in enumerate(new_list)}
            earliest = 0
            for pred in instance.predecessors.get(act, []):
                if pred in pos_map:
                    earliest = max(earliest, pos_map[pred] + 1)
            latest = len(new_list)
            for succ in instance.successors.get(act, []):
                if succ in pos_map:
                    latest = min(latest, pos_map[succ])

            if earliest > latest:
                # Can't reinsert; restore
                new_list.insert(idx, act)
                temp *= cooling_rate
                continue

            new_pos = rng.integers(earliest, latest + 1)
            new_list.insert(new_pos, act)

        # Decode and evaluate
        new_sol = serial_sgs(instance, priority_list=new_list)
        new_makespan = new_sol.makespan

        delta = new_makespan - current_makespan
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            activity_list = new_list
            current_makespan = new_makespan

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_start_times = new_sol.start_times.copy()

        temp *= cooling_rate

    return RCPSPSolution(start_times=best_start_times, makespan=best_makespan)


if __name__ == "__main__":
    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"Critical path LB: {inst.critical_path_length()}")

    sol_lft = serial_sgs(inst, priority_rule="lft")
    print(f"LFT SGS: makespan = {sol_lft.makespan}")

    sol_sa = simulated_annealing(inst, seed=42)
    print(f"SA:       makespan = {sol_sa.makespan}")

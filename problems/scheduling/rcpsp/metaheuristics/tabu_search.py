"""
Tabu Search for Resource-Constrained Project Scheduling (RCPSP).

Problem: PS | prec | Cmax

Representation: A precedence-feasible activity list decoded using the
Serial SGS into a schedule.

Neighborhoods:
- Swap: swap two non-dummy activities if precedence feasibility is maintained
- Shift: move a non-dummy activity to a different precedence-feasible position

Uses short-term memory preventing recently moved activities from being
moved again. Aspiration criterion overrides tabu when a global
improvement is found.

Warm-started with LFT priority rule via Serial SGS.

Complexity: O(iterations * n^2 * K) per run.

References:
    Nonobe, K. & Ibaraki, T. (2002). Formulation and tabu search
    algorithm for the resource constrained project scheduling problem.
    In: Ribeiro, C.C. & Hansen, P. (eds) Essays and Surveys in
    Metaheuristics, Springer, 557-588.
    https://doi.org/10.1007/978-1-4615-1507-4_25

    Kolisch, R. & Hartmann, S. (2006). Experimental investigation of
    heuristics for resource-constrained project scheduling. European
    Journal of Operational Research, 169(1), 16-37.
    https://doi.org/10.1016/j.ejor.2004.01.035
"""

from __future__ import annotations

import os
import sys
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


_inst = _load_mod("rcpsp_instance_ts", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution
validate_solution = _inst.validate_solution

_sgs = _load_mod(
    "rcpsp_serial_sgs_ts",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def _can_swap(
    instance: RCPSPInstance,
    activity_list: list[int],
    i: int,
    j: int,
) -> bool:
    """Check if swapping positions i and j maintains precedence feasibility."""
    if i > j:
        i, j = j, i

    act_i = activity_list[i]
    act_j = activity_list[j]

    # Check if act_i is predecessor of act_j or vice versa
    if act_j in instance.successors.get(act_i, []):
        return False
    if act_i in instance.successors.get(act_j, []):
        return False

    # Check all predecessors of act_j appear before position i
    test = list(activity_list)
    test[i], test[j] = test[j], test[i]
    pos = {a: p for p, a in enumerate(test)}

    for pred in instance.predecessors.get(act_j, []):
        if pos.get(pred, 0) >= pos[act_j]:
            return False
    for succ in instance.successors.get(act_j, []):
        if pos.get(succ, len(test)) <= pos[act_j]:
            return False
    for pred in instance.predecessors.get(act_i, []):
        if pos.get(pred, 0) >= pos[act_i]:
            return False
    for succ in instance.successors.get(act_i, []):
        if pos.get(succ, len(test)) <= pos[act_i]:
            return False

    return True


def tabu_search(
    instance: RCPSPInstance,
    max_iterations: int = 3000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> RCPSPSolution:
    """Solve RCPSP using Tabu Search with Serial SGS decoder.

    Args:
        instance: RCPSP instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best RCPSPSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    n = instance.n
    total = n + 2

    if tabu_tenure is None:
        tabu_tenure = max(3, int(n ** 0.5))

    # Initialize with LFT-based Serial SGS
    init_sol = serial_sgs(instance, priority_rule="lft")
    best_makespan = init_sol.makespan
    best_start_times = init_sol.start_times.copy()

    activity_list = instance.topological_order()
    current_sol = serial_sgs(instance, priority_list=activity_list)
    current_makespan = current_sol.makespan

    if current_makespan < best_makespan:
        best_makespan = current_makespan
        best_start_times = current_sol.start_times.copy()

    # Tabu list: activity -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        if n < 2:
            break

        best_delta = float("inf")
        best_move = None

        # Sample candidate moves
        n_candidates = min(n * 5, n * n)
        for _ in range(n_candidates):
            if rng.random() < 0.6:
                # Swap move
                i = rng.integers(1, total - 1)
                j = rng.integers(1, total - 1)
                if i == j:
                    continue

                act_i = activity_list[i]
                act_j = activity_list[j]

                is_tabu = (
                    (act_i in tabu_dict and tabu_dict[act_i] > iteration)
                    or (act_j in tabu_dict and tabu_dict[act_j] > iteration)
                )

                if not _can_swap(instance, activity_list, i, j):
                    continue

                new_list = list(activity_list)
                new_list[i], new_list[j] = new_list[j], new_list[i]
                new_sol = serial_sgs(instance, priority_list=new_list)
                delta = new_sol.makespan - current_makespan

                if is_tabu and current_makespan + delta >= best_makespan:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i, j, new_list, new_sol)
            else:
                # Shift move
                idx = rng.integers(1, total - 1)
                act = activity_list[idx]

                is_tabu = (
                    act in tabu_dict and tabu_dict[act] > iteration
                )

                new_list = list(activity_list)
                new_list.pop(idx)

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
                    continue

                new_pos = rng.integers(earliest, latest + 1)
                new_list.insert(new_pos, act)

                new_sol = serial_sgs(instance, priority_list=new_list)
                delta = new_sol.makespan - current_makespan

                if is_tabu and current_makespan + delta >= best_makespan:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("shift", idx, act, new_list, new_sol)

        if best_move is None:
            tabu_dict.clear()
            continue

        # Apply move
        if best_move[0] == "swap":
            _, i, j, new_list, new_sol = best_move
            tabu_dict[activity_list[i]] = iteration + tabu_tenure
            tabu_dict[activity_list[j]] = iteration + tabu_tenure
        else:  # shift
            _, idx, act, new_list, new_sol = best_move
            tabu_dict[act] = iteration + tabu_tenure

        activity_list = new_list
        current_makespan = new_sol.makespan

        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_start_times = new_sol.start_times.copy()

    return RCPSPSolution(start_times=best_start_times, makespan=best_makespan)


if __name__ == "__main__":
    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"Critical path LB: {inst.critical_path_length()}")

    sol_lft = serial_sgs(inst, priority_rule="lft")
    print(f"LFT SGS: makespan = {sol_lft.makespan}")

    sol_ts = tabu_search(inst, seed=42)
    print(f"TS:      makespan = {sol_ts.makespan}")

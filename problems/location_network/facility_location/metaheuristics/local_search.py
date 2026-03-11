"""
Local Search for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)

Neighborhoods:
- Add: open a currently closed facility
- Drop: close a currently open facility (keeping >= 1 open)
- Swap: close one facility and open another simultaneously

Uses best-improvement search with random restarts.
Warm-started with greedy-add heuristic.

Complexity: O(iterations * m^2 * n) per run.

References:
    Cornuéjols, G., Fisher, M.L. & Nemhauser, G.L. (1977). Location of
    bank accounts to optimize float: An analytic study of exact and
    approximate algorithms. Management Science, 23(8), 789-810.
    https://doi.org/10.1287/mnsc.23.8.789

    Ghosh, D. (2003). Neighborhood search heuristics for the
    uncapacitated facility location problem. European Journal of
    Operational Research, 150(1), 150-162.
    https://doi.org/10.1016/S0377-2217(02)00504-6
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


_inst = _load_mod("fl_instance_ls", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution


def _assign_and_cost(
    instance: FacilityLocationInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers to nearest open facility and compute total cost."""
    assignments = []
    total = sum(instance.fixed_costs[i] for i in open_set)
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best)
        total += instance.assignment_costs[best][j]
    return assignments, total


def local_search(
    instance: FacilityLocationInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using local search with add/drop/swap neighborhoods.

    Args:
        instance: A FacilityLocationInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        FacilityLocationSolution with the best solution found.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    start_time = time.time()

    # Warm-start with greedy-add
    _gr = _load_mod(
        "fl_gr_ls",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    init_sol = _gr.greedy_add(instance)
    open_set = set(init_sol.open_facilities)
    _, current_cost = _assign_and_cost(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        improved = False
        best_delta = 0.0
        best_move = None

        closed = set(range(m)) - open_set

        # Add moves
        for i in closed:
            trial = open_set | {i}
            _, cost = _assign_and_cost(instance, trial)
            delta = cost - current_cost
            if delta < best_delta - 1e-10:
                best_delta = delta
                best_move = ("add", i)

        # Drop moves
        if len(open_set) > 1:
            for i in list(open_set):
                trial = open_set - {i}
                _, cost = _assign_and_cost(instance, trial)
                delta = cost - current_cost
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_move = ("drop", i)

        # Swap moves
        for i_open in list(open_set):
            for i_closed in closed:
                trial = (open_set - {i_open}) | {i_closed}
                _, cost = _assign_and_cost(instance, trial)
                delta = cost - current_cost
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_move = ("swap", i_open, i_closed)

        if best_move is not None:
            if best_move[0] == "add":
                open_set.add(best_move[1])
            elif best_move[0] == "drop":
                open_set.remove(best_move[1])
            else:
                open_set.remove(best_move[1])
                open_set.add(best_move[2])

            current_cost += best_delta
            improved = True

            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_open = set(open_set)
        else:
            # Random perturbation to escape local optimum
            closed = set(range(m)) - open_set
            if closed:
                to_add = rng.choice(list(closed))
                open_set.add(to_add)
            if len(open_set) > 1:
                to_drop = rng.choice(list(open_set))
                open_set.remove(to_drop)
            _, current_cost = _assign_and_cost(instance, open_set)

    assignments, final_cost = _assign_and_cost(instance, best_open)
    return FacilityLocationSolution(
        open_facilities=sorted(best_open),
        assignments=assignments,
        cost=final_cost,
    )


if __name__ == "__main__":
    inst = FacilityLocationInstance.random(m=8, n=15, seed=42)
    print(f"UFLP: {inst.m} facilities, {inst.n} customers")

    _gr = _load_mod(
        "fl_gr_ls_main",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    gr_sol = _gr.greedy_add(inst)
    print(f"Greedy Add: cost={gr_sol.cost:.1f}")

    ls_sol = local_search(inst, seed=42)
    print(f"LS: cost={ls_sol.cost:.1f}")

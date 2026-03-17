"""
Tabu Search for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)

Neighborhoods:
- Toggle: open a closed facility or close an open facility
- Swap: close one facility and open another

Uses short-term memory preventing recently toggled facilities from
being toggled again. Aspiration criterion overrides tabu when a
move yields a new global best.

Warm-started with greedy add heuristic.

Complexity: O(iterations * m * n) per run.

References:
    Sun, M. (2006). Solving the uncapacitated facility location
    problem using tabu search. Computers & Operations Research,
    33(9), 2563-2589.
    https://doi.org/10.1016/j.cor.2005.07.014

    Ghosh, D. (2003). Neighborhood search heuristics for the
    uncapacitated facility location problem. European Journal of
    Operational Research, 150(1), 150-162.
    https://doi.org/10.1016/S0377-2217(02)00504-4
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


_inst = _load_mod("fl_instance_ts", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution

_greedy = _load_mod(
    "fl_greedy_ts",
    os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
)
greedy_add = _greedy.greedy_add


def _evaluate(
    instance: FacilityLocationInstance,
    open_set: set[int],
) -> tuple[float, list[int]]:
    """Evaluate an open facility set."""
    if not open_set:
        return float("inf"), [0] * instance.n

    assignments = []
    assign_cost = 0.0
    for j in range(instance.n):
        best_fac = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best_fac)
        assign_cost += instance.assignment_costs[best_fac][j]

    fixed_cost = sum(instance.fixed_costs[i] for i in open_set)
    return fixed_cost + assign_cost, assignments


def tabu_search(
    instance: FacilityLocationInstance,
    max_iterations: int = 2000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using Tabu Search.

    Args:
        instance: Facility location instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(m).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best FacilityLocationSolution found.
    """
    rng = np.random.default_rng(seed)
    m = instance.m
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(3, int(m ** 0.5))

    # Initialize with greedy add
    init_sol = greedy_add(instance)
    open_set = set(init_sol.open_facilities)
    current_cost, current_assign = _evaluate(instance, open_set)

    best_open = set(open_set)
    best_cost = current_cost
    best_assign = list(current_assign)

    # Tabu list: facility -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_delta = float("inf")
        best_move = None

        # ── Toggle neighborhood ──────────────────────────────────────────
        for i in range(m):
            is_tabu = (
                i in tabu_dict and tabu_dict[i] > iteration
            )

            if i in open_set:
                # Try closing facility i (must keep at least 1 open)
                if len(open_set) <= 1:
                    continue
                test_set = open_set - {i}
            else:
                # Try opening facility i
                test_set = open_set | {i}

            new_cost, _ = _evaluate(instance, test_set)
            delta = new_cost - current_cost

            if is_tabu and current_cost + delta >= best_cost:
                continue

            if delta < best_delta:
                best_delta = delta
                best_move = ("toggle", i)

        # ── Swap neighborhood ────────────────────────────────────────────
        closed = [i for i in range(m) if i not in open_set]
        for i_open in list(open_set):
            for i_closed in closed:
                is_tabu = (
                    (i_open in tabu_dict and tabu_dict[i_open] > iteration)
                    or (i_closed in tabu_dict and tabu_dict[i_closed] > iteration)
                )

                test_set = (open_set - {i_open}) | {i_closed}
                new_cost, _ = _evaluate(instance, test_set)
                delta = new_cost - current_cost

                if is_tabu and current_cost + delta >= best_cost:
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", i_open, i_closed)

        if best_move is None:
            tabu_dict.clear()
            continue

        # ── Apply move ───────────────────────────────────────────────────
        if best_move[0] == "toggle":
            _, fac = best_move
            if fac in open_set:
                open_set.remove(fac)
            else:
                open_set.add(fac)
            tabu_dict[fac] = iteration + tabu_tenure

        elif best_move[0] == "swap":
            _, i_open, i_closed = best_move
            open_set.remove(i_open)
            open_set.add(i_closed)
            tabu_dict[i_open] = iteration + tabu_tenure
            tabu_dict[i_closed] = iteration + tabu_tenure

        current_cost, current_assign = _evaluate(instance, open_set)

        if current_cost < best_cost:
            best_cost = current_cost
            best_open = set(open_set)
            best_assign = list(current_assign)

    return FacilityLocationSolution(
        open_facilities=sorted(best_open),
        assignments=best_assign,
        cost=best_cost,
    )


if __name__ == "__main__":
    from instance import small_uflp_3_5, medium_uflp_5_10

    print("=== TS for Facility Location ===\n")

    inst = small_uflp_3_5()
    sol = tabu_search(inst, seed=42)
    print(f"small_3_5: cost={sol.cost:.1f}, open={sol.open_facilities}")

    inst2 = medium_uflp_5_10()
    sol2 = tabu_search(inst2, seed=42)
    print(f"medium_5_10: cost={sol2.cost:.1f}, open={sol2.open_facilities}")

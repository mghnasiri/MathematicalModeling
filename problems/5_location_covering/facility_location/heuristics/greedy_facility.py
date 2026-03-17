"""
Greedy Heuristics for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)
Complexity: O(m * n * m) = O(m^2 * n)

Two greedy approaches:
- Greedy Add: Start with no facilities open. Iteratively open the facility
  that gives the largest cost reduction (fixed cost vs assignment savings).
- Greedy Drop: Start with all facilities open. Iteratively close the facility
  whose removal causes the smallest cost increase.

References:
    Cornuéjols, G., Fisher, M.L. & Nemhauser, G.L. (1977). Location of
    bank accounts to optimize float: An analytic study of exact and
    approximate algorithms. Management Science, 23(8), 789-810.
    https://doi.org/10.1287/mnsc.23.8.789

    Krarup, J. & Pruzan, P.M. (1983). The simple plant location
    problem: Survey and synthesis. European Journal of Operational
    Research, 12(1), 36-81.
    https://doi.org/10.1016/0377-2217(83)90181-9
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("fl_instance_gr", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution


def _assign_customers(
    instance: FacilityLocationInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign each customer to its nearest open facility.

    Returns:
        Tuple of (assignments, total_assignment_cost).
    """
    assignments = []
    total = 0.0
    for j in range(instance.n):
        best_fac = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best_fac)
        total += instance.assignment_costs[best_fac][j]
    return assignments, total


def greedy_add(instance: FacilityLocationInstance) -> FacilityLocationSolution:
    """Solve UFLP using greedy facility addition.

    Start with no facilities open. At each step, open the facility that
    maximizes cost reduction (assignment savings minus fixed cost).

    Args:
        instance: A FacilityLocationInstance.

    Returns:
        FacilityLocationSolution.
    """
    m, n = instance.m, instance.n
    open_set: set[int] = set()
    closed = set(range(m))

    # Must open at least one facility
    # Find best single facility
    best_fac = -1
    best_cost = float("inf")
    for i in range(m):
        cost = instance.fixed_costs[i] + sum(
            instance.assignment_costs[i][j] for j in range(n)
        )
        if cost < best_cost:
            best_cost = cost
            best_fac = i

    open_set.add(best_fac)
    closed.remove(best_fac)

    improved = True
    while improved and closed:
        improved = False
        assignments, assign_cost = _assign_customers(instance, open_set)
        current_cost = (
            sum(instance.fixed_costs[i] for i in open_set) + assign_cost
        )

        best_saving = 0.0
        best_candidate = -1

        for candidate in closed:
            # Cost with candidate added
            trial_open = open_set | {candidate}
            _, new_assign_cost = _assign_customers(instance, trial_open)
            new_cost = (
                sum(instance.fixed_costs[i] for i in trial_open)
                + new_assign_cost
            )
            saving = current_cost - new_cost
            if saving > best_saving:
                best_saving = saving
                best_candidate = candidate

        if best_candidate >= 0:
            open_set.add(best_candidate)
            closed.remove(best_candidate)
            improved = True

    assignments, assign_cost = _assign_customers(instance, open_set)
    total_cost = sum(instance.fixed_costs[i] for i in open_set) + assign_cost

    return FacilityLocationSolution(
        open_facilities=sorted(open_set),
        assignments=assignments,
        cost=total_cost,
    )


def greedy_drop(instance: FacilityLocationInstance) -> FacilityLocationSolution:
    """Solve UFLP using greedy facility removal.

    Start with all facilities open. At each step, close the facility
    whose removal causes the smallest cost increase (or largest decrease).

    Args:
        instance: A FacilityLocationInstance.

    Returns:
        FacilityLocationSolution.
    """
    m, n = instance.m, instance.n
    open_set = set(range(m))

    improved = True
    while improved and len(open_set) > 1:
        improved = False
        assignments, assign_cost = _assign_customers(instance, open_set)
        current_cost = (
            sum(instance.fixed_costs[i] for i in open_set) + assign_cost
        )

        best_saving = 0.0
        best_drop = -1

        for candidate in list(open_set):
            trial_open = open_set - {candidate}
            _, new_assign_cost = _assign_customers(instance, trial_open)
            new_cost = (
                sum(instance.fixed_costs[i] for i in trial_open)
                + new_assign_cost
            )
            saving = current_cost - new_cost
            if saving > best_saving:
                best_saving = saving
                best_drop = candidate

        if best_drop >= 0:
            open_set.remove(best_drop)
            improved = True

    assignments, assign_cost = _assign_customers(instance, open_set)
    total_cost = sum(instance.fixed_costs[i] for i in open_set) + assign_cost

    return FacilityLocationSolution(
        open_facilities=sorted(open_set),
        assignments=assignments,
        cost=total_cost,
    )


if __name__ == "__main__":
    from instance import small_uflp_3_5, medium_uflp_5_10

    print("=== Greedy Heuristics for UFLP ===\n")

    for name, inst_fn in [
        ("small_3_5", small_uflp_3_5),
        ("medium_5_10", medium_uflp_5_10),
    ]:
        inst = inst_fn()
        add_sol = greedy_add(inst)
        drop_sol = greedy_drop(inst)
        print(f"{name}: add={add_sol.cost:.1f} (open {add_sol.open_facilities}), "
              f"drop={drop_sol.cost:.1f} (open {drop_sol.open_facilities})")

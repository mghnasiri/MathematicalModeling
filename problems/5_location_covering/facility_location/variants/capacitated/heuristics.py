"""
Greedy Heuristic for Capacitated Facility Location (CFLP).

Problem: CFLP
Complexity: O(m^2 * n)

Greedy-add with capacity awareness: at each step, open the facility
that maximizes cost reduction while respecting capacity constraints.
Customers are assigned to the nearest open facility with sufficient
remaining capacity.

References:
    Cornuéjols, G., Sridharan, R. & Thizy, J.M. (1991). A comparison
    of heuristics and relaxations for the capacitated plant location
    problem. European Journal of Operational Research, 50(3), 280-297.
    https://doi.org/10.1016/0377-2217(91)90261-S
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


_inst = _load_mod("cflp_instance_h", os.path.join(_this_dir, "instance.py"))
CFLPInstance = _inst.CFLPInstance
CFLPSolution = _inst.CFLPSolution


def _assign_capacitated(
    instance: CFLPInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers to nearest open facility respecting capacity.

    Uses a greedy assignment: process customers by decreasing demand,
    assign to nearest facility with remaining capacity.
    """
    n, m = instance.n, instance.m
    remaining_cap = np.zeros(m)
    for i in open_set:
        remaining_cap[i] = instance.capacities[i]

    # Sort customers by demand (largest first for better packing)
    order = sorted(range(n), key=lambda j: instance.demands[j], reverse=True)
    assignments = [-1] * n

    for j in order:
        best_fac = -1
        best_cost = float("inf")
        for i in open_set:
            if remaining_cap[i] >= instance.demands[j] - 1e-10:
                if instance.assignment_costs[i][j] < best_cost:
                    best_cost = instance.assignment_costs[i][j]
                    best_fac = i
        if best_fac >= 0:
            assignments[j] = best_fac
            remaining_cap[best_fac] -= instance.demands[j]
        else:
            # Fallback: assign to facility with most remaining capacity
            best_fac = max(open_set, key=lambda i: remaining_cap[i])
            assignments[j] = best_fac
            remaining_cap[best_fac] -= instance.demands[j]

    total = sum(instance.fixed_costs[i] for i in open_set)
    total += sum(instance.assignment_costs[assignments[j]][j] for j in range(n))
    return assignments, total


def greedy_add(instance: CFLPInstance) -> CFLPSolution:
    """Solve CFLP using greedy facility addition.

    Iteratively open facilities that reduce total cost most, checking
    capacity feasibility.

    Args:
        instance: A CFLPInstance.

    Returns:
        CFLPSolution.
    """
    m = instance.m
    open_set: set[int] = set()
    closed = set(range(m))

    # Start with the facility that has lowest (fixed_cost / capacity) ratio
    # and can serve the most demand
    best_first = min(
        range(m),
        key=lambda i: instance.fixed_costs[i] / max(instance.capacities[i], 1e-10),
    )
    open_set.add(best_first)
    closed.remove(best_first)

    current_assignments, current_cost = _assign_capacitated(instance, open_set)

    # Phase 1: keep opening until capacity is sufficient
    while closed and not _capacity_feasible(instance, open_set, current_assignments):
        best_fac = min(
            closed,
            key=lambda i: instance.fixed_costs[i] / max(instance.capacities[i], 1e-10),
        )
        open_set.add(best_fac)
        closed.remove(best_fac)
        current_assignments, current_cost = _assign_capacitated(instance, open_set)

    # Phase 2: try opening more if it reduces cost
    improved = True
    while improved and closed:
        improved = False
        best_fac = -1
        best_cost = current_cost

        for candidate in closed:
            trial = open_set | {candidate}
            assignments, cost = _assign_capacitated(instance, trial)
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_fac = candidate

        if best_fac >= 0:
            open_set.add(best_fac)
            closed.remove(best_fac)
            current_assignments, current_cost = _assign_capacitated(instance, open_set)
            improved = True

    return CFLPSolution(
        open_facilities=sorted(open_set),
        assignments=current_assignments,
        cost=current_cost,
    )


def _capacity_feasible(
    instance: CFLPInstance, open_set: set[int], assignments: list[int]
) -> bool:
    """Check if assignments respect capacity constraints."""
    load = np.zeros(instance.m)
    for j, fac in enumerate(assignments):
        load[fac] += instance.demands[j]
    for i in open_set:
        if load[i] > instance.capacities[i] + 1e-10:
            return False
    return True


if __name__ == "__main__":
    inst = _inst.small_cflp_3_5()
    sol = greedy_add(inst)
    print(f"Greedy: cost={sol.cost:.1f}, open={sol.open_facilities}")

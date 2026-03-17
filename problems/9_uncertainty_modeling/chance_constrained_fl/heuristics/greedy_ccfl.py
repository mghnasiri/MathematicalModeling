"""
Greedy Heuristics for Chance-Constrained Facility Location

Two approaches:
    1. Greedy open: Iteratively open the facility that reduces cost
       most while respecting chance constraints.
    2. Mean-demand greedy: Solve deterministic problem on expected
       demands, verify chance feasibility.

Complexity: O(m^2 * n * S) for greedy open.

References:
    - Snyder, L.V. (2006). Facility location under uncertainty. IIE Trans.,
      38(7), 547-564. https://doi.org/10.1080/07408170500216480
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("ccfl_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
CCFLInstance = _inst.CCFLInstance
CCFLSolution = _inst.CCFLSolution


def _assign_customers(instance: CCFLInstance,
                      open_facs: list[int]) -> np.ndarray:
    """Assign each customer to cheapest open facility respecting chance constraints.

    Uses a greedy approach: sort customers by hardest-to-serve first,
    assign to cheapest feasible facility.
    """
    assignments = np.full(instance.n_customers, -1, dtype=int)
    fac_customers: dict[int, list[int]] = {i: [] for i in open_facs}

    # Sort customers by max demand (hardest first)
    max_demands = instance.demand_scenarios.max(axis=0)
    order = np.argsort(-max_demands)

    for j in order:
        best_fac = -1
        best_cost = float("inf")
        for i in open_facs:
            cost = instance.assignment_costs[i, j]
            # Check if adding this customer keeps the chance constraint
            test_customers = fac_customers[i] + [j]
            viol = instance.capacity_violation_prob(i, test_customers)
            if viol <= instance.alpha + 1e-9 and cost < best_cost:
                best_cost = cost
                best_fac = i

        if best_fac == -1:
            # Fallback: assign to cheapest open facility ignoring constraint
            best_fac = min(open_facs, key=lambda i: instance.assignment_costs[i, j])

        assignments[j] = best_fac
        fac_customers[best_fac].append(j)

    return assignments


def greedy_open(instance: CCFLInstance) -> CCFLSolution:
    """Iteratively open facilities to minimize cost.

    Start with no facilities open. At each step, open the facility
    that gives the greatest cost reduction. Repeat until all customers
    are served with chance-feasible assignments.

    Args:
        instance: CCFLInstance.

    Returns:
        CCFLSolution.
    """
    m = instance.n_facilities
    closed = set(range(m))
    opened = []

    # Open facilities one by one
    while closed:
        best_fac = -1
        best_cost = float("inf")
        best_assign = None

        for i in closed:
            test_open = opened + [i]
            assign = _assign_customers(instance, test_open)
            cost = instance.total_cost(test_open, assign)
            if cost < best_cost:
                best_cost = cost
                best_fac = i
                best_assign = assign

        if best_fac == -1:
            break

        opened.append(best_fac)
        closed.remove(best_fac)

        # Check if current solution is chance-feasible
        feasible, _ = instance.is_feasible(opened, best_assign)
        if feasible:
            # Try if we actually need this many — check if removing any
            # doesn't break feasibility
            break

    assignments = _assign_customers(instance, opened)

    # Compute max violation
    max_viol = 0.0
    for i in opened:
        custs = [j for j in range(instance.n_customers) if assignments[j] == i]
        viol = instance.capacity_violation_prob(i, custs)
        max_viol = max(max_viol, viol)

    return CCFLSolution(
        open_facilities=opened,
        assignments=assignments,
        total_cost=instance.total_cost(opened, assignments),
        max_violation_prob=max_viol,
    )


def mean_demand_greedy(instance: CCFLInstance) -> CCFLSolution:
    """Solve using mean demands as a deterministic proxy.

    Open cheapest facilities greedily using expected demands for
    capacity checks. Then verify chance feasibility.

    Args:
        instance: CCFLInstance.

    Returns:
        CCFLSolution.
    """
    mean_d = instance.mean_demands
    m = instance.n_facilities

    # Sort facilities by cost-effectiveness (fixed cost / capacity)
    efficiency = instance.fixed_costs / np.maximum(instance.capacities, 1e-9)
    order = np.argsort(efficiency)

    opened = []
    total_cap = 0.0
    total_demand = mean_d.sum()

    for i in order:
        opened.append(int(i))
        total_cap += instance.capacities[i]
        if total_cap >= total_demand:
            break

    if not opened:
        opened = [int(order[0])]

    assignments = _assign_customers(instance, opened)

    max_viol = 0.0
    for i in opened:
        custs = [j for j in range(instance.n_customers) if assignments[j] == i]
        viol = instance.capacity_violation_prob(i, custs)
        max_viol = max(max_viol, viol)

    return CCFLSolution(
        open_facilities=opened,
        assignments=assignments,
        total_cost=instance.total_cost(opened, assignments),
        max_violation_prob=max_viol,
    )

"""
Capacitated p-Median — Heuristics.

Algorithms:
    - Greedy add with capacity-aware assignment.
    - Teitz-Bart interchange with capacity checks.

References:
    Mulvey, J.M. & Beck, M.P. (1984). Solving capacitated clustering
    problems. European Journal of Operational Research, 18(3), 339-348.
    https://doi.org/10.1016/0377-2217(84)90155-3
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


_inst = _load_mod("cpmp_instance_h", os.path.join(_this_dir, "instance.py"))
CPMedianInstance = _inst.CPMedianInstance
CPMedianSolution = _inst.CPMedianSolution


def _assign_customers(instance: CPMedianInstance,
                      open_facs: list[int]) -> tuple[list[int], float]:
    """Assign customers to nearest open facility respecting capacity."""
    assignments = [-1] * instance.n
    loads = np.zeros(instance.m)
    total_cost = 0.0

    # Sort customers by distance to nearest facility (hardest first)
    cust_order = sorted(range(instance.n),
                        key=lambda j: min(instance.distance_matrix[i][j]
                                          for i in open_facs))

    for j in cust_order:
        best_fac = -1
        best_cost = float("inf")
        for i in open_facs:
            if loads[i] + instance.demands[j] <= instance.capacities[i] + 1e-6:
                c = instance.demands[j] * instance.distance_matrix[i][j]
                if c < best_cost:
                    best_cost = c
                    best_fac = i
        if best_fac == -1:
            # Fallback: nearest facility ignoring capacity
            best_fac = min(open_facs,
                           key=lambda i: instance.distance_matrix[i][j])
            best_cost = instance.demands[j] * instance.distance_matrix[best_fac][j]

        assignments[j] = best_fac
        loads[best_fac] += instance.demands[j]
        total_cost += best_cost

    return assignments, total_cost


def greedy_add(instance: CPMedianInstance) -> CPMedianSolution:
    """Greedy add heuristic for capacitated p-Median.

    Iteratively open the facility that most reduces total cost.

    Args:
        instance: CPMedian instance.

    Returns:
        CPMedianSolution.
    """
    open_facs = []
    closed = set(range(instance.m))

    for _ in range(instance.p):
        best_fac = None
        best_cost = float("inf")
        best_assign = None

        for i in closed:
            trial = open_facs + [i]
            assign, cost = _assign_customers(instance, trial)
            if cost < best_cost:
                best_cost = cost
                best_fac = i
                best_assign = assign

        open_facs.append(best_fac)
        closed.remove(best_fac)

    assignments, total_cost = _assign_customers(instance, open_facs)
    return CPMedianSolution(open_facilities=open_facs,
                            assignments=assignments, total_cost=total_cost)


def teitz_bart(instance: CPMedianInstance) -> CPMedianSolution:
    """Teitz-Bart interchange with capacity checks.

    Start from greedy solution, iteratively swap open/closed facilities.

    Args:
        instance: CPMedian instance.

    Returns:
        CPMedianSolution.
    """
    init = greedy_add(instance)
    open_facs = list(init.open_facilities)
    _, cost = _assign_customers(instance, open_facs)

    improved = True
    while improved:
        improved = False
        open_set = set(open_facs)
        closed = [i for i in range(instance.m) if i not in open_set]

        for i_out in list(open_facs):
            for i_in in closed:
                trial = [f for f in open_facs if f != i_out] + [i_in]
                assign, new_cost = _assign_customers(instance, trial)
                if new_cost < cost - 1e-6:
                    open_facs = trial
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break

    assignments, total_cost = _assign_customers(instance, open_facs)
    return CPMedianSolution(open_facilities=open_facs,
                            assignments=assignments, total_cost=total_cost)


if __name__ == "__main__":
    from instance import small_cpmp_6

    inst = small_cpmp_6()
    sol1 = greedy_add(inst)
    print(f"Greedy: {sol1}")
    sol2 = teitz_bart(inst)
    print(f"Teitz-Bart: {sol2}")

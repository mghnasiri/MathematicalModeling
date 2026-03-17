"""
MILP Formulation — Exact solver for the Multiple Knapsack Problem.

Problem: Multiple Knapsack (MKP)
Complexity: NP-hard; practical for small-to-medium instances via MILP.

Uses scipy.optimize.milp with the HiGHS solver. The formulation uses
binary decision variables x_{ij} (item i assigned to knapsack j).

    max  sum_{i,j} v_i * x_{ij}
    s.t. sum_j x_{ij} <= 1       for all i  (each item at most once)
         sum_i w_i * x_{ij} <= C_j for all j  (capacity)
         x_{ij} in {0, 1}

References:
    Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and
    Computer Implementations. John Wiley & Sons, Chichester.

    Pisinger, D. (1999). An exact algorithm for large multiple knapsack
    problems. European Journal of Operational Research, 114(3), 528-541.
    https://doi.org/10.1016/S0377-2217(98)00120-9
"""

from __future__ import annotations

import os
import importlib.util
import sys

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import eye as speye

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mkp_instance_ilp", os.path.join(_parent_dir, "instance.py"))
MultipleKnapsackInstance = _inst.MultipleKnapsackInstance
MultipleKnapsackSolution = _inst.MultipleKnapsackSolution


def milp_solver(
    instance: MultipleKnapsackInstance,
    time_limit: float = 60.0,
) -> MultipleKnapsackSolution | None:
    """Solve MKP exactly using MILP formulation via SciPy HiGHS.

    Decision variables: x_{ij} in {0,1} for i in [n], j in [m].
    Variable index: i * m + j.

    Args:
        instance: A MultipleKnapsackInstance.
        time_limit: Time limit in seconds.

    Returns:
        MultipleKnapsackSolution if feasible, None otherwise.
    """
    n = instance.n
    m = instance.m
    num_vars = n * m

    # Objective: maximize sum v_i * x_{ij} => minimize -v_i * x_{ij}
    c = np.zeros(num_vars)
    for i in range(n):
        for j in range(m):
            c[i * m + j] = -instance.values[i]

    # Constraint 1: each item assigned at most once
    # sum_j x_{ij} <= 1 for all i
    A_item = np.zeros((n, num_vars))
    for i in range(n):
        for j in range(m):
            A_item[i, i * m + j] = 1.0

    # Constraint 2: knapsack capacity
    # sum_i w_i * x_{ij} <= C_j for all j
    A_cap = np.zeros((m, num_vars))
    for j in range(m):
        for i in range(n):
            A_cap[j, i * m + j] = instance.weights[i]

    constraints = [
        LinearConstraint(A_item, ub=np.ones(n)),
        LinearConstraint(A_cap, ub=instance.capacities),
    ]

    # All variables are binary (integers in [0, 1])
    integrality = np.ones(num_vars)
    bounds_lower = np.zeros(num_vars)
    bounds_upper = np.ones(num_vars)

    from scipy.optimize import Bounds
    bounds = Bounds(lb=bounds_lower, ub=bounds_upper)

    options = {"time_limit": time_limit}

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=options,
    )

    if not result.success:
        return None

    x = result.x
    assignments: list[list[int]] = [[] for _ in range(m)]
    for i in range(n):
        for j in range(m):
            if x[i * m + j] > 0.5:
                assignments[j].append(i)

    total_value = sum(
        instance.values[i] for ks in assignments for i in ks
    )

    return MultipleKnapsackSolution(
        assignments=assignments, value=float(total_value)
    )


if __name__ == "__main__":
    _inst_mod = _load_mod("mkp_inst_ilp_main", os.path.join(_parent_dir, "instance.py"))
    small_mkp_6_2 = _inst_mod.small_mkp_6_2
    validate_solution = _inst_mod.validate_solution

    inst = small_mkp_6_2()
    sol = milp_solver(inst)
    if sol is not None:
        valid, errors = validate_solution(inst, sol)
        print(f"MILP: value={sol.value:.0f}, valid={valid}")
        print(f"  Assignments: {sol.assignments}")
    else:
        print("MILP: no feasible solution found")

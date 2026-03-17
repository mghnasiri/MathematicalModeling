"""
Deterministic Equivalent (Extensive Form) for Two-Stage Stochastic Programs

Converts the stochastic program into a single large LP by enumerating all
scenarios. The resulting LP has n1 + S*n2 variables and m1 + S*m2 constraints.

Solved via scipy.optimize.linprog (HiGHS solver).

Complexity: Polynomial in the size of the deterministic equivalent LP,
            but the LP grows linearly with the number of scenarios S.

References:
    - Birge, J.R. & Louveaux, F. (2011). Introduction to Stochastic
      Programming, 2nd ed. Springer, Chapter 3.
"""
from __future__ import annotations

import sys
import os

import numpy as np
from scipy.optimize import linprog

import importlib.util

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("sp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
TwoStageSPInstance = _inst.TwoStageSPInstance
TwoStageSPSolution = _inst.TwoStageSPSolution


def solve_deterministic_equivalent(
    instance: TwoStageSPInstance,
) -> TwoStageSPSolution | None:
    """Solve 2SSP by constructing and solving the deterministic equivalent LP.

    Variables: [x (n1), y_1 (n2), y_2 (n2), ..., y_S (n2)]

    Args:
        instance: TwoStageSPInstance to solve.

    Returns:
        TwoStageSPSolution or None if infeasible.
    """
    S = instance.n_scenarios
    n1 = instance.n1
    n2 = instance.n2
    m1 = instance.m1
    m2 = instance.m2
    total_vars = n1 + S * n2

    # Objective: c^T x + sum_s p_s * q_s^T y_s
    obj = np.zeros(total_vars)
    obj[:n1] = instance.c
    for s in range(S):
        start = n1 + s * n2
        obj[start:start + n2] = instance.probabilities[s] * instance.scenarios[s]["q"]

    # First-stage equality constraints: Ax = b
    A_eq_rows = []
    b_eq_list = []
    if m1 > 0:
        A_eq_fs = np.zeros((m1, total_vars))
        A_eq_fs[:, :n1] = instance.A
        A_eq_rows.append(A_eq_fs)
        b_eq_list.append(instance.b)

    # Second-stage constraints: T_s x + W_s y_s <= h_s  (per scenario)
    A_ub_rows = []
    b_ub_list = []
    for s in range(S):
        sc = instance.scenarios[s]
        row = np.zeros((m2, total_vars))
        row[:, :n1] = sc["T"]
        start = n1 + s * n2
        row[:, start:start + n2] = sc["W"]
        A_ub_rows.append(row)
        b_ub_list.append(sc["h"])

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub = np.concatenate(b_ub_list) if b_ub_list else None
    A_eq = np.vstack(A_eq_rows) if A_eq_rows else None
    b_eq = np.concatenate(b_eq_list) if b_eq_list else None

    bounds = [(0, None)] * total_vars

    result = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        return None

    x = result.x[:n1]
    first_stage_cost = float(np.dot(instance.c, x))

    recourse_solutions = {}
    expected_recourse = 0.0
    for s in range(S):
        start = n1 + s * n2
        y_s = result.x[start:start + n2]
        recourse_solutions[s] = y_s
        expected_recourse += instance.probabilities[s] * np.dot(
            instance.scenarios[s]["q"], y_s
        )

    return TwoStageSPSolution(
        x=x,
        first_stage_cost=first_stage_cost,
        expected_recourse_cost=expected_recourse,
        total_cost=first_stage_cost + expected_recourse,
        recourse_solutions=recourse_solutions,
    )


def solve_expected_value(instance: TwoStageSPInstance) -> TwoStageSPSolution | None:
    """Solve the Expected Value (EV) problem.

    Replace all scenarios with their expected values and solve the
    resulting deterministic problem. This gives the EV solution,
    which is a lower bound on the true stochastic solution.

    Args:
        instance: TwoStageSPInstance.

    Returns:
        TwoStageSPSolution using mean scenario, or None if infeasible.
    """
    S = instance.n_scenarios
    probs = instance.probabilities

    # Compute expected scenario
    q_mean = sum(probs[s] * instance.scenarios[s]["q"] for s in range(S))
    T_mean = sum(probs[s] * instance.scenarios[s]["T"] for s in range(S))
    W_mean = sum(probs[s] * instance.scenarios[s]["W"] for s in range(S))
    h_mean = sum(probs[s] * instance.scenarios[s]["h"] for s in range(S))

    mean_instance = TwoStageSPInstance(
        c=instance.c,
        A=instance.A,
        b=instance.b,
        scenarios=[{"q": q_mean, "T": T_mean, "W": W_mean, "h": h_mean}],
        probabilities=np.array([1.0]),
    )

    return solve_deterministic_equivalent(mean_instance)


if __name__ == "__main__":
    inst = TwoStageSPInstance.capacity_planning(n_facilities=3, n_scenarios=5)
    sol_de = solve_deterministic_equivalent(inst)
    print(f"Deterministic equivalent: {sol_de}")
    sol_ev = solve_expected_value(inst)
    print(f"Expected value solution: {sol_ev}")
    if sol_de and sol_ev:
        vss = sol_de.total_cost - sol_ev.total_cost
        print(f"Value of stochastic solution (VSS): {vss:.2f}")

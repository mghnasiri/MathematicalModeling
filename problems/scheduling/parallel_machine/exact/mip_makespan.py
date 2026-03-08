"""
MIP Formulation for Pm || Cmax — Mixed Integer Programming

Formulates the identical parallel machine makespan problem as a MIP.
Uses the assignment-based formulation with SciPy's HiGHS solver.

Formulation:
    min  Cmax
    s.t. sum_i x[i,j] = 1         for all j  (each job assigned once)
         Cmax >= sum_j p[j]*x[i,j] for all i  (makespan definition)
         x[i,j] in {0,1}

For unrelated machines, p[j] is replaced by p[i,j].

Notation: Pm || Cmax (or Rm || Cmax)
Complexity: NP-hard (MIP)
Reference: Pinedo, M.L. (2016). "Scheduling: Theory, Algorithms, and Systems"
           5th Edition, Springer.
"""

from __future__ import annotations
import sys
import os
import importlib.util
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_this_dir = os.path.dirname(os.path.abspath(__file__))
_instance_path = os.path.join(_this_dir, "..", "instance.py")
_spec = importlib.util.spec_from_file_location("pm_instance", _instance_path)
_pm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pm_instance", _pm_instance)
_spec.loader.exec_module(_pm_instance)

ParallelMachineInstance = _pm_instance.ParallelMachineInstance
ParallelMachineSolution = _pm_instance.ParallelMachineSolution
compute_makespan = _pm_instance.compute_makespan
compute_machine_loads = _pm_instance.compute_machine_loads


def mip_makespan(
    instance: ParallelMachineInstance,
    time_limit: float = 60.0,
) -> ParallelMachineSolution:
    """
    Solve Pm || Cmax (or Rm || Cmax) exactly using MIP.

    Variables:
        x[i,j] in {0,1}: 1 if job j assigned to machine i
        Cmax: makespan (continuous)

    Total variables: m*n + 1
    Variable ordering: [x[0,0], x[0,1], ..., x[m-1,n-1], Cmax]

    Args:
        instance: A ParallelMachineInstance.
        time_limit: Maximum solver time in seconds.

    Returns:
        ParallelMachineSolution with optimal (or best found) assignment.
    """
    n = instance.n
    m = instance.m

    num_x = m * n
    num_vars = num_x + 1  # +1 for Cmax

    # Variable indices
    def x_idx(i: int, j: int) -> int:
        return i * n + j

    cmax_idx = num_x

    # Objective: minimize Cmax
    c = np.zeros(num_vars)
    c[cmax_idx] = 1.0

    # Constraints
    A_eq_rows = []
    b_eq = []
    A_ub_rows = []
    b_ub = []

    # Constraint 1: each job assigned to exactly one machine
    # sum_i x[i,j] = 1 for all j
    for j in range(n):
        row = np.zeros(num_vars)
        for i in range(m):
            row[x_idx(i, j)] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    # Constraint 2: Cmax >= sum_j p[i,j] * x[i,j] for all i
    # Rewritten: sum_j p[i,j] * x[i,j] - Cmax <= 0
    for i in range(m):
        row = np.zeros(num_vars)
        for j in range(n):
            row[x_idx(i, j)] = instance.get_processing_time(j, i)
        row[cmax_idx] = -1.0
        A_ub_rows.append(row)
        b_ub.append(0.0)

    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq_arr = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    # Variable bounds
    lower_bounds = np.zeros(num_vars)
    upper_bounds = np.ones(num_vars)
    upper_bounds[cmax_idx] = float(instance.processing_times.sum())  # Cmax upper bound

    # Integrality: x variables are binary, Cmax is continuous
    integrality = np.ones(num_vars, dtype=int)
    integrality[cmax_idx] = 0

    # Build constraints for milp
    constraints = []
    if A_eq is not None:
        constraints.append(LinearConstraint(A_eq, b_eq_arr, b_eq_arr))
    if A_ub is not None:
        constraints.append(LinearConstraint(A_ub, -np.inf, b_ub_arr))

    bounds = Bounds(lower_bounds, upper_bounds)

    # Solve
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit},
    )

    if not result.success:
        # Fallback to LPT
        _lpt_path = os.path.join(_this_dir, "..", "heuristics", "lpt.py")
        _lpt_spec = importlib.util.spec_from_file_location("pm_lpt_fb", _lpt_path)
        _lpt_mod = importlib.util.module_from_spec(_lpt_spec)
        _lpt_spec.loader.exec_module(_lpt_mod)
        return _lpt_mod.lpt(instance)

    # Extract solution
    x_vals = result.x
    assignment: list[list[int]] = [[] for _ in range(m)]

    for j in range(n):
        for i in range(m):
            if x_vals[x_idx(i, j)] > 0.5:
                assignment[i].append(j)
                break

    ms = compute_makespan(instance, assignment)
    loads = compute_machine_loads(instance, assignment)
    return ParallelMachineSolution(
        assignment=assignment, makespan=ms, machine_loads=loads
    )


if __name__ == "__main__":
    print("=== MIP for Parallel Machine Makespan ===\n")

    instance = ParallelMachineInstance.random_identical(n=12, m=3, seed=42)
    print(f"Instance: {instance.n} jobs, {instance.m} machines")
    print(f"Processing times: {instance.processing_times}")

    sol = mip_makespan(instance)
    print(f"\nMIP Makespan: {sol.makespan:.0f}")
    for i, jobs in enumerate(sol.assignment):
        print(f"  Machine {i}: {jobs} (load={sol.machine_loads[i]:.0f})")

    # Compare with LPT
    from heuristics.lpt import lpt
    sol_lpt = lpt(instance)
    print(f"\nLPT Makespan: {sol_lpt.makespan:.0f}")

    # Lower bound
    lb = max(float(instance.processing_times.max()),
             float(instance.processing_times.sum()) / instance.m)
    print(f"Lower bound:  {lb:.0f}")

"""
MIP Formulation — Position-Based Model for Fm | prmu | Cmax

Formulates the permutation flow shop as a Mixed-Integer Program using
position-based assignment variables. This is the most natural MIP
formulation for PFSP because the permutation constraint maps directly
to assignment constraints.

Decision Variables:
    x[j][k] ∈ {0,1}  — 1 if job j is assigned to position k
    C[i][k]           — completion time of the job in position k on machine i

Constraints:
    1. Each job is assigned to exactly one position
    2. Each position has exactly one job
    3. Completion time recursion (linking x and C)
    4. Makespan definition

This formulation has O(n²) binary variables and O(n*m) continuous variables.
It's suitable for small instances (n ≤ ~15-20 with a good solver).

Two implementations are provided:
    1. scipy_mip(): Uses SciPy's milp solver (no external dependencies)
    2. ortools_mip(): Uses Google OR-Tools CP-SAT (better performance, optional)

Notation: Fm | prmu | Cmax
Reference: Wagner, H.M. (1959). "An Integer Linear-Programming Model for
           Machine Scheduling"
           Manne, A.S. (1960). "On the Job-Shop Scheduling Problem"
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def scipy_mip(
    instance: FlowShopInstance,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> FlowShopSolution:
    """
    Solve PFSP using SciPy's MILP solver (no external dependencies).

    Uses a linearized position-based formulation. SciPy's MILP solver
    (HiGHS backend) handles the branch-and-cut internally.

    Args:
        instance: A FlowShopInstance.
        time_limit: Maximum runtime in seconds.
        verbose: Print solver output.

    Returns:
        FlowShopSolution with the (hopefully optimal) permutation.

    Note:
        SciPy's MILP is limited compared to commercial solvers. For
        instances larger than ~12 jobs, consider OR-Tools or Gurobi.
    """
    from scipy.optimize import LinearConstraint, milp, Bounds

    n = instance.n
    m = instance.m
    p = instance.processing_times  # shape (m, n)

    # Variable layout:
    # x[j*n + k] = binary: job j in position k     (n*n variables)
    # C[i*n + k] = continuous: completion time       (m*n variables)
    # cmax       = continuous: makespan               (1 variable)
    # Total: n*n + m*n + 1 variables

    n_x = n * n       # binary assignment variables
    n_c = m * n       # completion time variables
    n_vars = n_x + n_c + 1

    idx_x = lambda j, k: j * n + k
    idx_c = lambda i, k: n_x + i * n + k
    idx_cmax = n_x + n_c

    # Objective: minimize cmax
    c_obj = np.zeros(n_vars)
    c_obj[idx_cmax] = 1.0

    # Variable bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for j in range(n):
        for k in range(n):
            ub[idx_x(j, k)] = 1.0  # binary

    # Integer constraints: x variables are binary
    integrality = np.zeros(n_vars)
    for j in range(n):
        for k in range(n):
            integrality[idx_x(j, k)] = 1  # 1 = integer

    constraints = []

    # Constraint 1: Each job assigned to exactly one position
    # sum_k x[j][k] = 1 for all j
    for j in range(n):
        A_row = np.zeros(n_vars)
        for k in range(n):
            A_row[idx_x(j, k)] = 1.0
        constraints.append(LinearConstraint(A_row, 1.0, 1.0))

    # Constraint 2: Each position has exactly one job
    # sum_j x[j][k] = 1 for all k
    for k in range(n):
        A_row = np.zeros(n_vars)
        for j in range(n):
            A_row[idx_x(j, k)] = 1.0
        constraints.append(LinearConstraint(A_row, 1.0, 1.0))

    # Constraint 3: Completion time on machine 0, position 0
    # C[0][0] = sum_j p[0][j] * x[j][0]
    # → C[0][0] - sum_j p[0][j] * x[j][0] = 0
    A_row = np.zeros(n_vars)
    A_row[idx_c(0, 0)] = 1.0
    for j in range(n):
        A_row[idx_x(j, 0)] = -float(p[0, j])
    constraints.append(LinearConstraint(A_row, 0.0, 0.0))

    # Constraint 4: Machine 0, positions k >= 1
    # C[0][k] = C[0][k-1] + sum_j p[0][j] * x[j][k]
    # → C[0][k] - C[0][k-1] - sum_j p[0][j] * x[j][k] = 0
    for k in range(1, n):
        A_row = np.zeros(n_vars)
        A_row[idx_c(0, k)] = 1.0
        A_row[idx_c(0, k - 1)] = -1.0
        for j in range(n):
            A_row[idx_x(j, k)] = -float(p[0, j])
        constraints.append(LinearConstraint(A_row, 0.0, 0.0))

    # Constraint 5: Machines i >= 1, position 0
    # C[i][0] = C[i-1][0] + sum_j p[i][j] * x[j][0]
    for i in range(1, m):
        A_row = np.zeros(n_vars)
        A_row[idx_c(i, 0)] = 1.0
        A_row[idx_c(i - 1, 0)] = -1.0
        for j in range(n):
            A_row[idx_x(j, 0)] = -float(p[i, j])
        constraints.append(LinearConstraint(A_row, 0.0, 0.0))

    # Constraint 6: General case — C[i][k] >= C[i-1][k] + p_ik AND
    #                                C[i][k] >= C[i][k-1] + p_ik
    # Since C[i][k] = max(C[i-1][k], C[i][k-1]) + sum_j p[i][j]*x[j][k],
    # we linearize with two >= constraints:
    for i in range(1, m):
        for k in range(1, n):
            # C[i][k] >= C[i-1][k] + sum_j p[i][j] * x[j][k]
            A_row1 = np.zeros(n_vars)
            A_row1[idx_c(i, k)] = 1.0
            A_row1[idx_c(i - 1, k)] = -1.0
            for j in range(n):
                A_row1[idx_x(j, k)] = -float(p[i, j])
            constraints.append(LinearConstraint(A_row1, 0.0, np.inf))

            # C[i][k] >= C[i][k-1] + sum_j p[i][j] * x[j][k]
            A_row2 = np.zeros(n_vars)
            A_row2[idx_c(i, k)] = 1.0
            A_row2[idx_c(i, k - 1)] = -1.0
            for j in range(n):
                A_row2[idx_x(j, k)] = -float(p[i, j])
            constraints.append(LinearConstraint(A_row2, 0.0, np.inf))

    # Constraint 7: Cmax >= C[m-1][k] for all k
    for k in range(n):
        A_row = np.zeros(n_vars)
        A_row[idx_cmax] = 1.0
        A_row[idx_c(m - 1, k)] = -1.0
        constraints.append(LinearConstraint(A_row, 0.0, np.inf))

    # Solve
    options = {"time_limit": time_limit, "disp": verbose}
    result = milp(
        c=c_obj,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options=options,
    )

    if not result.success:
        # Fallback to NEH
        from heuristics.neh import neh as neh_solve
        return neh_solve(instance)

    # Extract permutation from x variables
    x_vals = result.x[:n_x].reshape(n, n)
    permutation = []
    for k in range(n):
        for j in range(n):
            if x_vals[j, k] > 0.5:
                permutation.append(j)
                break

    makespan = compute_makespan(instance, permutation)

    if verbose:
        print(f"MIP makespan: {makespan}")
        print(f"Solver status: {result.message}")

    return FlowShopSolution(permutation=permutation, makespan=makespan)


def ortools_cpsat(
    instance: FlowShopInstance,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> FlowShopSolution:
    """
    Solve PFSP using Google OR-Tools CP-SAT solver.

    CP-SAT is a constraint programming solver that uses SAT/lazy clause
    generation. It is significantly more powerful than generic MIP solvers
    for scheduling problems because it natively handles the 'max'
    constraint without linearization.

    Args:
        instance: A FlowShopInstance.
        time_limit: Maximum runtime in seconds.
        verbose: Print solver output.

    Returns:
        FlowShopSolution with the optimal or best-found permutation.

    Raises:
        ImportError: If ortools is not installed.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError(
            "Google OR-Tools is required for CP-SAT. "
            "Install with: pip install ortools"
        )

    n = instance.n
    m = instance.m
    p = instance.processing_times

    model = cp_model.CpModel()

    # Upper bound for time horizon
    horizon = int(p.sum())

    # Variables: start[j][i] = start time of job j on machine i
    start = {}
    end = {}
    intervals = {}
    for j in range(n):
        for i in range(m):
            start[j, i] = model.new_int_var(0, horizon, f"start_{j}_{i}")
            dur = int(p[i, j])
            end[j, i] = model.new_int_var(0, horizon, f"end_{j}_{i}")
            intervals[j, i] = model.new_interval_var(
                start[j, i], dur, end[j, i], f"interval_{j}_{i}"
            )

    # Precedence: job j must finish on machine i before starting on i+1
    for j in range(n):
        for i in range(m - 1):
            model.add(start[j, i + 1] >= end[j, i])

    # No overlap: on each machine, no two jobs overlap
    for i in range(m):
        model.add_no_overlap([intervals[j, i] for j in range(n)])

    # Permutation constraint: same job order on all machines
    # For each pair of jobs (j1, j2), either j1 before j2 on ALL machines,
    # or j2 before j1 on ALL machines
    order = {}
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            b = model.new_bool_var(f"order_{j1}_{j2}")
            order[j1, j2] = b
            for i in range(m):
                # If b=1: j1 before j2 on machine i
                model.add(start[j2, i] >= end[j1, i]).only_enforce_if(b)
                # If b=0: j2 before j1 on machine i
                model.add(start[j1, i] >= end[j2, i]).only_enforce_if(~b)

    # Objective: minimize makespan
    makespan_var = model.new_int_var(0, horizon, "makespan")
    for j in range(n):
        model.add(makespan_var >= end[j, m - 1])
    model.minimize(makespan_var)

    # Warm start with NEH
    neh_sol = neh(instance)
    neh_completion = np.zeros((m, n), dtype=int)
    for pos, job in enumerate(neh_sol.permutation):
        if pos == 0:
            neh_completion[0, pos] = int(p[0, job])
        else:
            neh_completion[0, pos] = neh_completion[0, pos - 1] + int(p[0, job])
        for i in range(1, m):
            prev_m = neh_completion[i - 1, pos]
            prev_j = neh_completion[i, pos - 1] if pos > 0 else 0
            neh_completion[i, pos] = max(prev_m, prev_j) + int(p[i, job])

    # Set hint
    for pos, job in enumerate(neh_sol.permutation):
        for i in range(m):
            s = neh_completion[i, pos] - int(p[i, job])
            model.add_hint(start[job, i], s)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if not verbose:
        solver.parameters.log_search_progress = False

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Extract job order from start times on machine 0
        job_starts = [(solver.value(start[j, 0]), j) for j in range(n)]
        job_starts.sort()
        permutation = [j for _, j in job_starts]
        makespan = compute_makespan(instance, permutation)

        if verbose:
            status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
            print(f"CP-SAT [{status_name}] Makespan: {makespan}")

        return FlowShopSolution(permutation=permutation, makespan=makespan)
    else:
        # Fallback
        return neh_sol


if __name__ == "__main__":
    # Test SciPy MIP on a small instance
    print("=" * 50)
    print("SciPy MIP on 6×3 instance")
    print("=" * 50)
    instance = FlowShopInstance.random(n=6, m=3, seed=42)
    sol_mip = scipy_mip(instance, verbose=True)
    print(f"MIP permutation: {sol_mip.permutation}")
    print(f"MIP makespan:    {sol_mip.makespan}")

    neh_sol = neh(instance)
    print(f"NEH makespan:    {neh_sol.makespan}")

    # Test OR-Tools if available
    try:
        print("\n" + "=" * 50)
        print("OR-Tools CP-SAT on 10×3 instance")
        print("=" * 50)
        instance2 = FlowShopInstance.random(n=10, m=3, seed=42)
        sol_cp = ortools_cpsat(instance2, time_limit=10.0, verbose=True)
        print(f"CP-SAT permutation: {sol_cp.permutation}")
        print(f"CP-SAT makespan:    {sol_cp.makespan}")
    except ImportError as e:
        print(f"\nOR-Tools not available: {e}")
        print("Install with: pip install ortools")

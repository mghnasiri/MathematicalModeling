"""
MULTIFIT — Bin-Packing-Based Heuristic for Pm || Cmax

MULTIFIT reduces the parallel machine makespan problem to a series of bin
packing problems. It performs binary search on the makespan value, using
First Fit Decreasing (FFD) as the bin packing heuristic at each step.

Algorithm:
    1. Set lower bound L = max(max(pj), sum(pj)/m) and upper bound U = 2*L.
    2. Repeat for a fixed number of iterations:
       a. Set C = (L + U) / 2 (candidate makespan).
       b. Run FFD with bin capacity C and m bins:
          - Sort jobs by decreasing processing time.
          - Assign each job to the first bin where it fits.
       c. If all jobs fit in m bins: U = C (feasible).
       d. Otherwise: L = C (infeasible).
    3. Return the assignment from the last feasible C.

Approximation: MULTIFIT achieves a ratio of at most 1.22 for Pm || Cmax.

Notation: Pm || Cmax
Complexity: O(k * n * m) where k is the number of binary search iterations
Reference: Coffman, E.G., Garey, M.R. & Johnson, D.S. (1978). "An Application
           of Bin-Packing to Multiprocessor Scheduling"
           SIAM Journal on Computing, 7(1):1-17.
           DOI: 10.1137/0207001
"""

from __future__ import annotations
import sys
import os
import importlib.util

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


def multifit(
    instance: ParallelMachineInstance,
    max_iterations: int = 30,
) -> ParallelMachineSolution:
    """
    MULTIFIT heuristic for identical parallel machine makespan minimization.

    Args:
        instance: A ParallelMachineInstance (identical machines).
        max_iterations: Number of binary search iterations.

    Returns:
        ParallelMachineSolution with the assignment and makespan.
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times

    # Binary search bounds
    lower = max(float(p.max()), float(p.sum()) / m)
    upper = float(p.sum())  # worst case: all on one machine

    best_assignment = None
    best_ms = float('inf')

    for _ in range(max_iterations):
        mid = (lower + upper) / 2.0

        assignment = _first_fit_decreasing(instance, mid)

        if assignment is not None:
            # Feasible: all jobs fit in m machines
            ms = compute_makespan(instance, assignment)
            if ms < best_ms:
                best_ms = ms
                best_assignment = assignment
            upper = mid
        else:
            lower = mid

    # If no feasible solution found (shouldn't happen with correct bounds),
    # fall back to simple LPT
    if best_assignment is None:
        _lpt_path = os.path.join(_this_dir, "lpt.py")
        _lpt_spec = importlib.util.spec_from_file_location("pm_lpt_fb", _lpt_path)
        _lpt_mod = importlib.util.module_from_spec(_lpt_spec)
        _lpt_spec.loader.exec_module(_lpt_mod)
        return _lpt_mod.lpt(instance)

    loads = compute_machine_loads(instance, best_assignment)
    return ParallelMachineSolution(
        assignment=best_assignment, makespan=best_ms, machine_loads=loads
    )


def _first_fit_decreasing(
    instance: ParallelMachineInstance,
    capacity: float,
) -> list[list[int]] | None:
    """
    First Fit Decreasing bin packing with m bins of given capacity.

    Args:
        instance: A ParallelMachineInstance.
        capacity: Maximum load allowed on each machine.

    Returns:
        Assignment if all jobs fit, None otherwise.
    """
    n = instance.n
    m = instance.m

    # Sort jobs by decreasing processing time
    sorted_jobs = sorted(range(n), key=lambda j: instance.processing_times[j],
                         reverse=True)

    assignment: list[list[int]] = [[] for _ in range(m)]
    loads = [0.0] * m

    for job in sorted_jobs:
        pt = float(instance.processing_times[job])
        placed = False

        # First fit: assign to the first machine where it fits
        for i in range(m):
            if loads[i] + pt <= capacity + 1e-9:  # small tolerance
                assignment[i].append(job)
                loads[i] += pt
                placed = True
                break

        if not placed:
            return None  # Infeasible for this capacity

    return assignment


if __name__ == "__main__":
    print("=== MULTIFIT for Parallel Machines ===\n")

    instance = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)
    print(f"Instance: {instance.n} jobs, {instance.m} machines")

    sol = multifit(instance)
    print(f"\nMULTIFIT Makespan: {sol.makespan:.0f}")
    for i, jobs in enumerate(sol.assignment):
        print(f"  Machine {i}: {jobs} (load={sol.machine_loads[i]:.0f})")

    # Compare with LPT
    from heuristics.lpt import lpt
    sol_lpt = lpt(instance)
    print(f"\nLPT      Makespan: {sol_lpt.makespan:.0f}")

    # Lower bound
    lb = max(float(instance.processing_times.max()),
             float(instance.processing_times.sum()) / instance.m)
    print(f"Lower bound:       {lb:.0f}")

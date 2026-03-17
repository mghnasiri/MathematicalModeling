"""
Iterated Greedy for Parallel Machine Scheduling — Pm || Cmax

Problem notation: Pm || Cmax (extends to Qm, Rm)

Iterated Greedy repeatedly destroys the current solution by removing a
subset of jobs, then reconstructs by greedily reinserting them in the
best position. A Boltzmann-based acceptance criterion allows the search
to escape local optima.

Warm-started with LPT (Longest Processing Time) heuristic.

Complexity: O(iterations * d * n * m) where d = destruction size.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Framinan, J.M. & Leisten, R. (2008). A multi-objective iterated
    greedy search for flowshop scheduling with makespan and flowtime
    criteria. OR Spectrum, 30(4), 787-804.
    https://doi.org/10.1007/s00291-007-0098-z
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))

_instance_path = os.path.join(_this_dir, "..", "instance.py")
_spec = importlib.util.spec_from_file_location("pm_instance_ig", _instance_path)
_pm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pm_instance_ig", _pm_instance)
_spec.loader.exec_module(_pm_instance)

ParallelMachineInstance = _pm_instance.ParallelMachineInstance
ParallelMachineSolution = _pm_instance.ParallelMachineSolution
compute_makespan = _pm_instance.compute_makespan
compute_machine_loads = _pm_instance.compute_machine_loads


def _load_module(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def iterated_greedy(
    instance: ParallelMachineInstance,
    max_iterations: int = 5000,
    d: int | None = None,
    temperature_factor: float = 0.4,
    time_limit: float | None = None,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """Solve parallel machine makespan problem using Iterated Greedy.

    Args:
        instance: A ParallelMachineInstance.
        max_iterations: Maximum number of iterations.
        d: Number of jobs to remove in destruction phase. Default: max(2, n//5).
        temperature_factor: Controls acceptance probability. Higher = more
            permissive. Temperature = T_factor * (sum_pj / (n * m * 10)).
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        ParallelMachineSolution with the best assignment found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    m = instance.m
    start_time = time.time()

    if d is None:
        d = max(2, n // 5)
    d = min(d, n)

    # Warm-start with LPT
    _lpt_mod = _load_module(
        "pm_lpt_ig", os.path.join(_this_dir, "..", "heuristics", "lpt.py")
    )
    lpt_sol = _lpt_mod.lpt(instance)
    assignment = [jobs[:] for jobs in lpt_sol.assignment]
    current_ms = compute_makespan(instance, assignment)

    best_assignment = [jobs[:] for jobs in assignment]
    best_ms = current_ms

    # Temperature for acceptance
    if instance.machine_type == "unrelated":
        avg_p = float(np.mean(instance.processing_times))
    else:
        avg_p = float(np.mean(instance.processing_times))
    temperature = temperature_factor * avg_p

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: remove d random jobs
        all_jobs = []
        for mach, jobs in enumerate(assignment):
            for job in jobs:
                all_jobs.append(job)

        if len(all_jobs) <= d:
            removed = all_jobs[:]
        else:
            indices = rng.choice(len(all_jobs), size=d, replace=False)
            removed = [all_jobs[i] for i in indices]

        # Remove selected jobs from assignment
        removed_set = set(removed)
        new_assignment = []
        for jobs in assignment:
            new_assignment.append([j for j in jobs if j not in removed_set])
            # Remove from set as we encounter them to handle duplicates correctly
        assignment_partial = new_assignment

        # Reconstruction: greedily reinsert removed jobs (LPT order)
        removed_sorted = sorted(
            removed,
            key=lambda j: max(
                instance.get_processing_time(j, mach) for mach in range(m)
            ),
            reverse=True,
        )

        for job in removed_sorted:
            # Find machine with minimum resulting load
            best_mach = 0
            best_load = float("inf")
            for mach in range(m):
                load = sum(
                    instance.get_processing_time(j, mach)
                    for j in assignment_partial[mach]
                ) + instance.get_processing_time(job, mach)
                if load < best_load:
                    best_load = load
                    best_mach = mach
            assignment_partial[best_mach].append(job)

        new_ms = compute_makespan(instance, assignment_partial)

        # Acceptance criterion (Boltzmann)
        delta = new_ms - current_ms
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / temperature)):
            assignment = assignment_partial
            current_ms = new_ms

            if current_ms < best_ms:
                best_ms = current_ms
                best_assignment = [jobs[:] for jobs in assignment]

    loads = compute_machine_loads(instance, best_assignment)
    return ParallelMachineSolution(
        assignment=best_assignment,
        makespan=best_ms,
        machine_loads=loads,
    )


if __name__ == "__main__":
    print("=== Iterated Greedy for Parallel Machines ===\n")

    inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
    print(f"Instance: {inst.n} jobs, {inst.m} machines")

    sol = iterated_greedy(inst, seed=42)
    print(f"IG  Makespan: {sol.makespan:.0f}")
    for i, jobs in enumerate(sol.assignment):
        print(f"  Machine {i}: {jobs} (load={sol.machine_loads[i]:.1f})")

    # Compare with LPT
    _lpt_mod = _load_module(
        "pm_lpt_ig_main", os.path.join(_this_dir, "..", "heuristics", "lpt.py")
    )
    lpt_sol = _lpt_mod.lpt(inst)
    print(f"\nLPT Makespan: {lpt_sol.makespan:.0f}")

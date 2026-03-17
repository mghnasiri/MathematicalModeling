"""
Tabu Search for Parallel Machine Makespan — Pm || Cmax

Implements a Tabu Search with relocate and swap neighborhoods for the
parallel machine makespan objective. Supports identical, uniform, and
unrelated machine environments.

Neighborhoods:
- Relocate: move a job from the most loaded machine to another
- Swap: exchange two jobs between different machines

Uses short-term memory preventing recently moved jobs from being moved
again. Aspiration criterion overrides tabu when a move yields a new
global best.

Warm-started with LPT heuristic.

Notation: Pm || Cmax (or Qm, Rm)
Complexity: O(iterations * n * m) per run.

References:
    Piersma, N. & Van Dijk, W. (1996). A local search heuristic for
    unrelated parallel machine scheduling with efficient neighborhood
    search. Mathematical and Computer Modelling, 24(9), 11-19.
    https://doi.org/10.1016/0895-7177(96)00150-2

    Dell'Amico, M. & Martello, S. (1995). Optimal scheduling of tasks
    on identical parallel processors. ORSA Journal on Computing, 7(2),
    191-200.
    https://doi.org/10.1287/ijoc.7.2.191
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("pm_instance_ts", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan
compute_machine_loads = _inst.compute_machine_loads

_lpt = _load_mod(
    "pm_lpt_ts",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt.lpt


def tabu_search(
    instance: ParallelMachineInstance,
    max_iterations: int = 2000,
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """Solve parallel machine makespan using Tabu Search.

    Args:
        instance: Parallel machine instance (Pm, Qm, or Rm).
        max_iterations: Maximum iterations.
        tabu_tenure: Iterations a move stays tabu. Default: sqrt(n).
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best ParallelMachineSolution found.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(3, int(n ** 0.5))

    # Initialize with LPT
    init_sol = lpt(instance)
    assignment = [list(a) for a in init_sol.assignment]

    current_ms = compute_makespan(instance, assignment)
    best_assignment = [list(a) for a in assignment]
    best_ms = current_ms

    # Tabu list: job -> iteration when tabu expires
    tabu_dict: dict[int, int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        loads = compute_machine_loads(instance, assignment)
        best_delta = float("inf")
        best_move = None

        # ── Relocate neighborhood ────────────────────────────────────────
        for mi in range(m):
            if not assignment[mi]:
                continue
            for ji in range(len(assignment[mi])):
                job = assignment[mi][ji]
                is_tabu = (
                    job in tabu_dict and tabu_dict[job] > iteration
                )

                for mj in range(m):
                    if mi == mj:
                        continue

                    # Compute new loads after moving job from mi to mj
                    pt_mi = instance.get_processing_time(job, mi)
                    pt_mj = instance.get_processing_time(job, mj)
                    new_load_mi = loads[mi] - pt_mi
                    new_load_mj = loads[mj] + pt_mj
                    new_ms = max(
                        max(loads[k] for k in range(m) if k != mi and k != mj),
                        new_load_mi, new_load_mj,
                    ) if m > 2 else max(new_load_mi, new_load_mj)

                    if m > 2:
                        other_max = max(
                            loads[k] for k in range(m)
                            if k != mi and k != mj
                        )
                        new_ms = max(other_max, new_load_mi, new_load_mj)
                    else:
                        new_ms = max(new_load_mi, new_load_mj)

                    delta = new_ms - current_ms

                    if is_tabu and current_ms + delta >= best_ms:
                        continue

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("relocate", mi, ji, mj, job)

        # ── Swap neighborhood ────────────────────────────────────────────
        for mi in range(m):
            if not assignment[mi]:
                continue
            for mj in range(mi + 1, m):
                if not assignment[mj]:
                    continue
                for ji in range(len(assignment[mi])):
                    job_i = assignment[mi][ji]
                    for jj in range(len(assignment[mj])):
                        job_j = assignment[mj][jj]

                        is_tabu = (
                            (job_i in tabu_dict
                             and tabu_dict[job_i] > iteration)
                            or (job_j in tabu_dict
                                and tabu_dict[job_j] > iteration)
                        )

                        pt_i_mi = instance.get_processing_time(job_i, mi)
                        pt_i_mj = instance.get_processing_time(job_i, mj)
                        pt_j_mi = instance.get_processing_time(job_j, mi)
                        pt_j_mj = instance.get_processing_time(job_j, mj)

                        new_load_mi = loads[mi] - pt_i_mi + pt_j_mi
                        new_load_mj = loads[mj] - pt_j_mj + pt_i_mj

                        if m > 2:
                            other_max = max(
                                loads[k] for k in range(m)
                                if k != mi and k != mj
                            )
                            new_ms = max(other_max, new_load_mi, new_load_mj)
                        else:
                            new_ms = max(new_load_mi, new_load_mj)

                        delta = new_ms - current_ms

                        if is_tabu and current_ms + delta >= best_ms:
                            continue

                        if delta < best_delta:
                            best_delta = delta
                            best_move = (
                                "swap", mi, ji, mj, jj, job_i, job_j,
                            )

        if best_move is None:
            tabu_dict.clear()
            continue

        # ── Apply move ───────────────────────────────────────────────────
        if best_move[0] == "relocate":
            _, mi, ji, mj, job = best_move
            assignment[mi].pop(ji)
            assignment[mj].append(job)
            tabu_dict[job] = iteration + tabu_tenure

        elif best_move[0] == "swap":
            _, mi, ji, mj, jj, job_i, job_j = best_move
            assignment[mi][ji] = job_j
            assignment[mj][jj] = job_i
            tabu_dict[job_i] = iteration + tabu_tenure
            tabu_dict[job_j] = iteration + tabu_tenure

        current_ms = compute_makespan(instance, assignment)

        if current_ms < best_ms:
            best_ms = current_ms
            best_assignment = [list(a) for a in assignment]

    loads = compute_machine_loads(instance, best_assignment)
    return ParallelMachineSolution(
        assignment=best_assignment,
        makespan=best_ms,
        machine_loads=loads,
    )


if __name__ == "__main__":
    inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
    print(f"Jobs: {inst.n}, Machines: {inst.m}")

    sol_lpt = lpt(inst)
    print(f"LPT: Cmax = {sol_lpt.makespan:.1f}")

    sol_ts = tabu_search(inst, seed=42)
    print(f"TS:  Cmax = {sol_ts.makespan:.1f}")

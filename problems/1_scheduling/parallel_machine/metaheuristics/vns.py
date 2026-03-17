"""
Variable Neighborhood Search for Parallel Machine Scheduling (Pm || Cmax).

VNS uses multiple neighborhood structures to escape local optima:
    N1: Move — transfer a job from the most loaded machine to the least loaded
    N2: Swap — exchange jobs between different machines
    N3: Multi-move — move k jobs simultaneously from overloaded machines

Local search uses best-improvement move on the bottleneck machine.
Warm-started with LPT heuristic.

Complexity: O(iterations * k_max * n * m) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Hansen, P., Mladenović, N. & Pérez, J.A.M. (2010). Variable
    neighbourhood search: methods and applications. Annals of Operations
    Research, 175(1), 367-407.
    https://doi.org/10.1007/s10479-009-0657-6
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


_inst = _load_mod("pm_instance_vns", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan
compute_machine_loads = _inst.compute_machine_loads

_lpt = _load_mod(
    "pm_lpt_vns",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt.lpt


def vns(
    instance: ParallelMachineInstance,
    max_iterations: int = 500,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """Solve Pm||Cmax using Variable Neighborhood Search.

    Args:
        instance: A ParallelMachineInstance.
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        ParallelMachineSolution with the best assignment found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Warm-start
    init_sol = lpt(instance)
    assignment = [list(m_jobs) for m_jobs in init_sol.assignment]
    loads = compute_machine_loads(instance, assignment)
    current_ms = max(loads) if loads else 0.0

    best_assignment = [m[:] for m in assignment]
    best_ms = current_ms

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = [m[:] for m in assignment]
            _shake(instance, shaken, k, rng)

            # Local search
            _best_improvement_ls(instance, shaken)
            shaken_loads = compute_machine_loads(instance, shaken)
            shaken_ms = max(shaken_loads) if shaken_loads else 0.0

            if shaken_ms < current_ms - 1e-10:
                assignment = shaken
                loads = shaken_loads
                current_ms = shaken_ms
                k = 1

                if current_ms < best_ms - 1e-10:
                    best_ms = current_ms
                    best_assignment = [m[:] for m in assignment]
            else:
                k += 1

    return ParallelMachineSolution(
        assignment=best_assignment,
        makespan=best_ms,
        machine_loads=compute_machine_loads(instance, best_assignment),
    )


def _shake(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
    k: int,
    rng: np.random.Generator,
) -> None:
    """Shake: perform k random moves."""
    for _ in range(k):
        non_empty = [i for i in range(len(assignment)) if assignment[i]]
        if not non_empty:
            break
        src = non_empty[rng.integers(len(non_empty))]
        if not assignment[src]:
            continue
        idx = rng.integers(len(assignment[src]))
        job = assignment[src].pop(idx)
        dst = rng.integers(instance.m)
        assignment[dst].append(job)


def _best_improvement_ls(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
) -> None:
    """Best-improvement local search: move/swap on bottleneck."""
    improved = True
    while improved:
        improved = False
        loads = compute_machine_loads(instance, assignment)
        current_ms = max(loads) if loads else 0.0
        bottleneck = int(np.argmax(loads))

        best_delta = 0.0
        best_move = None

        # Move: from bottleneck to any other
        for idx, job in enumerate(assignment[bottleneck]):
            pt_bn = instance.get_processing_time(job, bottleneck)
            for mi in range(instance.m):
                if mi == bottleneck:
                    continue
                pt_mi = instance.get_processing_time(job, mi)
                new_bn = loads[bottleneck] - pt_bn
                new_mi = loads[mi] + pt_mi
                new_ms = max(new_bn, new_mi)
                for mj in range(instance.m):
                    if mj != bottleneck and mj != mi:
                        new_ms = max(new_ms, loads[mj])
                delta = new_ms - current_ms
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_move = ("move", idx, bottleneck, mi)

        # Swap: bottleneck with any other
        for idx_bn, job_bn in enumerate(assignment[bottleneck]):
            for mi in range(instance.m):
                if mi == bottleneck:
                    continue
                for idx_mi, job_mi in enumerate(assignment[mi]):
                    pt_bn_on_bn = instance.get_processing_time(job_bn, bottleneck)
                    pt_mi_on_mi = instance.get_processing_time(job_mi, mi)
                    pt_bn_on_mi = instance.get_processing_time(job_bn, mi)
                    pt_mi_on_bn = instance.get_processing_time(job_mi, bottleneck)

                    new_bn = loads[bottleneck] - pt_bn_on_bn + pt_mi_on_bn
                    new_mi = loads[mi] - pt_mi_on_mi + pt_bn_on_mi
                    new_ms = max(new_bn, new_mi)
                    for mj in range(instance.m):
                        if mj != bottleneck and mj != mi:
                            new_ms = max(new_ms, loads[mj])
                    delta = new_ms - current_ms
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_move = ("swap", idx_bn, bottleneck, idx_mi, mi)

        if best_move is not None:
            if best_move[0] == "move":
                _, idx, src, dst = best_move
                job = assignment[src].pop(idx)
                assignment[dst].append(job)
            else:
                _, idx_s, src, idx_d, dst = best_move
                assignment[src][idx_s], assignment[dst][idx_d] = (
                    assignment[dst][idx_d], assignment[src][idx_s])
            improved = True


if __name__ == "__main__":
    inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
    print(f"Parallel Machine: {inst.n} jobs, {inst.m} machines")

    lpt_sol = lpt(inst)
    print(f"LPT: makespan={lpt_sol.makespan:.1f}")

    vns_sol = vns(inst, seed=42)
    print(f"VNS: makespan={vns_sol.makespan:.1f}")

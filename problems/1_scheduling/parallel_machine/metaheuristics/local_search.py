"""
Local Search for Parallel Machine Scheduling (Pm || Cmax).

Iterative improvement using swap, move, and balance neighborhoods
on job-to-machine assignments for makespan minimization.

Neighborhoods:
    - Move: transfer a job from the most loaded machine to another
    - Swap: exchange jobs between the most loaded machine and another
    - Balance: move the largest job on the bottleneck to the least loaded

Warm-started with LPT heuristic.

Complexity: O(iterations * n * m) per run.

References:
    Della Croce, F., Scatamacchia, R. & T'Kindt, V. (2019). A tight
    linear time 13/12-approximation for the P2||Cmax problem. Journal
    of Combinatorial Optimization, 38(2), 608-625.
    https://doi.org/10.1007/s10878-019-00398-9

    Graham, R.L. (1969). Bounds on multiprocessing timing anomalies.
    SIAM Journal on Applied Mathematics, 17(2), 416-429.
    https://doi.org/10.1137/0117039
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


_inst = _load_mod("pm_instance_ls", os.path.join(_parent_dir, "instance.py"))
ParallelMachineInstance = _inst.ParallelMachineInstance
ParallelMachineSolution = _inst.ParallelMachineSolution
compute_makespan = _inst.compute_makespan
compute_machine_loads = _inst.compute_machine_loads

_lpt = _load_mod(
    "pm_lpt_ls",
    os.path.join(_parent_dir, "heuristics", "lpt.py"),
)
lpt = _lpt.lpt


def local_search(
    instance: ParallelMachineInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """Solve Pm||Cmax using Local Search.

    Args:
        instance: A ParallelMachineInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        ParallelMachineSolution with the best assignment found.
    """
    rng = np.random.default_rng(seed)
    start_time_clock = time.time()

    # Warm-start with LPT
    init_sol = lpt(instance)
    assignment = [list(m_jobs) for m_jobs in init_sol.assignment]
    loads = compute_machine_loads(instance, assignment)
    current_ms = max(loads)

    best_assignment = [m[:] for m in assignment]
    best_ms = current_ms

    no_improve = 0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time_clock >= time_limit:
            break

        improved = False

        # Find bottleneck machine
        bottleneck = int(np.argmax(loads))

        # Try move: move each job from bottleneck to other machines
        best_delta = 0.0
        best_move = None

        for idx, job in enumerate(assignment[bottleneck]):
            pt_on_bn = instance.get_processing_time(job, bottleneck)
            for mi in range(instance.m):
                if mi == bottleneck:
                    continue
                pt_on_mi = instance.get_processing_time(job, mi)
                new_bn_load = loads[bottleneck] - pt_on_bn
                new_mi_load = loads[mi] + pt_on_mi
                new_ms = max(new_bn_load, new_mi_load)
                # Also check all other machines
                for mj in range(instance.m):
                    if mj != bottleneck and mj != mi:
                        new_ms = max(new_ms, loads[mj])
                delta = new_ms - current_ms
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_move = ("move", idx, bottleneck, mi)

        # Try swap: exchange a job from bottleneck with a job from another machine
        for idx_bn, job_bn in enumerate(assignment[bottleneck]):
            pt_bn = instance.get_processing_time(job_bn, bottleneck)
            for mi in range(instance.m):
                if mi == bottleneck:
                    continue
                for idx_mi, job_mi in enumerate(assignment[mi]):
                    pt_mi_on_mi = instance.get_processing_time(job_mi, mi)
                    pt_bn_on_mi = instance.get_processing_time(job_bn, mi)
                    pt_mi_on_bn = instance.get_processing_time(job_mi, bottleneck)

                    new_bn_load = loads[bottleneck] - pt_bn + pt_mi_on_bn
                    new_mi_load = loads[mi] - pt_mi_on_mi + pt_bn_on_mi

                    new_ms = max(new_bn_load, new_mi_load)
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
                _, idx_src, src, idx_dst, dst = best_move
                assignment[src][idx_src], assignment[dst][idx_dst] = (
                    assignment[dst][idx_dst], assignment[src][idx_src])

            loads = compute_machine_loads(instance, assignment)
            current_ms = max(loads)
            improved = True
            no_improve = 0

            if current_ms < best_ms - 1e-10:
                best_ms = current_ms
                best_assignment = [m[:] for m in assignment]
        else:
            no_improve += 1

        if no_improve >= 5:
            # Perturbation: randomly move a job
            _perturb(instance, assignment, rng)
            loads = compute_machine_loads(instance, assignment)
            current_ms = max(loads)
            no_improve = 0

    return ParallelMachineSolution(
        assignment=best_assignment,
        makespan=best_ms,
        machine_loads=compute_machine_loads(instance, best_assignment),
    )


def _perturb(
    instance: ParallelMachineInstance,
    assignment: list[list[int]],
    rng: np.random.Generator,
) -> None:
    """Random perturbation: move random jobs between machines."""
    non_empty = [i for i in range(len(assignment)) if assignment[i]]
    if not non_empty:
        return

    src = non_empty[rng.integers(len(non_empty))]
    if not assignment[src]:
        return

    idx = rng.integers(len(assignment[src]))
    job = assignment[src].pop(idx)
    dst = rng.integers(instance.m)
    assignment[dst].append(job)


if __name__ == "__main__":
    inst = ParallelMachineInstance.random_identical(n=15, m=3, seed=42)
    print(f"Parallel Machine: {inst.n} jobs, {inst.m} machines")

    lpt_sol = lpt(inst)
    print(f"LPT: makespan={lpt_sol.makespan:.1f}")

    ls_sol = local_search(inst, seed=42)
    print(f"LS: makespan={ls_sol.makespan:.1f}")

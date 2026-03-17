"""
Iterated Greedy for Job Shop Scheduling.

Problem notation: Jm || Cmax

Iterated Greedy repeatedly removes a subset of jobs from the machine
sequences and reinserts them using a greedy dispatching decoder.
A Boltzmann acceptance criterion allows escaping local optima.

Warm-started with best dispatching rule (SPT/LPT/MWR).

Complexity: O(iterations * d * n * m) where d = destruction size.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Nawaz, M., Enscore, E.E. & Ham, I. (1983). A heuristic algorithm
    for the m-machine, n-job flow-shop sequencing problem. Omega,
    11(1), 91-95.
    https://doi.org/10.1016/0305-0483(83)90088-9
"""

from __future__ import annotations

import sys
import os
import math
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


_inst = _load_mod("jsp_instance_ig", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution

_disp = _load_mod(
    "jsp_disp_ig",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def iterated_greedy(
    instance: JobShopInstance,
    max_iterations: int = 2000,
    d: int | None = None,
    temperature_factor: float = 0.3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> JobShopSolution:
    """Solve Job Shop using Iterated Greedy.

    Args:
        instance: A JobShopInstance.
        max_iterations: Maximum number of iterations.
        d: Number of jobs to remove and reinsert. Default: max(2, n//3).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        JobShopSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    m = instance.m
    start_time_clock = time.time()

    if d is None:
        d = max(2, n // 3)
    d = min(d, n)

    # Warm-start: try multiple dispatching rules
    best_sol = None
    for rule in ["spt", "lpt", "mwr"]:
        sol = dispatching_rule(instance, rule=rule)
        if best_sol is None or sol.makespan < best_sol.makespan:
            best_sol = sol

    # Extract job order per machine from solution
    current_machine_seqs = _extract_machine_sequences(instance, best_sol)
    current_ms = best_sol.makespan
    best_ms = current_ms
    best_result = best_sol

    # Temperature
    avg_pt = np.mean([pt for job in instance.jobs for _, pt in job])
    temperature = temperature_factor * avg_pt

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time_clock >= time_limit:
            break

        # Destruction: remove d random jobs from machine sequences
        removed_jobs = set(rng.choice(n, size=d, replace=False).tolist())

        partial_seqs = {}
        for mach, seq in current_machine_seqs.items():
            partial_seqs[mach] = [(j, k) for j, k in seq if j not in removed_jobs]

        # Reconstruction: greedily reinsert removed jobs
        new_seqs = _reconstruct(instance, partial_seqs, removed_jobs, rng)

        # Decode to start times
        start_times = _decode_sequences(instance, new_seqs)
        new_ms = compute_makespan(instance, start_times)

        # Acceptance
        delta = new_ms - current_ms
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / max(temperature, 1e-10))):
            current_machine_seqs = new_seqs
            current_ms = new_ms

            if current_ms < best_ms:
                best_ms = current_ms
                best_result = JobShopSolution(
                    start_times=start_times,
                    makespan=current_ms,
                    machine_sequences=new_seqs,
                )

    return best_result


def _extract_machine_sequences(
    instance: JobShopInstance, sol: JobShopSolution
) -> dict[int, list[tuple[int, int]]]:
    """Extract machine sequences from solution."""
    if sol.machine_sequences:
        return {m: ops[:] for m, ops in sol.machine_sequences.items()}

    machine_ops: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for j in range(instance.n):
        for k, (mach, _) in enumerate(instance.jobs[j]):
            if (j, k) in sol.start_times:
                machine_ops.setdefault(mach, []).append(
                    (sol.start_times[(j, k)], (j, k))
                )
    return {
        m: [op for _, op in sorted(ops)]
        for m, ops in machine_ops.items()
    }


def _reconstruct(
    instance: JobShopInstance,
    partial_seqs: dict[int, list[tuple[int, int]]],
    removed_jobs: set[int],
    rng: np.random.Generator,
) -> dict[int, list[tuple[int, int]]]:
    """Greedily reinsert removed jobs into machine sequences."""
    seqs = {m: ops[:] for m, ops in partial_seqs.items()}

    # Process removed jobs in random order
    job_list = sorted(removed_jobs)
    rng.shuffle(job_list)

    for job in job_list:
        for k, (mach, pt) in enumerate(instance.jobs[job]):
            if mach not in seqs:
                seqs[mach] = []
            # Insert at the end (greedy — fast)
            seqs[mach].append((job, k))

    return seqs


def _decode_sequences(
    instance: JobShopInstance,
    machine_seqs: dict[int, list[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    """Decode machine sequences to start times."""
    n, m = instance.n, instance.m
    start_times: dict[tuple[int, int], int] = {}
    machine_time = [0] * m
    job_time = [0] * n

    # Build schedule by processing machines in a multi-pass approach
    total_ops = instance.total_operations()
    scheduled = set()
    max_passes = total_ops + 1

    for _ in range(max_passes):
        if len(scheduled) >= total_ops:
            break
        for mach in range(m):
            seq = machine_seqs.get(mach, [])
            for j, k in seq:
                if (j, k) in scheduled:
                    continue
                # Check if predecessor in job is scheduled
                if k > 0 and (j, k - 1) not in scheduled:
                    continue
                # Schedule
                _, pt = instance.jobs[j][k]
                pred_end = 0
                if k > 0 and (j, k - 1) in start_times:
                    _, prev_pt = instance.jobs[j][k - 1]
                    pred_end = start_times[(j, k - 1)] + prev_pt
                start = max(machine_time[mach], pred_end)
                start_times[(j, k)] = start
                machine_time[mach] = start + pt
                scheduled.add((j, k))

    return start_times


if __name__ == "__main__":
    from instance import ft06

    inst = ft06()
    print(f"ft06: {inst.n} jobs, {inst.m} machines")

    sol = iterated_greedy(inst, seed=42)
    print(f"IG: makespan={sol.makespan}")

    disp_sol = dispatching_rule(inst, rule="spt")
    print(f"SPT: makespan={disp_sol.makespan}")

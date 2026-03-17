"""
Simulated Annealing for Parallel Machine Scheduling — Pm || Cmax

Implements SA with move/swap neighborhoods for the parallel machine
makespan problem. Supports identical, uniform, and unrelated machines.

Neighborhoods:
    1. Relocate: move a random job from the most loaded machine to a
       random other machine.
    2. Swap: swap a random job from the most loaded machine with a
       random job from another machine.

The initial solution comes from LPT (Longest Processing Time) heuristic.
Temperature is auto-calibrated based on initial makespan.

Notation: Pm || Cmax (extends to Qm, Rm)
Complexity: O(iterations * m) per iteration.

Reference:
    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by Simulated Annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671

    Weng, M.X., Lu, J. & Ren, H. (2001). Unrelated parallel machine
    scheduling with setup consideration and a total weighted completion
    time objective. International Journal of Production Economics,
    70(3), 215-226.
    https://doi.org/10.1016/S0925-5273(00)00066-9
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
_spec = importlib.util.spec_from_file_location("pm_instance_sa", _instance_path)
_pm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pm_instance_sa", _pm_instance)
_spec.loader.exec_module(_pm_instance)

ParallelMachineInstance = _pm_instance.ParallelMachineInstance
ParallelMachineSolution = _pm_instance.ParallelMachineSolution
compute_makespan = _pm_instance.compute_makespan
compute_machine_loads = _pm_instance.compute_machine_loads

_lpt_path = os.path.join(_this_dir, "..", "heuristics", "lpt.py")
_spec2 = importlib.util.spec_from_file_location("pm_lpt_sa", _lpt_path)
_pm_lpt = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("pm_lpt_sa", _pm_lpt)
_spec2.loader.exec_module(_pm_lpt)

lpt = _pm_lpt.lpt


def simulated_annealing(
    instance: ParallelMachineInstance,
    max_iterations: int = 5000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """Simulated Annealing for parallel machine makespan.

    Args:
        instance: Parallel machine instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best ParallelMachineSolution found.
    """
    rng = np.random.default_rng(seed)
    n, m_count = instance.n, instance.m
    start_time_clock = time.time()

    # ── Initial solution from LPT ────────────────────────────────────────
    lpt_sol = lpt(instance)
    current = [list(machine_jobs) for machine_jobs in lpt_sol.assignment]
    current_ms = lpt_sol.makespan

    best = [list(mj) for mj in current]
    best_ms = current_ms

    if initial_temp is None:
        initial_temp = max(1.0, current_ms * 0.05)

    temp = initial_temp

    def _compute_loads(assignment):
        loads = []
        for i, jobs in enumerate(assignment):
            load = sum(instance.get_processing_time(j, i) for j in jobs)
            loads.append(load)
        return loads

    def _get_makespan(loads):
        return max(loads) if loads else 0.0

    current_loads = _compute_loads(current)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time_clock >= time_limit:
            break

        # Find most loaded machine
        max_load_machine = int(np.argmax(current_loads))

        if not current[max_load_machine]:
            temp *= cooling_rate
            continue

        # Choose neighborhood
        if rng.random() < 0.5 or m_count == 1:
            # Relocate: move a job from max-loaded to random other machine
            other_machines = [i for i in range(m_count) if i != max_load_machine]
            if not other_machines:
                temp *= cooling_rate
                continue
            target_machine = rng.choice(other_machines)
            job_idx = rng.integers(0, len(current[max_load_machine]))
            job = current[max_load_machine][job_idx]

            # Apply move
            new_assignment = [list(mj) for mj in current]
            new_assignment[max_load_machine].pop(job_idx)
            new_assignment[target_machine].append(job)
            move_type = "relocate"
        else:
            # Swap: swap a job on max-loaded with a job on another machine
            other_machines = [
                i for i in range(m_count)
                if i != max_load_machine and current[i]
            ]
            if not other_machines:
                temp *= cooling_rate
                continue
            target_machine = rng.choice(other_machines)
            job_idx1 = rng.integers(0, len(current[max_load_machine]))
            job_idx2 = rng.integers(0, len(current[target_machine]))

            new_assignment = [list(mj) for mj in current]
            new_assignment[max_load_machine][job_idx1], new_assignment[target_machine][job_idx2] = (
                new_assignment[target_machine][job_idx2],
                new_assignment[max_load_machine][job_idx1],
            )
            move_type = "swap"

        new_loads = _compute_loads(new_assignment)
        new_ms = _get_makespan(new_loads)

        delta = new_ms - current_ms

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current = new_assignment
            current_ms = new_ms
            current_loads = new_loads

            if current_ms < best_ms:
                best = [list(mj) for mj in current]
                best_ms = current_ms

        temp *= cooling_rate

    machine_loads = _compute_loads(best)
    return ParallelMachineSolution(
        assignment=best,
        makespan=best_ms,
        machine_loads=machine_loads,
    )


if __name__ == "__main__":
    inst = ParallelMachineInstance.random_identical(n=20, m=4, seed=42)
    print(f"Instance: {inst.n} jobs, {inst.m} machines")

    lpt_sol = lpt(inst)
    print(f"LPT: makespan = {lpt_sol.makespan:.1f}")

    sa_sol = simulated_annealing(inst, max_iterations=5000, seed=42)
    print(f"SA:  makespan = {sa_sol.makespan:.1f}")

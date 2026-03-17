"""
Iterated Greedy for Single Machine Scheduling.

Problem notation: 1 || ΣwjTj and 1 || ΣTj

Iterated Greedy repeatedly destroys the current solution by removing a
subset of jobs, then reconstructs by greedily reinserting them in the
best position. A Boltzmann-based acceptance criterion allows escaping
local optima.

Warm-started with ATC (for weighted tardiness) or EDD (for total tardiness).

Complexity: O(iterations * d * n) where d = destruction size.

References:
    Ruiz, R. & Stützle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound
    algorithm for the total weighted tardiness problem. Operations
    Research, 33(2), 363-377.
    https://doi.org/10.1287/opre.33.2.363
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


_inst = _load_mod("sm_instance_ig", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness


def iterated_greedy(
    instance: SingleMachineInstance,
    objective: str = "weighted_tardiness",
    max_iterations: int = 5000,
    d: int | None = None,
    temperature_factor: float = 0.5,
    time_limit: float | None = None,
    seed: int | None = None,
) -> SingleMachineSolution:
    """Solve single machine scheduling using Iterated Greedy.

    Args:
        instance: A SingleMachineInstance.
        objective: "weighted_tardiness" or "total_tardiness".
        max_iterations: Maximum number of iterations.
        d: Number of jobs to remove. Default: max(2, n//4).
        temperature_factor: Controls acceptance probability.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        SingleMachineSolution with the best sequence found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    if d is None:
        d = max(2, n // 4)
    d = min(d, n)

    eval_fn = (compute_weighted_tardiness if objective == "weighted_tardiness"
               else compute_total_tardiness)

    # Warm-start
    sequence = _warm_start(instance, objective)
    current_obj = eval_fn(instance, sequence)

    best_sequence = sequence[:]
    best_obj = current_obj

    # Temperature
    avg_p = float(np.mean(instance.processing_times))
    temperature = temperature_factor * avg_p

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Destruction: remove d random jobs
        indices = rng.choice(n, size=d, replace=False)
        removed = [sequence[i] for i in sorted(indices, reverse=True)]
        partial = [j for j in sequence if j not in set(removed)]

        # Reconstruction: greedily reinsert each removed job
        for job in removed:
            best_pos = 0
            best_val = float("inf")
            for pos in range(len(partial) + 1):
                trial = partial[:pos] + [job] + partial[pos:]
                val = eval_fn(instance, trial)
                if val < best_val:
                    best_val = val
                    best_pos = pos
            partial.insert(best_pos, job)

        new_obj = eval_fn(instance, partial)

        # Acceptance
        delta = new_obj - current_obj
        if delta < 0 or (temperature > 0 and
                         rng.random() < math.exp(-delta / temperature)):
            sequence = partial
            current_obj = new_obj

            if current_obj < best_obj:
                best_obj = current_obj
                best_sequence = sequence[:]

    return SingleMachineSolution(
        sequence=best_sequence,
        objective_value=eval_fn(instance, best_sequence),
    )


def _warm_start(instance: SingleMachineInstance, objective: str) -> list[int]:
    """Generate initial solution using appropriate heuristic."""
    if objective == "weighted_tardiness":
        _atc_mod = _load_mod(
            "sm_atc_ig",
            os.path.join(_parent_dir, "heuristics", "apparent_tardiness_cost.py"),
        )
        sol = _atc_mod.atc(instance)
        return sol.sequence
    else:
        # EDD for total tardiness
        _disp_mod = _load_mod(
            "sm_disp_ig",
            os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
        )
        sol = _disp_mod.edd(instance)
        return sol.sequence


if __name__ == "__main__":
    inst = SingleMachineInstance.random(n=15, seed=42)
    print(f"Single Machine: {inst.n} jobs")

    sol_wt = iterated_greedy(inst, objective="weighted_tardiness", seed=42)
    print(f"IG (weighted tardiness): {sol_wt.objective_value:.1f}")

    sol_tt = iterated_greedy(inst, objective="total_tardiness", seed=42)
    print(f"IG (total tardiness): {sol_tt.objective_value:.1f}")

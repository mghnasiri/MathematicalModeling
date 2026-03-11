"""
Variable Neighborhood Search for Single Machine Scheduling.

Problem notation: 1 || ΣwjTj and 1 || ΣTj

VNS uses multiple neighborhood structures:
    N1: Swap — exchange two adjacent jobs
    N2: Insert — move a job to a different position
    N3: Multi-swap — swap two non-adjacent jobs

Local search uses best-improvement swap.
Warm-started with ATC (weighted tardiness) or EDD (total tardiness).

Complexity: O(iterations * k_max * n^2) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound
    algorithm for the total weighted tardiness problem. Operations
    Research, 33(2), 363-377.
    https://doi.org/10.1287/opre.33.2.363
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


_inst = _load_mod("sm_instance_vns", os.path.join(_parent_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness


def vns(
    instance: SingleMachineInstance,
    objective: str = "weighted_tardiness",
    max_iterations: int = 1000,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> SingleMachineSolution:
    """Solve single machine scheduling using VNS.

    Args:
        instance: A SingleMachineInstance.
        objective: "weighted_tardiness" or "total_tardiness".
        max_iterations: Maximum number of iterations.
        k_max: Maximum neighborhood size.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        SingleMachineSolution with the best sequence found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    eval_fn = (compute_weighted_tardiness if objective == "weighted_tardiness"
               else compute_total_tardiness)

    # Warm-start
    sequence = _warm_start(instance, objective)
    current_obj = eval_fn(instance, sequence)

    best_sequence = sequence[:]
    best_obj = current_obj

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # Shaking
            shaken = sequence[:]
            _shake(shaken, k, rng)

            # Local search
            shaken, shaken_obj = _best_improvement_swap(instance, shaken, eval_fn)

            if shaken_obj < current_obj - 1e-10:
                sequence = shaken
                current_obj = shaken_obj
                k = 1

                if current_obj < best_obj - 1e-10:
                    best_obj = current_obj
                    best_sequence = sequence[:]
            else:
                k += 1

    return SingleMachineSolution(
        sequence=best_sequence,
        objective_value=eval_fn(instance, best_sequence),
    )


def _shake(sequence: list[int], k: int, rng: np.random.Generator) -> None:
    """Shake: perform k random perturbations."""
    n = len(sequence)
    if n < 2:
        return
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        sequence[i], sequence[j] = sequence[j], sequence[i]


def _best_improvement_swap(
    instance: SingleMachineInstance,
    sequence: list[int],
    eval_fn,
) -> tuple[list[int], float]:
    """Best-improvement swap local search."""
    n = len(sequence)
    improved = True
    current_obj = eval_fn(instance, sequence)

    while improved:
        improved = False
        best_delta = 0
        best_pair = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                sequence[i], sequence[j] = sequence[j], sequence[i]
                new_obj = eval_fn(instance, sequence)
                delta = new_obj - current_obj
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_pair = (i, j)
                sequence[i], sequence[j] = sequence[j], sequence[i]

        if best_pair is not None:
            i, j = best_pair
            sequence[i], sequence[j] = sequence[j], sequence[i]
            current_obj += best_delta
            improved = True

    return sequence, current_obj


def _warm_start(instance: SingleMachineInstance, objective: str) -> list[int]:
    """Generate initial solution using appropriate heuristic."""
    if objective == "weighted_tardiness":
        _atc_mod = _load_mod(
            "sm_atc_vns",
            os.path.join(_parent_dir, "heuristics", "apparent_tardiness_cost.py"),
        )
        sol = _atc_mod.atc(instance)
        return sol.sequence
    else:
        _disp_mod = _load_mod(
            "sm_disp_vns",
            os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
        )
        sol = _disp_mod.edd(instance)
        return sol.sequence


if __name__ == "__main__":
    inst = SingleMachineInstance.random(n=15, seed=42)
    print(f"Single Machine: {inst.n} jobs")

    sol_wt = vns(inst, objective="weighted_tardiness", seed=42)
    print(f"VNS (weighted tardiness): {sol_wt.objective_value:.1f}")

    sol_tt = vns(inst, objective="total_tardiness", seed=42)
    print(f"VNS (total tardiness): {sol_tt.objective_value:.1f}")

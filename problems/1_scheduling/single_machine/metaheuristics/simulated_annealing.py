"""
Simulated Annealing for Single Machine Weighted Tardiness — 1 || ΣwjTj

Implements SA with swap and insertion neighborhoods for the weighted
tardiness objective. The algorithm starts from an ATC heuristic solution
and explores the neighborhood using a cooling schedule.

The SA uses a geometric cooling schedule with epoch-based temperature
reduction. Each epoch evaluates a fixed number of neighbors before
decreasing the temperature.

Reference: Potts, C.N. & Van Wassenhove, L.N. (1991). "Single Machine
           Tardiness Sequencing Heuristics"
           IIE Transactions, 23(4):346-354.
           DOI: 10.1080/07408179108963868

           Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). "Optimization
           by Simulated Annealing"
           Science, 220(4598):671-680.
           DOI: 10.1126/science.220.4598.671
"""

from __future__ import annotations
import sys
import os
import math
import time
import importlib.util
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)

_instance_path = os.path.join(_sm_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_weighted_tardiness = _sm_instance.compute_weighted_tardiness
compute_total_tardiness = _sm_instance.compute_total_tardiness

_atc_path = os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py")
_spec2 = importlib.util.spec_from_file_location("sm_atc", _atc_path)
_sm_atc = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("sm_atc", _sm_atc)
_spec2.loader.exec_module(_sm_atc)

_rules_path = os.path.join(_sm_dir, "heuristics", "dispatching_rules.py")
_spec3 = importlib.util.spec_from_file_location("sm_dispatching", _rules_path)
_sm_rules = importlib.util.module_from_spec(_spec3)
sys.modules.setdefault("sm_dispatching", _sm_rules)
_spec3.loader.exec_module(_sm_rules)

atc = _sm_atc.atc
edd = _sm_rules.edd


def simulated_annealing_wt(
    instance: SingleMachineInstance,
    cooling_rate: float = 0.995,
    initial_temp_factor: float = 0.5,
    epoch_length: int | None = None,
    time_limit: float | None = None,
    max_iterations: int = 50000,
    seed: int | None = None,
) -> SingleMachineSolution:
    """
    Simulated Annealing for 1 || ΣwjTj.

    Uses mixed swap/insertion neighborhood with geometric cooling.
    Warm-started with ATC heuristic.

    Args:
        instance: A SingleMachineInstance (must have weights and due dates).
        cooling_rate: Temperature reduction factor per epoch.
        initial_temp_factor: Controls initial temperature.
        epoch_length: Iterations per temperature level. Default: n².
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of iterations.
        seed: Random seed for reproducibility.

    Returns:
        SingleMachineSolution with best ΣwjTj found.
    """
    assert instance.due_dates is not None, "SA requires due dates"

    rng = np.random.default_rng(seed)
    n = instance.n

    if epoch_length is None:
        epoch_length = n * n

    # Initial solution via ATC
    initial = atc(instance)
    current_seq = list(initial.sequence)
    current_obj = compute_weighted_tardiness(instance, current_seq)

    best_seq = list(current_seq)
    best_obj = current_obj

    # Temperature based on average weighted tardiness
    total_p = float(instance.processing_times.sum())
    temperature = initial_temp_factor * total_p / n

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        for _ in range(epoch_length):
            if iteration >= max_iterations:
                break
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            iteration += 1

            # Generate neighbor: 50% swap, 50% insertion
            neighbor = list(current_seq)
            if rng.random() < 0.5:
                # Swap
                i, j = rng.choice(n, size=2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                # Insertion
                i = rng.integers(0, n)
                job = neighbor.pop(i)
                j = rng.integers(0, n)
                neighbor.insert(j, job)

            neighbor_obj = compute_weighted_tardiness(instance, neighbor)
            delta = neighbor_obj - current_obj

            if delta <= 0:
                current_seq = neighbor
                current_obj = neighbor_obj
            elif temperature > 0:
                if rng.random() < math.exp(-delta / temperature):
                    current_seq = neighbor
                    current_obj = neighbor_obj

            if current_obj < best_obj:
                best_seq = list(current_seq)
                best_obj = current_obj

        temperature *= cooling_rate

        if temperature < 1e-10:
            break

    return SingleMachineSolution(
        sequence=best_seq,
        objective_value=best_obj,
        objective_name="ΣwjTj",
    )


def simulated_annealing_tt(
    instance: SingleMachineInstance,
    cooling_rate: float = 0.995,
    initial_temp_factor: float = 0.5,
    epoch_length: int | None = None,
    time_limit: float | None = None,
    max_iterations: int = 50000,
    seed: int | None = None,
) -> SingleMachineSolution:
    """
    Simulated Annealing for 1 || ΣTj (total tardiness).

    Same framework as SA for ΣwjTj but uses EDD as initial solution
    and optimizes total (unweighted) tardiness.

    Args:
        instance: A SingleMachineInstance (must have due dates).
        cooling_rate: Temperature reduction factor per epoch.
        initial_temp_factor: Controls initial temperature.
        epoch_length: Iterations per temperature level. Default: n².
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of iterations.
        seed: Random seed for reproducibility.

    Returns:
        SingleMachineSolution with best ΣTj found.
    """
    assert instance.due_dates is not None, "SA requires due dates"

    rng = np.random.default_rng(seed)
    n = instance.n

    if epoch_length is None:
        epoch_length = n * n

    # Initial solution via EDD
    initial = edd(instance)
    current_seq = list(initial.sequence)
    current_obj = compute_total_tardiness(instance, current_seq)

    best_seq = list(current_seq)
    best_obj = current_obj

    total_p = float(instance.processing_times.sum())
    temperature = initial_temp_factor * total_p / n

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        for _ in range(epoch_length):
            if iteration >= max_iterations:
                break
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            iteration += 1

            neighbor = list(current_seq)
            if rng.random() < 0.5:
                i, j = rng.choice(n, size=2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                i = rng.integers(0, n)
                job = neighbor.pop(i)
                j = rng.integers(0, n)
                neighbor.insert(j, job)

            neighbor_obj = compute_total_tardiness(instance, neighbor)
            delta = neighbor_obj - current_obj

            if delta <= 0:
                current_seq = neighbor
                current_obj = neighbor_obj
            elif temperature > 0:
                if rng.random() < math.exp(-delta / temperature):
                    current_seq = neighbor
                    current_obj = neighbor_obj

            if current_obj < best_obj:
                best_seq = list(current_seq)
                best_obj = current_obj

        temperature *= cooling_rate

        if temperature < 1e-10:
            break

    return SingleMachineSolution(
        sequence=best_seq,
        objective_value=best_obj,
        objective_name="ΣTj",
    )


if __name__ == "__main__":
    print("=== SA for Single Machine ===\n")

    inst = SingleMachineInstance.random(n=20, seed=42)
    print(f"Instance: {inst.n} jobs")

    sol_atc = atc(inst)
    print(f"ATC  ΣwjTj: {sol_atc.objective_value}")

    sol_sa = simulated_annealing_wt(inst, max_iterations=10000, seed=42)
    print(f"SA   ΣwjTj: {sol_sa.objective_value}")

    if sol_atc.objective_value > 0:
        improvement = (sol_atc.objective_value - sol_sa.objective_value) / sol_atc.objective_value * 100
        print(f"Improvement: {improvement:.1f}%")

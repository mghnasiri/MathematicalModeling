"""
Tabu Search for Single Machine Weighted Tardiness — 1 || ΣwjTj

Implements a Tabu Search with swap and insertion neighborhoods for the
weighted tardiness objective. Uses short-term memory (tabu list) to
prevent revisiting recent moves and an aspiration criterion that overrides
the tabu status when a global improvement is found.

Algorithm:
    1. Start from ATC or EDD heuristic solution.
    2. At each iteration, evaluate all swap/insert neighbors.
    3. Select the best non-tabu neighbor (or tabu if aspiration met).
    4. Record the reverse move in the tabu list.
    5. Update global best if improved.
    6. Repeat until max iterations or time limit.

Tabu list stores (job, position) pairs to prevent a recently moved job
from returning to its previous position for a number of iterations
(the tabu tenure).

Notation: 1 || ΣwjTj
Complexity: O(iterations * n^2) per run.

Reference:
    Potts, C.N. & Van Wassenhove, L.N. (1991). Single machine tardiness
    sequencing heuristics. IIE Transactions, 23(4), 346-354.
    https://doi.org/10.1080/07408179108963868

    Glover, F. (1989). Tabu search — Part I. ORSA Journal on Computing,
    1(3), 190-206.
    https://doi.org/10.1287/ijoc.1.3.190
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)

_instance_path = os.path.join(_sm_dir, "instance.py")
_spec = importlib.util.spec_from_file_location("sm_instance_ts", _instance_path)
_sm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("sm_instance_ts", _sm_instance)
_spec.loader.exec_module(_sm_instance)

SingleMachineInstance = _sm_instance.SingleMachineInstance
SingleMachineSolution = _sm_instance.SingleMachineSolution
compute_weighted_tardiness = _sm_instance.compute_weighted_tardiness
compute_total_tardiness = _sm_instance.compute_total_tardiness

_atc_path = os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py")
_spec2 = importlib.util.spec_from_file_location("sm_atc_ts", _atc_path)
_sm_atc = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("sm_atc_ts", _sm_atc)
_spec2.loader.exec_module(_sm_atc)

_rules_path = os.path.join(_sm_dir, "heuristics", "dispatching_rules.py")
_spec3 = importlib.util.spec_from_file_location("sm_dispatching_ts", _rules_path)
_sm_rules = importlib.util.module_from_spec(_spec3)
sys.modules.setdefault("sm_dispatching_ts", _sm_rules)
_spec3.loader.exec_module(_sm_rules)

atc = _sm_atc.atc
edd = _sm_rules.edd


def tabu_search(
    instance: SingleMachineInstance,
    objective: str = "weighted_tardiness",
    max_iterations: int = 2000,
    tabu_tenure: int | None = None,
    neighborhood: str = "swap",
    time_limit: float | None = None,
    seed: int | None = None,
) -> SingleMachineSolution:
    """Tabu Search for single machine tardiness objectives.

    Args:
        instance: Single machine instance (must have due_dates).
        objective: "weighted_tardiness" (ΣwjTj) or "total_tardiness" (ΣTj).
        max_iterations: Maximum iterations.
        tabu_tenure: Number of iterations a move stays tabu. Default: sqrt(n).
        neighborhood: "swap" or "insert" move type.
        time_limit: Time limit in seconds.
        seed: Random seed (used for tie-breaking).

    Returns:
        Best SingleMachineSolution found.
    """
    rng = np.random.default_rng(seed)
    assert instance.due_dates is not None, "Due dates required"
    n = instance.n
    start_time = time.time()

    if tabu_tenure is None:
        tabu_tenure = max(5, int(n ** 0.5))

    eval_fn = (
        compute_weighted_tardiness if objective == "weighted_tardiness"
        else compute_total_tardiness
    )
    obj_name = "ΣwjTj" if objective == "weighted_tardiness" else "ΣTj"

    # ── Initial solution ─────────────────────────────────────────────────
    if objective == "weighted_tardiness" and instance.weights is not None:
        try:
            init_sol = atc(instance)
            current = list(init_sol.sequence)
        except Exception:
            init_sol = edd(instance)
            current = list(init_sol.sequence)
    else:
        init_sol = edd(instance)
        current = list(init_sol.sequence)

    current_obj = eval_fn(instance, current)
    best = list(current)
    best_obj = current_obj

    # Tabu list: dict (job, position) -> iteration when tabu expires
    tabu_dict: dict[tuple[int, int], int] = {}

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        best_neighbor = None
        best_neighbor_obj = float("inf")
        best_move = None

        if neighborhood == "swap":
            # Evaluate all swap neighbors
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = list(current)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    obj_val = eval_fn(instance, candidate)

                    # Check tabu status
                    move = (current[i], j, current[j], i)
                    is_tabu = (
                        (current[i], j) in tabu_dict
                        and tabu_dict[(current[i], j)] > iteration
                    ) or (
                        (current[j], i) in tabu_dict
                        and tabu_dict[(current[j], i)] > iteration
                    )

                    # Aspiration: accept if improves global best
                    if is_tabu and obj_val >= best_obj:
                        continue

                    if obj_val < best_neighbor_obj:
                        best_neighbor_obj = obj_val
                        best_neighbor = candidate
                        best_move = move

        else:  # insert
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    candidate = list(current)
                    job = candidate.pop(i)
                    insert_pos = j if j < i else j - 1
                    insert_pos = max(0, min(insert_pos, len(candidate)))
                    candidate.insert(insert_pos, job)
                    obj_val = eval_fn(instance, candidate)

                    is_tabu = (
                        (job, insert_pos) in tabu_dict
                        and tabu_dict[(job, insert_pos)] > iteration
                    )

                    if is_tabu and obj_val >= best_obj:
                        continue

                    if obj_val < best_neighbor_obj:
                        best_neighbor_obj = obj_val
                        best_neighbor = candidate
                        best_move = (job, i)  # record job's old position

        if best_neighbor is None:
            # All neighbors are tabu; clear tabu list and retry
            tabu_dict.clear()
            continue

        # Apply move
        current = best_neighbor
        current_obj = best_neighbor_obj

        # Record tabu: prevent reverse move
        if neighborhood == "swap" and best_move is not None:
            job_a, pos_a, job_b, pos_b = best_move
            tabu_dict[(job_a, pos_b)] = iteration + tabu_tenure
            tabu_dict[(job_b, pos_a)] = iteration + tabu_tenure
        elif best_move is not None:
            job, old_pos = best_move
            tabu_dict[(job, old_pos)] = iteration + tabu_tenure

        if current_obj < best_obj:
            best = list(current)
            best_obj = current_obj

    return SingleMachineSolution(
        sequence=best,
        objective_value=best_obj,
        objective_name=obj_name,
    )


if __name__ == "__main__":
    inst = SingleMachineInstance.random(n=15, seed=42)
    print(f"Jobs: {inst.n}")
    print(f"Processing times: {inst.processing_times}")
    print(f"Due dates: {inst.due_dates}")
    print(f"Weights: {inst.weights}")

    sol_atc = atc(inst)
    print(f"\nATC: ΣwjTj = {sol_atc.objective_value}")

    sol_ts = tabu_search(inst, objective="weighted_tardiness", seed=42)
    print(f"TS:  ΣwjTj = {sol_ts.objective_value}")

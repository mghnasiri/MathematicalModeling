"""
Local Search for Single Machine Scheduling — 1 || ΣwjTj / 1 || ΣTj

Implements iterative improvement with swap and insertion neighborhoods.
Uses best-improvement strategy: evaluate all neighbors and accept the
best improving move. Stops when no improving neighbor exists.

Neighborhoods:
- Swap: exchange two adjacent jobs in the sequence
- Insert: remove a job and re-insert it at another position

Warm-started with ATC heuristic (for ΣwjTj) or EDD (for ΣTj).

Complexity: O(n^2 * iterations) per run.

References:
    Potts, C.N. & Van Wassenhove, L.N. (1991). Single machine
    tardiness sequencing heuristics. IIE Transactions, 23(4),
    346-354.
    https://doi.org/10.1080/07408179108963868

    Crauwels, H.A.J., Potts, C.N. & Van Wassenhove, L.N. (1998).
    Local search heuristics for the single machine total weighted
    tardiness scheduling problem. INFORMS Journal on Computing,
    10(3), 341-350.
    https://doi.org/10.1287/ijoc.10.3.341
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_sm_dir = os.path.dirname(_this_dir)


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("sm_instance_ls", os.path.join(_sm_dir, "instance.py"))
SingleMachineInstance = _inst.SingleMachineInstance
SingleMachineSolution = _inst.SingleMachineSolution
compute_weighted_tardiness = _inst.compute_weighted_tardiness
compute_total_tardiness = _inst.compute_total_tardiness

_rules = _load_mod(
    "sm_dispatching_ls",
    os.path.join(_sm_dir, "heuristics", "dispatching_rules.py"),
)
edd = _rules.edd

_atc_mod = _load_mod(
    "sm_atc_ls",
    os.path.join(_sm_dir, "heuristics", "apparent_tardiness_cost.py"),
)
atc = _atc_mod.atc


def local_search(
    instance: SingleMachineInstance,
    objective: str = "weighted_tardiness",
    neighborhood: str = "swap",
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> SingleMachineSolution:
    """Solve single machine scheduling using Local Search.

    Args:
        instance: Single machine instance (must have due_dates).
        objective: "weighted_tardiness" (ΣwjTj) or "total_tardiness" (ΣTj).
        neighborhood: "swap" (adjacent swap), "insert" (insertion), or
                      "both" (try both).
        max_iterations: Maximum iterations without improvement.
        time_limit: Time limit in seconds.
        seed: Random seed (used only for initial perturbation if needed).

    Returns:
        Best SingleMachineSolution found.
    """
    assert instance.due_dates is not None, "Due dates required"
    n = instance.n
    start_time = time.time()

    eval_fn = (
        compute_weighted_tardiness if objective == "weighted_tardiness"
        else compute_total_tardiness
    )
    obj_name = "ΣwjTj" if objective == "weighted_tardiness" else "ΣTj"

    # Initialize with ATC or EDD
    if objective == "weighted_tardiness" and instance.weights is not None:
        try:
            init_sol = atc(instance)
            seq = list(init_sol.sequence)
        except Exception:
            init_sol = edd(instance)
            seq = list(init_sol.sequence)
    else:
        init_sol = edd(instance)
        seq = list(init_sol.sequence)

    current_obj = eval_fn(instance, seq)
    best_seq = list(seq)
    best_obj = current_obj

    use_swap = neighborhood in ("swap", "both")
    use_insert = neighborhood in ("insert", "both")

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        improved = False

        # ── Swap neighborhood (adjacent) ─────────────────────────────────
        if use_swap:
            best_delta = 0
            best_swap = None

            for i in range(n - 1):
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                new_obj = eval_fn(instance, seq)
                delta = new_obj - current_obj
                if delta < best_delta:
                    best_delta = delta
                    best_swap = i
                seq[i], seq[i + 1] = seq[i + 1], seq[i]

            if best_swap is not None:
                seq[best_swap], seq[best_swap + 1] = (
                    seq[best_swap + 1], seq[best_swap],
                )
                current_obj += best_delta
                improved = True

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_seq = list(seq)

        # ── Insert neighborhood ──────────────────────────────────────────
        if use_insert:
            best_delta = 0
            best_insert = None

            for i in range(n):
                job = seq[i]
                test = seq[:i] + seq[i + 1:]
                for j in range(n):
                    if j == i:
                        continue
                    insert_pos = j if j < i else j - 1
                    insert_pos = max(0, min(insert_pos, len(test)))
                    candidate = list(test)
                    candidate.insert(insert_pos, job)
                    new_obj = eval_fn(instance, candidate)
                    delta = new_obj - current_obj
                    if delta < best_delta:
                        best_delta = delta
                        best_insert = (i, insert_pos)

            if best_insert is not None:
                i, insert_pos = best_insert
                job = seq.pop(i)
                seq.insert(insert_pos, job)
                current_obj += best_delta
                improved = True

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_seq = list(seq)

        if not improved:
            break

    return SingleMachineSolution(
        sequence=best_seq,
        objective_value=best_obj,
        objective_name=obj_name,
    )


if __name__ == "__main__":
    inst = SingleMachineInstance.random(n=15, seed=42)
    print(f"Jobs: {inst.n}")

    sol_atc = atc(inst)
    print(f"ATC: ΣwjTj = {sol_atc.objective_value}")

    sol_ls = local_search(inst, objective="weighted_tardiness", neighborhood="both")
    print(f"LS:  ΣwjTj = {sol_ls.objective_value}")

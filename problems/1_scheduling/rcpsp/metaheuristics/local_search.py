"""
Local Search for RCPSP.

Problem notation: PS | prec | Cmax

Iterative improvement using swap and shift neighborhoods on activity lists,
decoded via Serial SGS. Each move maintains precedence feasibility.

Neighborhoods:
    - Swap: exchange two adjacent activities in the priority list
    - Shift: move an activity to a different position in the list
    - Both: alternate between swap and shift

Warm-started with best SGS priority rule (LFT/EST/MTS/GRPW).

Complexity: O(iterations * n^2 * K * T) per run.

References:
    Kolisch, R. & Hartmann, S. (2006). Experimental investigation of
    heuristics for resource-constrained project scheduling. European
    Journal of Operational Research, 169(1), 16-37.
    https://doi.org/10.1016/j.ejor.2004.01.035

    Hartmann, S. (2002). A self-adapting genetic algorithm for project
    scheduling under resource constraints. Naval Research Logistics,
    49(5), 433-448.
    https://doi.org/10.1002/nav.10029
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


_inst = _load_mod("rcpsp_instance_ls", os.path.join(_parent_dir, "instance.py"))
RCPSPInstance = _inst.RCPSPInstance
RCPSPSolution = _inst.RCPSPSolution

_sgs = _load_mod(
    "rcpsp_sgs_ls",
    os.path.join(_parent_dir, "heuristics", "serial_sgs.py"),
)
serial_sgs = _sgs.serial_sgs


def local_search(
    instance: RCPSPInstance,
    neighborhood: str = "both",
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> RCPSPSolution:
    """Solve RCPSP using iterative local search on activity lists.

    Args:
        instance: An RCPSPInstance.
        neighborhood: "swap", "shift", or "both".
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        RCPSPSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    total = instance.n + 2
    start_time = time.time()

    # Warm-start: try all priority rules, pick best
    best_sol = None
    best_list = None
    for rule in ["lft", "est", "mts", "grpw"]:
        sol = serial_sgs(instance, priority_rule=rule)
        if best_sol is None or sol.makespan < best_sol.makespan:
            best_sol = sol
            best_list = _extract_priority_list(instance, sol, rule)

    current_list = best_list[:]
    current_ms = best_sol.makespan
    best_ms = current_ms
    best_result = best_sol

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        improved = False

        if neighborhood == "swap" or neighborhood == "both":
            improved = _try_swap_neighborhood(
                instance, current_list, current_ms, best_ms, rng
            )
            if improved:
                sol = serial_sgs(instance, priority_list=current_list)
                current_ms = sol.makespan
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_result = sol
                    best_list = current_list[:]

        if (not improved) and (neighborhood == "shift" or neighborhood == "both"):
            improved = _try_shift_neighborhood(
                instance, current_list, current_ms, best_ms, rng
            )
            if improved:
                sol = serial_sgs(instance, priority_list=current_list)
                current_ms = sol.makespan
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_result = sol
                    best_list = current_list[:]

        if not improved:
            # Perturbation: random swap to escape local optimum
            _random_swap(instance, current_list, rng)
            sol = serial_sgs(instance, priority_list=current_list)
            current_ms = sol.makespan
            if current_ms < best_ms:
                best_ms = current_ms
                best_result = sol
                best_list = current_list[:]

    return best_result


def _extract_priority_list(
    instance: RCPSPInstance,
    sol: RCPSPSolution,
    rule: str,
) -> list[int]:
    """Build a precedence-feasible priority list from a solution."""
    total = instance.n + 2
    # Sort activities by start time (ties broken by index)
    activities = list(range(total))
    activities.sort(key=lambda a: (sol.start_times[a], a))
    return activities


def _is_precedence_feasible(instance: RCPSPInstance, plist: list[int]) -> bool:
    """Check if a priority list respects precedence."""
    pos = {act: i for i, act in enumerate(plist)}
    for act in range(instance.n + 2):
        for succ in instance.successors.get(act, []):
            if pos[act] >= pos[succ]:
                return False
    return True


def _try_swap_neighborhood(
    instance: RCPSPInstance,
    plist: list[int],
    current_ms: int,
    best_ms: int,
    rng: np.random.Generator,
) -> bool:
    """Try sampled adjacent swaps, accept first improvement."""
    total = len(plist)
    indices = list(range(1, total - 1))  # skip source and sink
    rng.shuffle(indices)

    for idx in indices[:min(len(indices), total * 2)]:
        if idx + 1 >= total:
            continue
        a, b = plist[idx], plist[idx + 1]

        # Check if swap is precedence-feasible
        # Can't swap if a must come before b (a is predecessor of b)
        if b in instance.successors.get(a, []):
            continue
        if a in instance.predecessors.get(b, []):
            continue

        # Swap
        plist[idx], plist[idx + 1] = b, a
        sol = serial_sgs(instance, priority_list=plist)

        if sol.makespan < current_ms:
            return True
        else:
            # Undo
            plist[idx], plist[idx + 1] = a, b

    return False


def _try_shift_neighborhood(
    instance: RCPSPInstance,
    plist: list[int],
    current_ms: int,
    best_ms: int,
    rng: np.random.Generator,
) -> bool:
    """Try sampled shift moves, accept first improvement."""
    total = len(plist)
    # Pick a random non-dummy activity to shift
    non_dummy = [i for i in range(1, total - 1)]
    rng.shuffle(non_dummy)

    for idx in non_dummy[:min(len(non_dummy), total)]:
        act = plist[idx]
        preds = instance.predecessors.get(act, [])
        succs = instance.successors.get(act, [])

        # Find valid range
        pos_map = {a: i for i, a in enumerate(plist)}
        earliest = max((pos_map[p] for p in preds), default=0) + 1
        latest = min((pos_map[s] for s in succs), default=total - 1) - 1

        if latest <= earliest or latest == idx:
            continue

        # Try a few random positions
        for _ in range(3):
            new_pos = rng.integers(earliest, latest + 1)
            if new_pos == idx:
                continue

            # Perform shift
            old_list = plist[:]
            plist.pop(idx)
            if new_pos > idx:
                new_pos -= 1
            plist.insert(new_pos, act)

            sol = serial_sgs(instance, priority_list=plist)
            if sol.makespan < current_ms:
                return True
            else:
                plist[:] = old_list

    return False


def _random_swap(
    instance: RCPSPInstance,
    plist: list[int],
    rng: np.random.Generator,
) -> None:
    """Perform a random feasible adjacent swap for perturbation."""
    total = len(plist)
    for _ in range(10):
        idx = rng.integers(1, total - 2)
        a, b = plist[idx], plist[idx + 1]
        # Can't swap if a must come before b
        if b in instance.successors.get(a, []):
            continue
        if a in instance.predecessors.get(b, []):
            continue
        plist[idx], plist[idx + 1] = b, a
        return


if __name__ == "__main__":
    inst = RCPSPInstance.random(n=10, num_resources=2, seed=42)
    print(f"RCPSP: {inst.n} activities, {inst.num_resources} resources")
    print(f"Critical path: {inst.critical_path_length()}")

    sol = local_search(inst, seed=42)
    print(f"Local Search: makespan={sol.makespan}")

    sgs_sol = serial_sgs(inst, priority_rule="lft")
    print(f"SGS (LFT): makespan={sgs_sol.makespan}")

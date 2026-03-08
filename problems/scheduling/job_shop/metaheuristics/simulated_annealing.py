"""
Simulated Annealing for Job Shop Scheduling (Jm || Cmax)

Implements SA with critical-path-based neighborhood. Moves swap adjacent
operations on a critical block, which is the most effective neighborhood
structure for JSP.

Complexity: O(iterations * n * m) per run.

Reference:
    Van Laarhoven, P.J.M., Aarts, E.H.L. & Lenstra, J.K. (1992).
    "Job shop scheduling by simulated annealing."
    Operations Research, 40(1), 113-125.
    https://doi.org/10.1287/opre.40.1.113
"""

from __future__ import annotations
import sys
import os
import math
import importlib.util
import numpy as np
from collections import deque

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_mod("job_shop_instance_mod", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences

_disp = _load_mod("job_shop_dispatching_mod", os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"))
dispatching_rule = _disp.dispatching_rule


def simulated_annealing(
    instance: JobShopInstance,
    max_iterations: int = 10000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.995,
    seed: int | None = None,
) -> JobShopSolution:
    """
    Simulated Annealing for Jm||Cmax.

    Args:
        instance: JSP instance.
        max_iterations: Maximum number of iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Temperature multiplier per iteration.
        seed: Random seed.

    Returns:
        Best JobShopSolution found.
    """
    rng = np.random.default_rng(seed)

    # Get initial solution from best dispatching rule
    best_init = None
    for rule in ["spt", "mwr", "lpt", "lwr"]:
        sol = dispatching_rule(instance, rule=rule)
        if best_init is None or sol.makespan < best_init.makespan:
            best_init = sol

    # Work with machine sequence representation
    machine_seqs = _extract_machine_sequences(instance, best_init.start_times)
    current_makespan = best_init.makespan
    best_makespan = current_makespan
    best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}

    if initial_temp is None:
        initial_temp = current_makespan * 0.05

    temp = initial_temp

    # Precompute operation info for fast lookup
    op_machine = {}  # (j,k) -> machine
    op_pt = {}       # (j,k) -> processing time
    for j in range(instance.n):
        for k, (mach, pt) in enumerate(instance.jobs[j]):
            op_machine[(j, k)] = mach
            op_pt[(j, k)] = pt

    for iteration in range(max_iterations):
        # Get critical path and find swappable pairs
        start_times = _build_start_times_fast(instance, machine_seqs, op_machine, op_pt)
        if start_times is None:
            break
        neighbors = _get_critical_path_neighbors_fast(
            instance, start_times, machine_seqs, op_machine, op_pt
        )

        if not neighbors:
            temp *= cooling_rate
            continue

        # Pick a random neighbor
        idx = rng.integers(0, len(neighbors))
        mach, pos = neighbors[idx]

        # Perform swap
        seq = machine_seqs[mach]
        seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]

        # Evaluate
        new_start_times = _build_start_times_fast(instance, machine_seqs, op_machine, op_pt)
        if new_start_times is None:
            # Cycle created — undo swap
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
            temp *= cooling_rate
            continue

        new_makespan = max(
            new_start_times[op] + op_pt[op]
            for op in new_start_times
        )

        delta = new_makespan - current_makespan

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_makespan = new_makespan
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}
        else:
            # Undo swap
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]

        temp *= cooling_rate

    # Build final solution
    final_start_times = _build_start_times_fast(instance, best_machine_seqs, op_machine, op_pt)
    machine_sequences = build_machine_sequences(instance, final_start_times)

    return JobShopSolution(
        start_times=final_start_times,
        makespan=best_makespan,
        machine_sequences=machine_sequences,
    )


def _extract_machine_sequences(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
) -> dict[int, list[tuple[int, int]]]:
    """Extract machine sequences ordered by start time."""
    machine_ops: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for j in range(instance.n):
        for k, (mach, _) in enumerate(instance.jobs[j]):
            machine_ops.setdefault(mach, []).append(
                (start_times.get((j, k), 0), (j, k))
            )
    return {
        m: [op for _, op in sorted(ops)]
        for m, ops in machine_ops.items()
    }


def _build_start_times_fast(
    instance: JobShopInstance,
    machine_seqs: dict[int, list[tuple[int, int]]],
    op_machine: dict[tuple[int, int], int],
    op_pt: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """
    Build start times using topological sort + longest path.
    Single pass, O(n*m).
    """
    # Build adjacency and in-degree
    all_ops = []
    in_degree: dict[tuple[int, int], int] = {}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            op = (j, k)
            all_ops.append(op)
            in_degree[op] = 0
            adj[op] = []

    for j in range(instance.n):
        for k in range(len(instance.jobs[j]) - 1):
            adj[(j, k)].append((j, k + 1))
            in_degree[(j, k + 1)] += 1

    for mach, seq in machine_seqs.items():
        for i in range(len(seq) - 1):
            adj[seq[i]].append(seq[i + 1])
            in_degree[seq[i + 1]] += 1

    # Kahn's algorithm for topological sort
    topo_order = []
    queue = deque(op for op in all_ops if in_degree[op] == 0)

    while queue:
        op = queue.popleft()
        topo_order.append(op)
        for succ in adj[op]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # If cycle detected, return None (infeasible)
    if len(topo_order) != len(all_ops):
        return None

    # Compute longest path (start times) in topological order
    start_times = {op: 0 for op in all_ops}
    for op in topo_order:
        s = start_times[op]
        pt = op_pt[op]
        for succ in adj[op]:
            start_times[succ] = max(start_times[succ], s + pt)

    return start_times


def _get_critical_path_neighbors_fast(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
    machine_seqs: dict[int, list[tuple[int, int]]],
    op_machine: dict[tuple[int, int], int],
    op_pt: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """Find swappable pairs on critical path."""
    # Compute completion times
    completion = {op: start_times[op] + op_pt[op] for op in start_times}
    makespan = max(completion.values()) if completion else 0

    # Trace critical path backward
    critical_ops = set()
    stack = [op for op, c in completion.items() if c == makespan]

    # Build predecessor lookup for fast tracing
    machine_pred: dict[tuple[int, int], tuple[int, int]] = {}
    for mach, seq in machine_seqs.items():
        for i in range(len(seq) - 1):
            machine_pred[seq[i + 1]] = seq[i]

    while stack:
        op = stack.pop()
        if op in critical_ops:
            continue
        critical_ops.add(op)
        j, k = op
        s = start_times[op]

        # Job predecessor
        if k > 0 and start_times[(j, k-1)] + op_pt[(j, k-1)] == s:
            stack.append((j, k-1))

        # Machine predecessor
        if op in machine_pred:
            pred = machine_pred[op]
            if completion[pred] == s:
                stack.append(pred)

    # Find adjacent pairs on same machine within critical path
    neighbors = []
    for mach, seq in machine_seqs.items():
        for i in range(len(seq) - 1):
            if seq[i] in critical_ops and seq[i + 1] in critical_ops:
                neighbors.append((mach, i))

    return neighbors


if __name__ == "__main__":
    from instance import ft06

    print("=== Simulated Annealing on ft06 (optimal=55) ===\n")
    inst = ft06()
    sol = simulated_annealing(inst, max_iterations=5000, seed=42)
    print(f"SA makespan: {sol.makespan}")

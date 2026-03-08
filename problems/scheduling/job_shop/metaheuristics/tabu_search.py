"""
Tabu Search for Job Shop Scheduling (Jm || Cmax)

Implements a Tabu Search with N1 (adjacent swap) neighborhood on critical
blocks. Uses short-term memory to prevent cycling and an aspiration
criterion that accepts moves improving the best-known solution.

Complexity: O(iterations * n * m) per run.

Reference:
    Nowicki, E. & Smutnicki, C. (1996).
    "A Fast Taboo Search Algorithm for the Job Shop Problem."
    Management Science, 42(6), 797-813.
    https://doi.org/10.1287/mnsc.42.6.797
"""

from __future__ import annotations
import sys
import os
import importlib.util
from collections import deque
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

_inst = _load_mod("job_shop_instance_mod", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences

_disp = _load_mod("job_shop_dispatching_mod", os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"))
dispatching_rule = _disp.dispatching_rule


def tabu_search(
    instance: JobShopInstance,
    max_iterations: int = 5000,
    tabu_tenure: int | None = None,
    seed: int | None = None,
) -> JobShopSolution:
    """
    Tabu Search for Jm||Cmax.

    Args:
        instance: JSP instance.
        max_iterations: Maximum iterations.
        tabu_tenure: Length of tabu list. If None, set to n + m/2.
        seed: Random seed.

    Returns:
        Best JobShopSolution found.
    """
    rng = np.random.default_rng(seed)

    if tabu_tenure is None:
        tabu_tenure = instance.n + instance.m // 2

    # Precompute operation info
    op_pt: dict[tuple[int, int], int] = {}
    for j in range(instance.n):
        for k, (mach, pt) in enumerate(instance.jobs[j]):
            op_pt[(j, k)] = pt

    # Initial solution from dispatching rules
    best_init = None
    for rule in ["spt", "mwr", "lpt", "lwr"]:
        sol = dispatching_rule(instance, rule=rule)
        if best_init is None or sol.makespan < best_init.makespan:
            best_init = sol

    machine_seqs = _extract_machine_sequences(instance, best_init.start_times)
    current_makespan = best_init.makespan
    best_makespan = current_makespan
    best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}

    tabu_list: list[tuple[tuple[int, int], tuple[int, int]]] = []
    no_improve = 0

    for iteration in range(max_iterations):
        start_times = _build_start_times_fast(instance, machine_seqs, op_pt)
        if start_times is None:
            break
        neighbors = _get_critical_swaps(instance, start_times, machine_seqs, op_pt)

        if not neighbors:
            no_improve += 1
            if no_improve > max_iterations // 5:
                break
            continue

        best_move = None
        best_move_ms = float('inf')

        for mach, pos in neighbors:
            seq = machine_seqs[mach]
            op1, op2 = seq[pos], seq[pos + 1]

            is_tabu = (op1, op2) in tabu_list or (op2, op1) in tabu_list

            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
            new_st = _build_start_times_fast(instance, machine_seqs, op_pt)
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]

            if new_st is None:
                continue  # Skip moves creating cycles

            new_ms = max(new_st[op] + op_pt[op] for op in new_st)

            if is_tabu and new_ms >= best_makespan:
                continue

            if new_ms < best_move_ms:
                best_move_ms = new_ms
                best_move = (mach, pos)

        if best_move is None:
            no_improve += 1
            if no_improve > max_iterations // 5:
                break
            continue

        mach, pos = best_move
        seq = machine_seqs[mach]
        op1, op2 = seq[pos], seq[pos + 1]
        seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
        current_makespan = best_move_ms

        tabu_list.append((op1, op2))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}
            no_improve = 0
        else:
            no_improve += 1

    final_st = _build_start_times_fast(instance, best_machine_seqs, op_pt)
    machine_sequences = build_machine_sequences(instance, final_st)

    return JobShopSolution(
        start_times=final_st,
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
    op_pt: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    """Build start times using topological sort + longest path. O(n*m)."""
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

    # Kahn's topological sort
    topo_order = []
    queue = deque(op for op in all_ops if in_degree[op] == 0)
    while queue:
        op = queue.popleft()
        topo_order.append(op)
        for succ in adj[op]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # Cycle detection
    if len(topo_order) != len(all_ops):
        return None

    # Longest path
    start_times = {op: 0 for op in all_ops}
    for op in topo_order:
        s = start_times[op]
        pt = op_pt[op]
        for succ in adj[op]:
            start_times[succ] = max(start_times[succ], s + pt)

    return start_times


def _get_critical_swaps(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
    machine_seqs: dict[int, list[tuple[int, int]]],
    op_pt: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """Find adjacent swap moves on critical path blocks."""
    completion = {op: start_times[op] + op_pt[op] for op in start_times}
    makespan = max(completion.values()) if completion else 0

    machine_pred: dict[tuple[int, int], tuple[int, int]] = {}
    for mach, seq in machine_seqs.items():
        for i in range(len(seq) - 1):
            machine_pred[seq[i + 1]] = seq[i]

    critical_ops = set()
    stack = [op for op, c in completion.items() if c == makespan]

    while stack:
        op = stack.pop()
        if op in critical_ops:
            continue
        critical_ops.add(op)
        j, k = op
        s = start_times[op]

        if k > 0 and start_times[(j, k-1)] + op_pt[(j, k-1)] == s:
            stack.append((j, k-1))

        if op in machine_pred:
            pred = machine_pred[op]
            if completion[pred] == s:
                stack.append(pred)

    neighbors = []
    for mach, seq in machine_seqs.items():
        for i in range(len(seq) - 1):
            if seq[i] in critical_ops and seq[i + 1] in critical_ops:
                neighbors.append((mach, i))

    return neighbors


if __name__ == "__main__":
    from instance import ft06

    print("=== Tabu Search on ft06 (optimal=55) ===\n")
    inst = ft06()
    sol = tabu_search(inst, max_iterations=3000, seed=42)
    print(f"TS makespan: {sol.makespan}")

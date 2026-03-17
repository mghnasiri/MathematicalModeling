"""
Local Search for Job Shop Scheduling (Jm || Cmax).

Critical-path-based neighborhood: swap adjacent operations on the same
machine within a critical block. This is the most effective neighborhood
for JSP (Nowicki & Smutnicki, 1996).

Uses best-improvement search with random restarts.
Warm-started with best dispatching rule.

Complexity: O(iterations * n * m) per run.

References:
    Nowicki, E. & Smutnicki, C. (1996). A fast taboo search algorithm
    for the job shop problem. Management Science, 42(6), 797-813.
    https://doi.org/10.1287/mnsc.42.6.797

    Van Laarhoven, P.J.M., Aarts, E.H.L. & Lenstra, J.K. (1992).
    Job shop scheduling by simulated annealing. Operations Research,
    40(1), 113-125.
    https://doi.org/10.1287/opre.40.1.113
"""

from __future__ import annotations

import sys
import os
import time
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


_inst = _load_mod("jsp_instance_ls", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences

_disp = _load_mod(
    "jsp_disp_ls",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def local_search(
    instance: JobShopInstance,
    max_iterations: int = 1000,
    time_limit: float | None = None,
    seed: int | None = None,
) -> JobShopSolution:
    """Solve Jm||Cmax using local search with critical-path neighborhood.

    Args:
        instance: A JobShopInstance.
        max_iterations: Maximum number of iterations.
        time_limit: Maximum wall-clock time in seconds.
        seed: Random seed for reproducibility.

    Returns:
        JobShopSolution with the best schedule found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Precompute operation info
    op_machine = {}
    op_pt = {}
    for j in range(instance.n):
        for k, (mach, pt) in enumerate(instance.jobs[j]):
            op_machine[(j, k)] = mach
            op_pt[(j, k)] = pt

    # Warm-start with best dispatching rule
    best_init = None
    for rule in ["spt", "mwr", "lpt", "lwr"]:
        sol = dispatching_rule(instance, rule=rule)
        if best_init is None or sol.makespan < best_init.makespan:
            best_init = sol

    machine_seqs = _extract_machine_sequences(instance, best_init.start_times)
    current_ms = best_init.makespan
    best_ms = current_ms
    best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}

    stagnant = 0

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        start_times = _build_start_times(instance, machine_seqs, op_machine, op_pt)
        if start_times is None:
            break

        neighbors = _get_critical_neighbors(
            instance, start_times, machine_seqs, op_machine, op_pt
        )

        if not neighbors:
            stagnant += 1
            if stagnant > 5:
                # Random perturbation
                _random_perturbation(machine_seqs, rng)
                stagnant = 0
            continue

        # Best-improvement
        best_delta = 0
        best_move = None

        for mach, pos in neighbors:
            seq = machine_seqs[mach]
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
            new_st = _build_start_times(instance, machine_seqs, op_machine, op_pt)
            if new_st is not None:
                new_ms = max(new_st[op] + op_pt[op] for op in new_st)
                delta = new_ms - current_ms
                if delta < best_delta:
                    best_delta = delta
                    best_move = (mach, pos)
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]

        if best_move is not None:
            mach, pos = best_move
            seq = machine_seqs[mach]
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
            current_ms += best_delta
            stagnant = 0

            if current_ms < best_ms:
                best_ms = current_ms
                best_machine_seqs = {m: list(s) for m, s in machine_seqs.items()}
        else:
            stagnant += 1
            if stagnant > 5:
                _random_perturbation(machine_seqs, rng)
                stagnant = 0

    final_st = _build_start_times(instance, best_machine_seqs, op_machine, op_pt)
    return JobShopSolution(
        start_times=final_st,
        makespan=best_ms,
        machine_sequences=build_machine_sequences(instance, final_st),
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


def _build_start_times(
    instance: JobShopInstance,
    machine_seqs: dict[int, list[tuple[int, int]]],
    op_machine: dict,
    op_pt: dict,
) -> dict[tuple[int, int], int] | None:
    """Build start times via topological sort. Returns None if cycle."""
    all_ops = []
    in_degree = {}
    adj = {}

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

    topo = []
    queue = deque(op for op in all_ops if in_degree[op] == 0)
    while queue:
        op = queue.popleft()
        topo.append(op)
        for succ in adj[op]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(topo) != len(all_ops):
        return None

    start_times = {op: 0 for op in all_ops}
    for op in topo:
        s = start_times[op]
        pt = op_pt[op]
        for succ in adj[op]:
            start_times[succ] = max(start_times[succ], s + pt)

    return start_times


def _get_critical_neighbors(
    instance: JobShopInstance,
    start_times: dict,
    machine_seqs: dict,
    op_machine: dict,
    op_pt: dict,
) -> list[tuple[int, int]]:
    """Find swappable adjacent pairs on the critical path."""
    completion = {op: start_times[op] + op_pt[op] for op in start_times}
    makespan = max(completion.values()) if completion else 0

    critical_ops = set()
    stack = [op for op, c in completion.items() if c == makespan]

    machine_pred = {}
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

        if k > 0 and start_times[(j, k - 1)] + op_pt[(j, k - 1)] == s:
            stack.append((j, k - 1))

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


def _random_perturbation(
    machine_seqs: dict[int, list[tuple[int, int]]],
    rng: np.random.Generator,
) -> None:
    """Random swap on a random machine to escape local optima."""
    machines = [m for m, seq in machine_seqs.items() if len(seq) >= 2]
    if not machines:
        return
    mach = machines[rng.integers(len(machines))]
    seq = machine_seqs[mach]
    i = rng.integers(len(seq) - 1)
    seq[i], seq[i + 1] = seq[i + 1], seq[i]


if __name__ == "__main__":
    _inst_mod = _load_mod("jsp_instance_ls_main", os.path.join(_parent_dir, "instance.py"))
    inst = _inst_mod.ft06()
    print(f"ft06: {inst.n} jobs, {inst.m} machines (optimal=55)")

    sol = local_search(inst, seed=42)
    print(f"LS: makespan={sol.makespan}")

    disp_sol = dispatching_rule(inst, rule="spt")
    print(f"SPT: makespan={disp_sol.makespan}")

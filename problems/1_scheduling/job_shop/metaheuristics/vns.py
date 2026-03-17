"""
Variable Neighborhood Search for Job Shop Scheduling (Jm || Cmax).

Problem: Jm || Cmax (Job Shop Makespan)

Uses multiple neighborhood structures of increasing size:
- N1: swap two adjacent operations on a critical block (Nowicki-Smutnicki)
- N2: insert an operation from one critical block position to another
- N3: swap two random operations on the same machine

Shaking perturbs the solution at the current neighborhood level.
Local search applies best-improvement within N1.

Warm-started with best dispatching rule solution.

Complexity: O(iterations * n * m) per run.

References:
    Mladenović, N. & Hansen, P. (1997). Variable neighborhood search.
    Computers & Operations Research, 24(11), 1097-1100.
    https://doi.org/10.1016/S0305-0548(97)00031-2

    Amiri, M., Zandieh, M., Vahdani, B., Soltani, R. & Roshanaei, V.
    (2010). An un-constrained non-linear optimization using VNS for
    the job shop scheduling problem. Applied Mathematical Modelling,
    34(4), 1058-1073.
    https://doi.org/10.1016/j.apm.2009.07.016
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


_inst = _load_mod("job_shop_instance_vns", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences

_disp = _load_mod(
    "job_shop_dispatching_vns",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


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
) -> dict[tuple[int, int], int] | None:
    """Build start times via topological sort. Returns None if cycle."""
    op_pt = {}
    for j in range(instance.n):
        for k, (_, pt) in enumerate(instance.jobs[j]):
            op_pt[(j, k)] = pt

    all_ops = list(op_pt.keys())
    in_degree = {op: 0 for op in all_ops}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {
        op: [] for op in all_ops
    }

    for j in range(instance.n):
        for k in range(len(instance.jobs[j]) - 1):
            adj[(j, k)].append((j, k + 1))
            in_degree[(j, k + 1)] += 1

    for seq in machine_seqs.values():
        for i in range(len(seq) - 1):
            adj[seq[i]].append(seq[i + 1])
            in_degree[seq[i + 1]] += 1

    queue = deque(op for op in all_ops if in_degree[op] == 0)
    topo = []
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


def _get_makespan(
    instance: JobShopInstance,
    machine_seqs: dict[int, list[tuple[int, int]]],
) -> tuple[int, dict[tuple[int, int], int] | None]:
    """Compute makespan from machine sequences."""
    st = _build_start_times(instance, machine_seqs)
    if st is None:
        return float("inf"), None
    op_pt = {}
    for j in range(instance.n):
        for k, (_, pt) in enumerate(instance.jobs[j]):
            op_pt[(j, k)] = pt
    ms = max(st[op] + op_pt[op] for op in st) if st else 0
    return ms, st


def _get_critical_neighbors(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
    machine_seqs: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Find swappable pairs on critical path (N1 neighborhood)."""
    op_pt = {}
    for j in range(instance.n):
        for k, (_, pt) in enumerate(instance.jobs[j]):
            op_pt[(j, k)] = pt

    completion = {op: start_times[op] + op_pt[op] for op in start_times}
    makespan = max(completion.values()) if completion else 0

    critical_ops = set()
    machine_pred = {}
    for seq in machine_seqs.values():
        for i in range(len(seq) - 1):
            machine_pred[seq[i + 1]] = seq[i]

    stack = [op for op, c in completion.items() if c == makespan]
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


def vns(
    instance: JobShopInstance,
    max_iterations: int = 1000,
    k_max: int = 3,
    time_limit: float | None = None,
    seed: int | None = None,
) -> JobShopSolution:
    """Solve JSP using Variable Neighborhood Search.

    Args:
        instance: JSP instance.
        max_iterations: Maximum VNS iterations.
        k_max: Number of neighborhood structures.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best JobShopSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()

    # Initialize with best dispatching rule
    best_init = None
    for rule in ["spt", "mwr", "lpt", "lwr"]:
        sol = dispatching_rule(instance, rule=rule)
        if best_init is None or sol.makespan < best_init.makespan:
            best_init = sol

    machine_seqs = _extract_machine_sequences(instance, best_init.start_times)
    current_ms, current_st = _get_makespan(instance, machine_seqs)

    best_ms = current_ms
    best_seqs = {m: list(s) for m, s in machine_seqs.items()}
    best_st = dict(current_st)

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        k = 1
        while k <= k_max:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            # ── Shaking ──────────────────────────────────────────────────
            shaken_seqs = {m: list(s) for m, s in machine_seqs.items()}

            if k == 1:
                # N1: swap adjacent on critical path
                neighbors = _get_critical_neighbors(
                    instance, current_st, shaken_seqs,
                )
                if neighbors:
                    mach, pos = neighbors[rng.integers(0, len(neighbors))]
                    seq = shaken_seqs[mach]
                    seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
            elif k == 2:
                # N2: insert within same machine
                machines = [m for m, s in shaken_seqs.items() if len(s) >= 2]
                if machines:
                    mach = machines[rng.integers(0, len(machines))]
                    seq = shaken_seqs[mach]
                    i = rng.integers(0, len(seq))
                    op = seq.pop(i)
                    j = rng.integers(0, len(seq) + 1)
                    seq.insert(j, op)
            else:
                # N3: multiple random swaps (larger perturbation)
                machines = [m for m, s in shaken_seqs.items() if len(s) >= 2]
                n_swaps = min(3, len(machines))
                for _ in range(n_swaps):
                    if machines:
                        mach = machines[rng.integers(0, len(machines))]
                        seq = shaken_seqs[mach]
                        if len(seq) >= 2:
                            i, j = rng.choice(len(seq), size=2, replace=False)
                            seq[i], seq[j] = seq[j], seq[i]

            # ── Local search (best-improvement N1) ───────────────────────
            local_ms, local_st = _get_makespan(instance, shaken_seqs)
            if local_st is None:
                k += 1
                continue

            improved = True
            while improved:
                improved = False
                neighbors = _get_critical_neighbors(
                    instance, local_st, shaken_seqs,
                )
                best_local_delta = 0
                best_local_move = None

                for mach, pos in neighbors:
                    seq = shaken_seqs[mach]
                    seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
                    new_ms, new_st = _get_makespan(instance, shaken_seqs)
                    seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]

                    if new_st is not None:
                        delta = new_ms - local_ms
                        if delta < best_local_delta:
                            best_local_delta = delta
                            best_local_move = (mach, pos, new_ms, new_st)

                if best_local_move is not None:
                    mach, pos, new_ms, new_st = best_local_move
                    shaken_seqs[mach][pos], shaken_seqs[mach][pos + 1] = (
                        shaken_seqs[mach][pos + 1], shaken_seqs[mach][pos],
                    )
                    local_ms = new_ms
                    local_st = new_st
                    improved = True

            # ── Move or not ──────────────────────────────────────────────
            if local_ms < current_ms:
                machine_seqs = shaken_seqs
                current_ms = local_ms
                current_st = local_st
                k = 1  # Reset to first neighborhood

                if current_ms < best_ms:
                    best_ms = current_ms
                    best_seqs = {m: list(s) for m, s in machine_seqs.items()}
                    best_st = dict(current_st)
            else:
                k += 1

    final_machine_sequences = build_machine_sequences(instance, best_st)
    return JobShopSolution(
        start_times=best_st,
        makespan=best_ms,
        machine_sequences=final_machine_sequences,
    )


if __name__ == "__main__":
    from instance import ft06

    print("=== VNS on ft06 (optimal=55) ===\n")
    inst = ft06()
    sol = vns(inst, max_iterations=200, seed=42)
    print(f"VNS makespan: {sol.makespan}")

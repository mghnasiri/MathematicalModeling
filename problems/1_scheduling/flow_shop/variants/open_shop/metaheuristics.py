"""
Open Shop Scheduling — Metaheuristics.

Algorithms:
    - Simulated Annealing with operation reordering.

References:
    Liaw, C.F. (2000). A hybrid genetic algorithm for the open shop
    scheduling problem. European Journal of Operational Research, 124(1),
    28-42. https://doi.org/10.1016/S0377-2217(99)00168-X
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("os_instance_m", os.path.join(_this_dir, "instance.py"))
OpenShopInstance = _inst.OpenShopInstance
OpenShopSolution = _inst.OpenShopSolution

_heur = _load_mod("os_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_open_shop = _heur.greedy_open_shop


def _decode(instance: OpenShopInstance,
            job_machine_order: list[list[int]]) -> OpenShopSolution:
    """Decode machine orderings into a schedule via greedy scheduling."""
    n, m = instance.n, instance.m
    machine_avail = [0.0] * m
    job_avail = [0.0] * n
    schedule = [[] for _ in range(n)]
    next_op = [0] * n

    import heapq
    ops = []
    for j in range(n):
        mach = job_machine_order[j][0]
        ops.append((0.0, j, mach))
    heapq.heapify(ops)

    count = 0
    while ops and count < n * m:
        _, j, mach = heapq.heappop(ops)
        dur = instance.processing_times[j][mach]
        st = max(job_avail[j], machine_avail[mach])
        schedule[j].append((mach, st))
        machine_avail[mach] = st + dur
        job_avail[j] = st + dur
        next_op[j] += 1
        count += 1
        if next_op[j] < m:
            next_mach = job_machine_order[j][next_op[j]]
            est = max(job_avail[j], machine_avail[next_mach])
            heapq.heappush(ops, (est, j, next_mach))

    makespan = instance.makespan(schedule)
    return OpenShopSolution(schedule=schedule, makespan=makespan)


def simulated_annealing(
    instance: OpenShopInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> OpenShopSolution:
    """SA for open shop scheduling.

    Representation: per-job machine orderings. Moves: swap two machines
    in a job's ordering.

    Args:
        instance: Open shop instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        OpenShopSolution.
    """
    rng = np.random.default_rng(seed)
    n, m = instance.n, instance.m

    # Initialize: LPT ordering per job
    orders = []
    for j in range(n):
        order = sorted(range(m),
                       key=lambda k: instance.processing_times[j][k],
                       reverse=True)
        orders.append(order)

    sol = _decode(instance, orders)
    cost = sol.makespan

    best_orders = [list(o) for o in orders]
    best_cost = cost

    temp = best_cost * 0.15
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_orders = [list(o) for o in orders]
        move = rng.integers(0, 2)

        if move == 0:
            # Swap two machines in a random job's ordering
            j = int(rng.integers(0, n))
            i1 = int(rng.integers(0, m))
            i2 = int(rng.integers(0, m - 1))
            if i2 >= i1:
                i2 += 1
            new_orders[j][i1], new_orders[j][i2] = \
                new_orders[j][i2], new_orders[j][i1]
        else:
            # Insertion move in a random job's ordering
            j = int(rng.integers(0, n))
            i = int(rng.integers(0, m))
            k = int(rng.integers(0, m - 1))
            mach = new_orders[j].pop(i)
            new_orders[j].insert(k, mach)

        new_sol = _decode(instance, new_orders)
        new_cost = new_sol.makespan
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            orders = new_orders
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_orders = [list(o) for o in orders]

        temp *= cooling_rate

    return _decode(instance, best_orders)


if __name__ == "__main__":
    from instance import small_os_3x3

    inst = small_os_3x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

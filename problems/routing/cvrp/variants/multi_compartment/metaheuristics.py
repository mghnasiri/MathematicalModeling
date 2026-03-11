"""
Multi-Compartment VRP — Metaheuristics.

Algorithms:
    - Simulated Annealing with relocate and swap moves.

References:
    Derigs, U., Gottlieb, J. & Kalkoff, J. (2011). Vehicle routing with
    compartments: Applications, modelling and heuristics. OR Spectrum, 33(4),
    885-914. https://doi.org/10.1007/s00291-009-0175-6
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


_inst = _load_mod("mcvrp_instance_m", os.path.join(_this_dir, "instance.py"))
MCVRPInstance = _inst.MCVRPInstance
MCVRPSolution = _inst.MCVRPSolution
validate_solution = _inst.validate_solution

_heur = _load_mod("mcvrp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
savings_mcvrp = _heur.savings_mcvrp


def _route_feasible(instance: MCVRPInstance, route: list[int]) -> bool:
    loads = instance.route_loads(route)
    return bool(np.all(loads <= instance.compartment_capacities + 1e-6))


def simulated_annealing(
    instance: MCVRPInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> MCVRPSolution:
    """SA for MCVRP with relocate, swap, and 2-opt moves.

    Args:
        instance: MCVRP instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        MCVRPSolution.
    """
    rng = np.random.default_rng(seed)

    init = savings_mcvrp(instance)
    routes = [list(r) for r in init.routes]
    cost = init.total_distance

    best_routes = [list(r) for r in routes]
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_routes = [list(r) for r in routes]
        move = rng.integers(0, 3)

        if move == 0 and len(new_routes) >= 2:
            # Inter-route relocate
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            if not new_routes[ri]:
                temp *= cooling_rate
                continue
            idx = int(rng.integers(0, len(new_routes[ri])))
            cust = new_routes[ri].pop(idx)
            pos = int(rng.integers(0, len(new_routes[rj]) + 1))
            new_routes[rj].insert(pos, cust)

        elif move == 1 and len(new_routes) >= 2:
            # Inter-route swap
            ri = int(rng.integers(0, len(new_routes)))
            rj = int(rng.integers(0, len(new_routes) - 1))
            if rj >= ri:
                rj += 1
            if not new_routes[ri] or not new_routes[rj]:
                temp *= cooling_rate
                continue
            i1 = int(rng.integers(0, len(new_routes[ri])))
            i2 = int(rng.integers(0, len(new_routes[rj])))
            new_routes[ri][i1], new_routes[rj][i2] = \
                new_routes[rj][i2], new_routes[ri][i1]

        elif move == 2 and len(new_routes) > 0:
            # Intra-route 2-opt
            ri = int(rng.integers(0, len(new_routes)))
            r = new_routes[ri]
            if len(r) >= 3:
                i = int(rng.integers(0, len(r) - 1))
                j = int(rng.integers(i + 1, len(r)))
                r[i:j + 1] = reversed(r[i:j + 1])

        # Remove empty routes
        new_routes = [r for r in new_routes if r]

        # Check feasibility
        feasible = True
        for r in new_routes:
            if not _route_feasible(instance, r):
                feasible = False
                break

        if not feasible:
            temp *= cooling_rate
            continue

        new_cost = sum(instance.route_distance(r) for r in new_routes)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            routes = new_routes
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]

        temp *= cooling_rate

    return MCVRPSolution(routes=best_routes, total_distance=best_cost)


if __name__ == "__main__":
    from instance import small_mcvrp_6

    inst = small_mcvrp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

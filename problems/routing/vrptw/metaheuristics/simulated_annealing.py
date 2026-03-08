"""
Simulated Annealing for VRPTW.

Problem: VRPTW (Vehicle Routing Problem with Time Windows)

Neighborhoods:
- Relocate: move a customer from one route to another (feasibility-checked)
- Swap: exchange customers between two routes (feasibility-checked)

All moves are checked for time window and capacity feasibility before
acceptance. Warm-started with Solomon's insertion heuristic.

References:
    Chiang, W.-C. & Russell, R.A. (1996). Simulated annealing
    metaheuristics for the vehicle routing problem with time windows.
    Annals of Operations Research, 63(1), 3-27.
    https://doi.org/10.1007/BF02601637

    Czech, Z.J. & Czarnas, P. (2002). Parallel simulated annealing
    for the vehicle routing problem with time windows. Proceedings
    of the 10th Euromicro Workshop on Parallel, Distributed and
    Network-based Processing, 376-383.
    https://doi.org/10.1109/EMPDP.2002.994313
"""

from __future__ import annotations

import os
import math
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module("vrptw_instance_sa", os.path.join(_parent_dir, "instance.py"))
VRPTWInstance = _inst.VRPTWInstance
VRPTWSolution = _inst.VRPTWSolution


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in routes]


def simulated_annealing(
    instance: VRPTWInstance,
    initial_routes: list[list[int]] | None = None,
    max_iterations: int = 50_000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
) -> VRPTWSolution:
    """Solve VRPTW using simulated annealing.

    Args:
        instance: A VRPTWInstance.
        initial_routes: Starting routes. If None, uses Solomon insertion.
        max_iterations: Maximum number of iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        seed: Random seed for reproducibility.

    Returns:
        VRPTWSolution with the best routes found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Initialize
    if initial_routes is None:
        _si_mod = _load_module(
            "vrptw_si_sa",
            os.path.join(_parent_dir, "heuristics", "solomon_insertion.py"))
        init_sol = _si_mod.solomon_insertion(instance)
        routes = _copy_routes(init_sol.routes)
    else:
        routes = _copy_routes(initial_routes)

    current_cost = instance.total_distance(routes)
    best_routes = _copy_routes(routes)
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = best_cost * 0.05

    temp = initial_temp

    for iteration in range(max_iterations):
        non_empty = [i for i, r in enumerate(routes) if r]
        if len(non_empty) < 1:
            continue

        move_type = rng.integers(0, 2)
        new_routes = _copy_routes(routes)

        if move_type == 0 and len(non_empty) >= 1:
            # Relocate
            src_idx = rng.choice(non_empty)
            src_route = new_routes[src_idx]
            if not src_route:
                continue
            cust_pos = rng.integers(0, len(src_route))
            customer = src_route[cust_pos]

            # Build candidate route indices
            candidates = [i for i in range(len(new_routes)) if i != src_idx]
            if not candidates:
                continue
            dst_idx = rng.choice(candidates)
            dst_route = new_routes[dst_idx]

            # Remove customer from source
            src_route.pop(cust_pos)

            # Try inserting at random position
            ins_pos = rng.integers(0, len(dst_route) + 1)
            dst_route.insert(ins_pos, customer)

            # Check feasibility
            if not instance.route_feasible(dst_route):
                continue

        elif move_type == 1 and len(non_empty) >= 2:
            # Swap
            r1_idx, r2_idx = rng.choice(non_empty, size=2, replace=False)
            r1 = new_routes[r1_idx]
            r2 = new_routes[r2_idx]
            if not r1 or not r2:
                continue

            p1 = rng.integers(0, len(r1))
            p2 = rng.integers(0, len(r2))

            r1[p1], r2[p2] = r2[p2], r1[p1]

            # Check feasibility of both routes
            if not instance.route_feasible(r1) or not instance.route_feasible(r2):
                continue
        else:
            continue

        # Remove empty routes
        new_routes = [r for r in new_routes if r]

        new_cost = instance.total_distance(new_routes)
        delta = new_cost - current_cost

        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            routes = new_routes
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = _copy_routes(routes)

        temp *= cooling_rate

    best_routes = [r for r in best_routes if r]
    return VRPTWSolution(
        routes=best_routes,
        distance=instance.total_distance(best_routes),
    )


if __name__ == "__main__":
    from instance import solomon_c101_mini, tight_tw5

    print("=== Simulated Annealing for VRPTW ===\n")

    for name, inst_fn in [
        ("solomon_c101_mini", solomon_c101_mini),
        ("tight_tw5", tight_tw5),
    ]:
        inst = inst_fn()
        sol = simulated_annealing(inst, seed=42)
        print(f"{name}: {sol}")

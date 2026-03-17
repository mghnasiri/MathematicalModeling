"""
Simulated Annealing for Uncapacitated Facility Location (UFLP).

Problem: UFLP (Uncapacitated Facility Location)

Neighborhoods:
- Toggle: open a closed facility or close an open one (keeping >= 1 open)
- Swap: close one facility and open another

Warm-started with greedy-add heuristic.

References:
    Ghosh, D. (2003). Neighborhood search heuristics for the
    uncapacitated facility location problem. European Journal of
    Operational Research, 150(1), 150-162.
    https://doi.org/10.1016/S0377-2217(02)00504-6

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import os
import sys
import math
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("fl_instance_sa", os.path.join(_parent_dir, "instance.py"))
FacilityLocationInstance = _inst.FacilityLocationInstance
FacilityLocationSolution = _inst.FacilityLocationSolution


def _assign_and_cost(
    instance: FacilityLocationInstance, open_set: set[int]
) -> tuple[list[int], float]:
    """Assign customers and compute total cost."""
    assignments = []
    total = sum(instance.fixed_costs[i] for i in open_set)
    for j in range(instance.n):
        best = min(open_set, key=lambda i: instance.assignment_costs[i][j])
        assignments.append(best)
        total += instance.assignment_costs[best][j]
    return assignments, total


def simulated_annealing(
    instance: FacilityLocationInstance,
    max_iterations: int = 30_000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
) -> FacilityLocationSolution:
    """Solve UFLP using simulated annealing.

    Args:
        instance: A FacilityLocationInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. Auto-calibrated if None.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.

    Returns:
        FacilityLocationSolution.
    """
    rng = np.random.default_rng(seed)
    m = instance.m

    # Warm-start with greedy add
    _gr_mod = _load_mod(
        "fl_gr_sa",
        os.path.join(_parent_dir, "heuristics", "greedy_facility.py"),
    )
    init_sol = _gr_mod.greedy_add(instance)
    open_set = set(init_sol.open_facilities)

    _, current_cost = _assign_and_cost(instance, open_set)
    best_open = set(open_set)
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = best_cost * 0.05

    temp = initial_temp

    for _ in range(max_iterations):
        move_type = rng.integers(0, 2)
        new_open = set(open_set)

        if move_type == 0:
            # Toggle a random facility
            fac = rng.integers(0, m)
            if fac in new_open:
                if len(new_open) > 1:
                    new_open.remove(fac)
                else:
                    continue
            else:
                new_open.add(fac)
        else:
            # Swap: close one, open another
            if len(new_open) < 1 or len(new_open) >= m:
                continue
            close_fac = rng.choice(list(new_open))
            closed = list(set(range(m)) - new_open)
            open_fac = rng.choice(closed)
            new_open.remove(close_fac)
            new_open.add(open_fac)

        _, new_cost = _assign_and_cost(instance, new_open)
        delta = new_cost - current_cost

        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            open_set = new_open
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_open = set(open_set)

        temp *= cooling_rate

    assignments, total_cost = _assign_and_cost(instance, best_open)
    return FacilityLocationSolution(
        open_facilities=sorted(best_open),
        assignments=assignments,
        cost=total_cost,
    )


if __name__ == "__main__":
    from instance import small_uflp_3_5

    inst = small_uflp_3_5()
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: {sol}")

"""
Simulated Annealing for Chance-Constrained Facility Location

Neighborhood moves: toggle facility open/closed, swap open/closed pair.
Penalty for chance constraint violations.

References:
    - Snyder, L.V. (2006). Facility location under uncertainty. IIE Trans.,
      38(7), 547-564. https://doi.org/10.1080/07408170500216480
"""
from __future__ import annotations

import sys
import os
import importlib.util
import math

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("ccfl_instance", os.path.join(_base, "instance.py"))
_heur = _load_parent("ccfl_greedy", os.path.join(_base, "heuristics", "greedy_ccfl.py"))

CCFLInstance = _inst.CCFLInstance
CCFLSolution = _inst.CCFLSolution
_assign_customers = _heur._assign_customers


def simulated_annealing(
    instance: CCFLInstance,
    max_iterations: int = 3000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    penalty_weight: float = 500.0,
    seed: int = 42,
) -> CCFLSolution:
    """SA for chance-constrained facility location.

    Args:
        instance: CCFLInstance.
        max_iterations: Number of iterations.
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor.
        penalty_weight: Penalty for chance constraint violations.
        seed: Random seed.

    Returns:
        Best CCFLSolution found.
    """
    rng = np.random.default_rng(seed)
    m = instance.n_facilities

    # Initialize with greedy
    init_sol = _heur.greedy_open(instance)
    current_open = set(init_sol.open_facilities)

    def evaluate(open_set):
        open_list = sorted(open_set)
        if not open_list:
            return float("inf"), None, None, 1.0
        assign = _assign_customers(instance, open_list)
        cost = instance.total_cost(open_list, assign)
        max_viol = 0.0
        for i in open_list:
            custs = [j for j in range(instance.n_customers) if assign[j] == i]
            viol = instance.capacity_violation_prob(i, custs)
            max_viol = max(max_viol, viol)
        penalty = penalty_weight * max(0, max_viol - instance.alpha)
        return cost + penalty, assign, open_list, max_viol

    current_obj, current_assign, _, current_viol = evaluate(current_open)
    best_open = set(current_open)
    best_obj = current_obj
    best_assign = current_assign
    best_viol = current_viol
    temp = initial_temp

    for _ in range(max_iterations):
        neighbor_open = set(current_open)

        r = rng.random()
        if r < 0.4:
            # Toggle a random facility
            i = rng.integers(m)
            if i in neighbor_open:
                if len(neighbor_open) > 1:
                    neighbor_open.remove(i)
            else:
                neighbor_open.add(i)
        elif r < 0.8:
            # Swap: close one, open another
            if neighbor_open and len(neighbor_open) < m:
                close_i = rng.choice(list(neighbor_open))
                closed = set(range(m)) - neighbor_open
                open_i = rng.choice(list(closed))
                neighbor_open.remove(close_i)
                neighbor_open.add(open_i)
        else:
            # Add a facility
            closed = set(range(m)) - neighbor_open
            if closed:
                neighbor_open.add(rng.choice(list(closed)))

        if not neighbor_open:
            continue

        n_obj, n_assign, n_list, n_viol = evaluate(neighbor_open)
        delta = n_obj - current_obj

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_open = neighbor_open
            current_obj = n_obj

            if n_obj < best_obj and n_viol <= instance.alpha + 1e-9:
                best_open = set(neighbor_open)
                best_obj = n_obj
                best_assign = n_assign
                best_viol = n_viol

        temp *= cooling_rate

    open_list = sorted(best_open)
    if best_assign is None:
        best_assign = _assign_customers(instance, open_list)
        best_viol = 0.0

    return CCFLSolution(
        open_facilities=open_list,
        assignments=best_assign,
        total_cost=instance.total_cost(open_list, best_assign),
        max_violation_prob=best_viol,
    )

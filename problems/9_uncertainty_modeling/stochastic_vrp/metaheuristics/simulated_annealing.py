"""
Simulated Annealing for Stochastic VRP

Uses relocate and swap inter-route moves with a combined objective:
total distance + expected recourse cost + penalty for overflow violations.

References:
    - Gendreau, M., Laporte, G. & Séguin, R. (1996). Stochastic vehicle
      routing. EJOR, 88(1), 3-12.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import math
import copy

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("svrp_instance", os.path.join(_base, "instance.py"))
_heur = _load_parent("svrp_cw", os.path.join(_base, "heuristics", "chance_constrained_cw.py"))

StochasticVRPInstance = _inst.StochasticVRPInstance
StochasticVRPSolution = _inst.StochasticVRPSolution


def simulated_annealing(
    instance: StochasticVRPInstance,
    max_iterations: int = 5000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    penalty_weight: float = 200.0,
    seed: int = 42,
) -> StochasticVRPSolution:
    """SA for stochastic VRP with overflow penalty.

    Args:
        instance: StochasticVRPInstance.
        max_iterations: Number of SA iterations.
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor.
        penalty_weight: Penalty for overflow constraint violation.
        seed: Random seed.

    Returns:
        Best StochasticVRPSolution found.
    """
    rng = np.random.default_rng(seed)

    # Initialize with savings heuristic
    init_sol = _heur.chance_constrained_savings(instance)
    current_routes = [list(r) for r in init_sol.routes]

    def evaluate(routes):
        dist = instance.solution_total_distance(routes)
        recourse = instance.expected_recourse_cost(routes)
        max_overflow = 0.0
        total_penalty = 0.0
        for r in routes:
            if r:
                op = instance.route_overflow_probability(r)
                max_overflow = max(max_overflow, op)
                if op > instance.alpha:
                    total_penalty += penalty_weight * (op - instance.alpha)
        return dist + recourse + total_penalty, max_overflow

    current_obj, current_overflow = evaluate(current_routes)
    best_routes = copy.deepcopy(current_routes)
    best_obj = current_obj
    best_overflow = current_overflow
    temp = initial_temp

    for _ in range(max_iterations):
        neighbor = copy.deepcopy(current_routes)
        non_empty = [i for i, r in enumerate(neighbor) if r]

        if len(non_empty) < 1:
            break

        move = rng.random()

        if move < 0.5 and len(non_empty) >= 2:
            # Relocate: move a customer from one route to another
            r1_idx = rng.choice(non_empty)
            r2_idx = rng.choice(non_empty)
            while r2_idx == r1_idx and len(non_empty) > 1:
                r2_idx = rng.choice(non_empty)

            if neighbor[r1_idx]:
                pos1 = rng.integers(len(neighbor[r1_idx]))
                cust = neighbor[r1_idx].pop(pos1)
                pos2 = rng.integers(len(neighbor[r2_idx]) + 1)
                neighbor[r2_idx].insert(pos2, cust)
        elif move < 0.8 and len(non_empty) >= 2:
            # Swap: exchange customers between two routes
            r1_idx = rng.choice(non_empty)
            r2_idx = rng.choice(non_empty)
            while r2_idx == r1_idx and len(non_empty) > 1:
                r2_idx = rng.choice(non_empty)

            if neighbor[r1_idx] and neighbor[r2_idx]:
                p1 = rng.integers(len(neighbor[r1_idx]))
                p2 = rng.integers(len(neighbor[r2_idx]))
                neighbor[r1_idx][p1], neighbor[r2_idx][p2] = (
                    neighbor[r2_idx][p2], neighbor[r1_idx][p1]
                )
        else:
            # Intra-route 2-opt
            r_idx = rng.choice(non_empty)
            route = neighbor[r_idx]
            if len(route) >= 3:
                i = rng.integers(len(route) - 1)
                j = rng.integers(i + 1, len(route))
                route[i:j + 1] = route[i:j + 1][::-1]

        # Remove empty routes
        neighbor = [r for r in neighbor if r]

        n_obj, n_overflow = evaluate(neighbor)
        delta = n_obj - current_obj

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current_routes = neighbor
            current_obj = n_obj

            if n_obj < best_obj and n_overflow <= instance.alpha + 1e-9:
                best_routes = copy.deepcopy(neighbor)
                best_obj = n_obj
                best_overflow = n_overflow

        temp *= cooling_rate

    total_dist = instance.solution_total_distance(best_routes)
    recourse = instance.expected_recourse_cost(best_routes)

    return StochasticVRPSolution(
        routes=best_routes,
        total_distance=total_dist,
        expected_total_cost=total_dist + recourse,
        max_overflow_prob=best_overflow,
        n_routes=len(best_routes),
    )

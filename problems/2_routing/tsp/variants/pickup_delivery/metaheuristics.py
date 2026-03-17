"""
Simulated Annealing for Pickup and Delivery Problem (PDP).

Problem: 1-PDTSP

Uses Or-opt (relocate) and swap moves that preserve pickup-before-delivery
precedence. Moves that violate precedence are rejected.

Warm-started with nearest feasible neighbor.

Complexity: O(iterations * n) per run.

References:
    Savelsbergh, M.W.P. & Sol, M. (1995). The general pickup and delivery
    problem. Transportation Science, 29(1), 17-29.
    https://doi.org/10.1287/trsc.29.1.17

    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

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


_inst = _load_mod("pdp_instance_meta", os.path.join(_this_dir, "instance.py"))
PDPInstance = _inst.PDPInstance
PDPSolution = _inst.PDPSolution

_heur = _load_mod("pdp_heuristics_meta", os.path.join(_this_dir, "heuristics.py"))
nearest_feasible = _heur.nearest_feasible


def simulated_annealing(
    instance: PDPInstance,
    max_iterations: int = 50000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> PDPSolution:
    """Solve PDP using Simulated Annealing.

    Args:
        instance: A PDPInstance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. Auto-calibrated if None.
        cooling_rate: Geometric cooling factor.
        time_limit: Time limit in seconds.
        seed: Random seed.

    Returns:
        PDPSolution with the best feasible tour found.
    """
    rng = np.random.default_rng(seed)
    n = instance.num_locations
    np_pairs = instance.num_pairs
    start_time = time.time()

    if n <= 2:
        tour = list(range(n))
        return PDPSolution(
            tour=tour,
            distance=instance.tour_distance(tour),
            feasible=instance.precedence_feasible(tour),
        )

    init_sol = nearest_feasible(instance)
    tour = init_sol.tour[:]
    current_dist = init_sol.distance

    best_tour = tour[:]
    best_dist = current_dist

    if initial_temp is None:
        initial_temp = current_dist * 0.1

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        move = rng.integers(0, 3)
        new_tour = tour[:]

        if move == 0:
            # Relocate: move a non-depot location to a new position
            idx = rng.integers(1, n)
            loc = new_tour.pop(idx)
            new_pos = rng.integers(1, len(new_tour) + 1)
            new_tour.insert(new_pos, loc)

        elif move == 1 and n > 3:
            # Swap two non-depot locations
            i = rng.integers(1, n)
            j = rng.integers(1, n)
            if i != j:
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        elif move == 2 and np_pairs >= 1:
            # Pair relocate: move both pickup and delivery of a random pair
            pair = rng.integers(1, np_pairs + 1)
            pickup = pair
            delivery = pair + np_pairs

            p_idx = new_tour.index(pickup)
            d_idx = new_tour.index(delivery)

            # Remove both
            new_tour = [x for x in new_tour if x != pickup and x != delivery]

            # Reinsert at random valid positions
            p_pos = rng.integers(1, len(new_tour) + 1)
            new_tour.insert(p_pos, pickup)
            d_pos = rng.integers(p_pos + 1, len(new_tour) + 1)
            new_tour.insert(d_pos, delivery)

        else:
            temp *= cooling_rate
            continue

        # Check precedence
        if not instance.precedence_feasible(new_tour):
            temp *= cooling_rate
            continue

        new_dist = instance.tour_distance(new_tour)
        delta = new_dist - current_dist
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            tour = new_tour
            current_dist = new_dist

            if current_dist < best_dist - 1e-10:
                best_dist = current_dist
                best_tour = tour[:]

        temp *= cooling_rate

    return PDPSolution(
        tour=best_tour,
        distance=instance.tour_distance(best_tour),
        feasible=instance.precedence_feasible(best_tour),
    )


if __name__ == "__main__":
    inst = PDPInstance.random(num_pairs=5, seed=42)
    print(f"PDP: {inst.num_pairs} pairs, {inst.num_locations} locations")

    nf_sol = nearest_feasible(inst)
    print(f"Nearest feasible: dist={nf_sol.distance:.1f}")

    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: dist={sa_sol.distance:.1f}")

"""
Simulated Annealing for Variable-Size Bin Packing.

Problem: VS-BPP

Permutation encoding with FFD decoder. SA explores item orderings;
decoder assigns items to cheapest bins using best-fit.

Warm-started with FFD best-type heuristic.

Complexity: O(iterations * n^2) per run.

References:
    Friesen, D.K. & Langston, M.A. (1986). Variable sized bin packing.
    SIAM Journal on Computing, 15(1), 222-230.
    https://doi.org/10.1137/0215016
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


_inst = _load_mod("vsbpp_instance_meta", os.path.join(_this_dir, "instance.py"))
VSBPPInstance = _inst.VSBPPInstance
VSBPPSolution = _inst.VSBPPSolution

_heur = _load_mod("vsbpp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
ffd_best_type = _heur.ffd_best_type


def _decode(instance: VSBPPInstance, perm: list[int]) -> VSBPPSolution:
    """Decode permutation into VS-BPP solution using best-fit."""
    type_order = sorted(range(instance.num_bin_types),
                        key=lambda t: instance.bin_costs[t])

    bins: list[tuple[int, list[int], float]] = []

    for i in perm:
        size = instance.item_sizes[i]
        best_bin = -1
        best_remaining = float("inf")
        for b_idx, (btype, items, remaining) in enumerate(bins):
            if remaining >= size - 1e-10 and remaining - size < best_remaining:
                best_remaining = remaining - size
                best_bin = b_idx

        if best_bin >= 0:
            btype, items, remaining = bins[best_bin]
            items.append(i)
            bins[best_bin] = (btype, items, remaining - size)
        else:
            placed = False
            for t in type_order:
                if instance.bin_capacities[t] >= size - 1e-10:
                    bins.append((t, [i], instance.bin_capacities[t] - size))
                    placed = True
                    break
            if not placed:
                t = int(np.argmax(instance.bin_capacities))
                bins.append((t, [i], instance.bin_capacities[t] - size))

    result_bins = [(btype, items) for btype, items, _ in bins]
    total_cost = sum(instance.bin_costs[btype] for btype, _ in result_bins)
    return VSBPPSolution(bins=result_bins, total_cost=total_cost)


def simulated_annealing(
    instance: VSBPPInstance,
    max_iterations: int = 20000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> VSBPPSolution:
    """Solve VS-BPP using SA with permutation encoding."""
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    perm = sorted(range(n), key=lambda i: instance.item_sizes[i], reverse=True)
    current_sol = _decode(instance, perm)
    current_cost = current_sol.total_cost

    best_perm = perm[:]
    best_sol = current_sol
    best_cost = current_cost

    if initial_temp is None:
        initial_temp = current_cost * 0.3

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        new_perm = perm[:]
        if rng.random() < 0.5:
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            i = rng.integers(0, n)
            item = new_perm.pop(i)
            j = rng.integers(0, len(new_perm) + 1)
            new_perm.insert(j, item)

        new_sol = _decode(instance, new_perm)
        new_cost = new_sol.total_cost

        delta = new_cost - current_cost
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            perm = new_perm
            current_cost = new_cost

            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_perm = perm[:]
                best_sol = new_sol

        temp *= cooling_rate

    return best_sol


if __name__ == "__main__":
    inst = VSBPPInstance.random(n=15, seed=42)
    ffd_sol = ffd_best_type(inst)
    print(f"FFD: cost={ffd_sol.total_cost:.1f}")
    sa_sol = simulated_annealing(inst, seed=42)
    print(f"SA: cost={sa_sol.total_cost:.1f}")

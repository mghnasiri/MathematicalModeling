"""
Stochastic Flow Shop — Metaheuristics.

Algorithms:
    - Simulated Annealing with Monte Carlo makespan estimation.

References:
    Gourgand, M., Grangeon, N. & Norre, S. (2000). A review of the static
    stochastic flow-shop scheduling problem. Journal of Decision Systems,
    9(2), 1-31. https://doi.org/10.1080/12460125.2000.9736710
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


_inst = _load_mod("stoch_fs_instance_m", os.path.join(_this_dir, "instance.py"))
StochasticFlowShopInstance = _inst.StochasticFlowShopInstance
StochasticFlowShopSolution = _inst.StochasticFlowShopSolution

_heur = _load_mod("stoch_fs_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
neh_deterministic = _heur.neh_deterministic


def simulated_annealing(
    instance: StochasticFlowShopInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    num_samples: int = 20,
    seed: int | None = None,
    time_limit: float | None = None,
) -> StochasticFlowShopSolution:
    """Simulated Annealing for stochastic flow shop.

    Uses Monte Carlo sampling to estimate expected makespan.
    Swap and insertion neighborhoods.

    Args:
        instance: Stochastic flow shop instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        num_samples: MC samples per evaluation.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        StochasticFlowShopSolution.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Pre-generate sample sets (refreshed periodically)
    sample_sets = [instance.sample_times(rng) for _ in range(num_samples)]

    def avg_makespan(perm):
        total = 0.0
        for times in sample_sets:
            total += instance.makespan(perm, times)
        return total / num_samples

    # Initialize from NEH deterministic
    init = neh_deterministic(instance)
    perm = list(init.permutation)
    cost = avg_makespan(perm)

    best_perm = list(perm)
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        # Refresh samples periodically for robustness
        if it > 0 and it % 5000 == 0:
            sample_sets = [instance.sample_times(rng) for _ in range(num_samples)]
            cost = avg_makespan(perm)
            best_cost = avg_makespan(best_perm)

        new_perm = list(perm)
        move = rng.integers(0, 2)

        if move == 0:
            # Swap
            i, j = rng.integers(0, n, size=2)
            while i == j:
                j = int(rng.integers(0, n))
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insertion
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            job = new_perm.pop(i)
            new_perm.insert(j, job)

        new_cost = avg_makespan(new_perm)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            perm = new_perm
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)

        temp *= cooling_rate

    # Final evaluation with more samples
    expected = instance.expected_makespan(best_perm, num_samples=100, seed=seed)
    return StochasticFlowShopSolution(permutation=best_perm,
                                       expected_makespan=expected)


if __name__ == "__main__":
    from instance import small_stoch_fs_4x3

    inst = small_stoch_fs_4x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")

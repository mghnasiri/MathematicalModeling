"""
Stochastic Flow Shop — Heuristics.

Algorithms:
    - NEH on mean processing times (deterministic proxy).
    - Stochastic NEH with Monte Carlo evaluation.

References:
    Framinan, J.M. & Perez-Gonzalez, P. (2015). On heuristic solutions
    for the stochastic flowshop scheduling problem. European Journal of
    Operational Research, 246(2), 413-420.
    https://doi.org/10.1016/j.ejor.2015.05.006

    Nawaz, M., Enscore, E.E. & Ham, I. (1983). A heuristic algorithm for
    the m-machine, n-job flow-shop sequencing problem. OMEGA, 11(1), 91-95.
    https://doi.org/10.1016/0305-0483(83)90088-9
"""

from __future__ import annotations

import sys
import os
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


_inst = _load_mod("stoch_fs_instance_h", os.path.join(_this_dir, "instance.py"))
StochasticFlowShopInstance = _inst.StochasticFlowShopInstance
StochasticFlowShopSolution = _inst.StochasticFlowShopSolution


def neh_deterministic(instance: StochasticFlowShopInstance) -> StochasticFlowShopSolution:
    """NEH heuristic using mean processing times as deterministic proxy.

    Args:
        instance: Stochastic flow shop instance.

    Returns:
        StochasticFlowShopSolution.
    """
    # Sort jobs by decreasing total mean processing time
    total_times = instance.mean_times.sum(axis=1)
    sorted_jobs = sorted(range(instance.n), key=lambda j: total_times[j],
                         reverse=True)

    perm = [sorted_jobs[0]]
    for j in sorted_jobs[1:]:
        best_perm = None
        best_ms = float("inf")
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [j] + perm[pos:]
            ms = instance.deterministic_makespan(trial)
            if ms < best_ms:
                best_ms = ms
                best_perm = trial
        perm = best_perm

    expected = instance.expected_makespan(perm, num_samples=50, seed=0)
    return StochasticFlowShopSolution(permutation=perm,
                                       expected_makespan=expected)


def neh_stochastic(
    instance: StochasticFlowShopInstance,
    num_samples: int = 30,
    seed: int | None = None,
) -> StochasticFlowShopSolution:
    """NEH heuristic with Monte Carlo evaluation at each insertion.

    Uses sampled makespans instead of deterministic proxy for position
    selection, yielding better solutions for high-variance instances.

    Args:
        instance: Stochastic flow shop instance.
        num_samples: Samples per evaluation.
        seed: Random seed.

    Returns:
        StochasticFlowShopSolution.
    """
    rng = np.random.default_rng(seed)

    # Pre-sample processing times
    all_samples = [instance.sample_times(rng) for _ in range(num_samples)]

    def avg_makespan(perm):
        total = 0.0
        for times in all_samples:
            total += instance.makespan(perm, times)
        return total / num_samples

    # Sort by decreasing total mean
    total_times = instance.mean_times.sum(axis=1)
    sorted_jobs = sorted(range(instance.n), key=lambda j: total_times[j],
                         reverse=True)

    perm = [sorted_jobs[0]]
    for j in sorted_jobs[1:]:
        best_perm = None
        best_ms = float("inf")
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [j] + perm[pos:]
            ms = avg_makespan(trial)
            if ms < best_ms:
                best_ms = ms
                best_perm = trial
        perm = best_perm

    expected = instance.expected_makespan(perm, num_samples=100, seed=seed)
    return StochasticFlowShopSolution(permutation=perm,
                                       expected_makespan=expected)


if __name__ == "__main__":
    from instance import small_stoch_fs_4x3

    inst = small_stoch_fs_4x3()
    sol1 = neh_deterministic(inst)
    print(f"NEH (det): {sol1}")

    sol2 = neh_stochastic(inst, seed=42)
    print(f"NEH (stoch): {sol2}")

"""
Simulated Annealing (SA) for Fm | prmu | Cmax

A trajectory-based metaheuristic inspired by the physical annealing process
in metallurgy. Unlike local search, SA can escape local optima by accepting
worse solutions with a probability that decreases over time (temperature).

Algorithm:
    1. Generate initial solution (NEH)
    2. Set initial temperature T₀ (auto-calibrated or user-specified)
    3. Repeat until stopping criterion:
        a. Generate a neighbor (random swap or insertion)
        b. Compute Δ = f(neighbor) - f(current)
        c. If Δ < 0: accept (improvement)
        d. If Δ ≥ 0: accept with probability exp(-Δ/T)
        e. Update temperature: T ← α·T  (geometric cooling)
    4. Return best solution found

Key Parameters:
    - T₀ (initial temperature): Controls initial acceptance of bad moves.
      Too high → random walk. Too low → greedy local search.
    - α (cooling rate): 0.95-0.999. Controls how fast exploration → exploitation.
      When time_limit is set, α is auto-calibrated so T reaches min_temp
      exactly when time runs out.
    - Neighborhood: insertion moves tend to work better than swaps for PFSP.

The Metropolis criterion exp(-Δ/T) is the core mechanism:
    - At high T: exp(-Δ/T) ≈ 1 → accept almost everything (exploration)
    - At low T: exp(-Δ/T) ≈ 0 → only accept improvements (exploitation)
    - At T=0: pure greedy descent

Notation: Fm | prmu | Cmax
Complexity: Depends on cooling schedule; each iteration is O(n * m)
Reference: Osman, I.H. & Potts, C.N. (1989). "Simulated Annealing for
           Permutation Flow-Shop Scheduling"
           Omega, 17(6):551-557.

           Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). "Optimization
           by Simulated Annealing"
           Science, 220(4598):671-680.
"""

from __future__ import annotations
import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def simulated_annealing(
    instance: FlowShopInstance,
    cooling_rate: float = 0.995,
    initial_temp: float | None = None,
    min_temp: float = 0.1,
    moves_per_temp: int | None = None,
    neighborhood: str = "insertion",
    time_limit: float | None = None,
    max_iterations: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> FlowShopSolution:
    """
    Apply Simulated Annealing to the permutation flow shop.

    Args:
        instance: A FlowShopInstance.
        cooling_rate: Temperature multiplier per step (α). Default: 0.995.
            When time_limit is set and cooling_rate is at default, α is
            auto-calibrated so temperature reaches min_temp at end of budget.
        initial_temp: Starting temperature. If None, auto-calibrated so
            ~80% of worsening moves are accepted initially.
        min_temp: Stop when temperature drops below this. Default: 0.1.
        moves_per_temp: Number of moves to try at each temperature level.
            Default: n (short epochs for faster cooling progression).
        neighborhood: "insertion" (stronger) or "swap" (faster). Default: "insertion".
        time_limit: Maximum wall-clock seconds. Default: None (unlimited).
        max_iterations: Maximum total moves. Default: None (unlimited).
        seed: Random seed for reproducibility.
        verbose: Print progress information.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Use shorter epoch length for faster cooling progression
    if moves_per_temp is None:
        moves_per_temp = n

    # Initialize with NEH
    neh_sol = neh(instance)
    current_perm = list(neh_sol.permutation)
    current_ms = neh_sol.makespan
    best_perm = list(current_perm)
    best_ms = current_ms

    # Auto-calibrate initial temperature if not provided
    if initial_temp is None:
        initial_temp = _calibrate_temperature(
            instance, current_perm, current_ms, neighborhood, rng
        )

    # If time_limit is set, compute adaptive cooling rate to ensure
    # temperature reaches min_temp by the end of the time budget.
    if time_limit is not None and cooling_rate == 0.995:
        # Probe speed: time a small batch of moves
        t0_probe = time.time()
        probe_perm = list(current_perm)
        for _ in range(min(100, n * 5)):
            if neighborhood == "insertion":
                probe_perm, _, _ = _random_insertion(probe_perm, rng)
            else:
                probe_perm, _, _ = _random_swap(probe_perm, rng)
            compute_makespan(instance, probe_perm)
        probe_time = time.time() - t0_probe
        moves_per_sec = min(100, n * 5) / max(probe_time, 1e-6)

        # Estimate total cooling steps
        total_est_moves = moves_per_sec * time_limit * 0.9  # 90% budget
        n_cooling_steps = max(total_est_moves / moves_per_temp, 10)

        # Set α so T reaches min_temp: min_temp = T₀ × α^steps
        ratio = min_temp / max(initial_temp, 1.0)
        if ratio > 0:
            cooling_rate = ratio ** (1.0 / n_cooling_steps)
            cooling_rate = max(0.9, min(cooling_rate, 0.9999))

    T = initial_temp
    total_moves = 0
    accepted = 0
    start_time = time.time()

    if verbose:
        print(f"SA: T₀={initial_temp:.1f}, α={cooling_rate:.6f}, "
              f"epoch={moves_per_temp}, neighborhood={neighborhood}")
        print(f"SA: Initial makespan={current_ms} (NEH)")

    while T > min_temp:
        # Check stopping criteria
        if time_limit is not None and time.time() - start_time >= time_limit:
            break
        if max_iterations is not None and total_moves >= max_iterations:
            break

        for _ in range(moves_per_temp):
            total_moves += 1

            # Generate neighbor
            if neighborhood == "insertion":
                neighbor, i, j = _random_insertion(current_perm, rng)
            else:
                neighbor, i, j = _random_swap(current_perm, rng)

            neighbor_ms = compute_makespan(instance, neighbor)
            delta = neighbor_ms - current_ms

            # Metropolis criterion
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_perm = neighbor
                current_ms = neighbor_ms
                accepted += 1

                if current_ms < best_ms:
                    best_perm = list(current_perm)
                    best_ms = current_ms

            # Time check inside inner loop (every n moves)
            if total_moves % n == 0:
                if time_limit is not None and time.time() - start_time >= time_limit:
                    break
                if max_iterations is not None and total_moves >= max_iterations:
                    break

        # Cool down
        T *= cooling_rate

    if verbose:
        elapsed = time.time() - start_time
        acc_rate = accepted / max(total_moves, 1) * 100
        print(f"SA: Best={best_ms}, moves={total_moves}, "
              f"accept={acc_rate:.1f}%, T_final={T:.2f}, "
              f"time={elapsed:.2f}s")

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


def _calibrate_temperature(
    instance: FlowShopInstance,
    perm: list[int],
    current_ms: int,
    neighborhood: str,
    rng: np.random.Generator,
    target_acceptance: float = 0.8,
    n_samples: int = 100,
) -> float:
    """
    Auto-calibrate T₀ so that ~target_acceptance fraction of uphill moves
    are accepted at the initial temperature.

    Uses the method: T₀ = -Δ_avg / ln(target_acceptance)
    where Δ_avg is the average cost increase of worsening neighbors.
    """
    deltas: list[int] = []

    for _ in range(n_samples):
        if neighborhood == "insertion":
            neighbor, _, _ = _random_insertion(perm, rng)
        else:
            neighbor, _, _ = _random_swap(perm, rng)

        ms = compute_makespan(instance, neighbor)
        delta = ms - current_ms
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 100.0  # Default if no worsening moves found

    avg_delta = sum(deltas) / len(deltas)
    T0 = -avg_delta / math.log(target_acceptance)
    return max(T0, 1.0)


def _random_swap(
    perm: list[int], rng: np.random.Generator
) -> tuple[list[int], int, int]:
    """Generate a neighbor by swapping two random positions."""
    n = len(perm)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    neighbor = list(perm)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor, int(i), int(j)


def _random_insertion(
    perm: list[int], rng: np.random.Generator
) -> tuple[list[int], int, int]:
    """Generate a neighbor by removing a job and reinserting elsewhere."""
    n = len(perm)
    i = int(rng.integers(n))
    j = int(rng.integers(n - 1))
    if j >= i:
        j += 1

    job = perm[i]
    neighbor = perm[:i] + perm[i + 1:]
    insert_pos = min(j, len(neighbor))
    neighbor.insert(insert_pos, job)
    return neighbor, i, j


if __name__ == "__main__":
    print("=== Simulated Annealing for PFSP ===\n")

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.neh import neh as neh_heuristic

    neh_sol = neh_heuristic(instance)
    print(f"NEH baseline:       {neh_sol.makespan}")

    sol_swap = simulated_annealing(
        instance, neighborhood="swap", time_limit=0.5, seed=42, verbose=True
    )
    print(f"SA (swap, 0.5s):    {sol_swap.makespan}\n")

    sol_insert = simulated_annealing(
        instance, neighborhood="insertion", time_limit=0.5, seed=42, verbose=True
    )
    print(f"SA (insert, 0.5s):  {sol_insert.makespan}\n")

    sol_long = simulated_annealing(
        instance, neighborhood="insertion", time_limit=2.0, seed=42, verbose=True
    )
    print(f"SA (insert, 2.0s):  {sol_long.makespan}\n")

    # Compare with IG
    from metaheuristics.iterated_greedy import iterated_greedy
    sol_ig = iterated_greedy(instance, time_limit=0.5, seed=42)
    print(f"IG (0.5s):          {sol_ig.makespan}")

"""
Simulated Annealing (SA) — Classic Metaheuristic for Fm | prmu | Cmax

Simulated Annealing is a probabilistic metaheuristic inspired by the
physical annealing process in metallurgy. Starting from an initial solution,
it iteratively explores the neighborhood by accepting improving moves
deterministically and worsening moves with a probability that decreases
over time (as the "temperature" cools).

Algorithm:
    1. Generate initial solution using NEH.
    2. Set initial temperature T_0 based on instance characteristics.
    3. Repeat until termination:
       a. Generate a neighbor by applying a random insertion move.
       b. Compute the change in objective: delta = new_ms - current_ms.
       c. If delta <= 0 (improvement): accept the move.
       d. If delta > 0 (worsening): accept with probability exp(-delta / T).
       e. Update the best solution found.
       f. Cool the temperature: T = alpha * T.
    4. Return the best solution found.

Key parameters:
    - T_0: Initial temperature (calibrated so ~40% of worsening moves
      are accepted at the start)
    - alpha: Cooling rate (typically 0.95-0.999)
    - epoch_length: Number of iterations at each temperature level
    - Termination: Time limit or maximum iterations

Neighborhood: Insertion moves are used as they are the strongest single-job
    neighborhood for PFSP (Taillard, 1990; Ruiz & Stuetzle, 2007).

Notation: Fm | prmu | Cmax
Complexity: Depends on cooling schedule; each iteration is O(n * m)
Reference: Osman, I.H. & Potts, C.N. (1989). "Simulated Annealing for
           Permutation Flow-Shop Scheduling"
           Omega, 17(6):551-557.
           DOI: 10.1016/0305-0483(89)90059-5

           Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). "Optimization
           by Simulated Annealing"
           Science, 220(4598):671-680.
           DOI: 10.1126/science.220.4598.671
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
    initial_temp_factor: float = 0.5,
    epoch_length: int | None = None,
    time_limit: float | None = None,
    max_iterations: int = 50000,
    seed: int | None = None,
) -> FlowShopSolution:
    """
    Apply Simulated Annealing to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance.
        cooling_rate: Temperature multiplier per epoch (0 < alpha < 1).
            Higher values cool slower, allowing more exploration.
        initial_temp_factor: Controls initial temperature.
            T_0 = factor * (total processing time) / (n * m).
        epoch_length: Number of iterations per temperature level.
            Default: n * (n - 1) / 2 (all possible insertion pairs).
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum total iterations (if no time_limit).
        seed: Random seed for reproducibility.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    m = instance.m
    p = instance.processing_times

    # Default epoch length: number of possible insertion moves
    if epoch_length is None:
        epoch_length = max(n * (n - 1) // 2, 10)

    # Initial solution via NEH
    initial = neh(instance)
    current_perm = list(initial.permutation)
    current_ms = initial.makespan

    best_perm = list(current_perm)
    best_ms = current_ms

    # Initial temperature calibration
    # Scale temperature relative to instance characteristics
    total_processing = float(p.sum())
    temperature = initial_temp_factor * total_processing / (n * m)

    # Minimum temperature threshold
    min_temperature = temperature * 1e-6

    start_time = time.time()
    iteration = 0
    epoch_counter = 0

    while True:
        # Check termination
        if time_limit is not None:
            if time.time() - start_time >= time_limit:
                break
        else:
            if iteration >= max_iterations:
                break

        iteration += 1
        epoch_counter += 1

        # Generate neighbor: random insertion move
        # Remove a job from position i and insert at position j
        i = rng.integers(0, n)
        j = rng.integers(0, n - 1)
        if j >= i:
            j += 1  # Ensure j != i

        # Perform the insertion
        new_perm = list(current_perm)
        job = new_perm.pop(i)
        insert_pos = j if j < i else j - 1
        # Clamp to valid range
        insert_pos = max(0, min(insert_pos, len(new_perm)))
        new_perm.insert(insert_pos, job)

        new_ms = compute_makespan(instance, new_perm)
        delta = new_ms - current_ms

        # Acceptance criterion
        if delta <= 0:
            # Always accept improvements
            current_perm = new_perm
            current_ms = new_ms
        elif temperature > 0:
            # Accept worsening moves with Boltzmann probability
            accept_prob = math.exp(-delta / temperature)
            if rng.random() < accept_prob:
                current_perm = new_perm
                current_ms = new_ms

        # Update best solution
        if current_ms < best_ms:
            best_perm = list(current_perm)
            best_ms = current_ms

        # Cool temperature at end of each epoch
        if epoch_counter >= epoch_length:
            temperature *= cooling_rate
            epoch_counter = 0

            # Stop if temperature is effectively zero
            if temperature < min_temperature:
                break

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


if __name__ == "__main__":
    print("=" * 60)
    print("Simulated Annealing — Permutation Flow Shop")
    print("=" * 60)

    # Small instance comparison
    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.cds import cds
    sol_cds = cds(instance)
    sol_neh = neh(instance)

    print(f"\nCDS  Makespan:  {sol_cds.makespan}")
    print(f"NEH  Makespan:  {sol_neh.makespan}")

    # SA with different seeds
    for s in [42, 123, 456]:
        sol_sa = simulated_annealing(instance, max_iterations=20000, seed=s)
        print(f"SA   Makespan:  {sol_sa.makespan}  (seed={s})")

    # Time-limited run on larger instance
    print("\n" + "=" * 60)
    print("Larger Instance: 50x10")
    print("=" * 60)

    large_instance = FlowShopInstance.random(n=50, m=10, seed=123)
    sol_neh_lg = neh(large_instance)
    print(f"NEH Makespan:   {sol_neh_lg.makespan}")

    t0 = time.time()
    sol_sa_lg = simulated_annealing(large_instance, time_limit=2.0, seed=42)
    elapsed = time.time() - t0
    print(f"SA  Makespan:   {sol_sa_lg.makespan}  ({elapsed:.1f}s)")

    from metaheuristics.iterated_greedy import iterated_greedy
    t0 = time.time()
    sol_ig_lg = iterated_greedy(large_instance, time_limit=2.0, seed=42)
    elapsed = time.time() - t0
    print(f"IG  Makespan:   {sol_ig_lg.makespan}  ({elapsed:.1f}s)")

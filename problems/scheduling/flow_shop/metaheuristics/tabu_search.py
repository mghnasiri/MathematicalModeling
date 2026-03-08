"""
Tabu Search (TS) for Fm | prmu | Cmax

A memory-based metaheuristic that explores the search space by always moving
to the best non-tabu neighbor (even if it worsens the objective). A short-term
memory (tabu list) prevents revisiting recently explored solutions.

Algorithm:
    1. Generate initial solution (NEH)
    2. At each iteration:
        a. Evaluate all neighbors in the neighborhood
        b. Select the best non-tabu neighbor (or any neighbor if it beats
           the global best — aspiration criterion)
        c. Make the move and add its reverse to the tabu list
        d. Update global best if improved
    3. Repeat until stopping criterion

Tabu Structure (for insertion neighborhood):
    When job j is removed from position i and inserted at position k,
    the reverse move (placing j back at position i) is declared tabu
    for 'tenure' iterations. This prevents cycling.

    tabu[job][position] = iteration + tenure

    A tabu move is allowed if it leads to a new global best
    (aspiration criterion).

Key Parameters:
    - Tabu tenure: Controls memory horizon. Following Taillard (1990):
      random tenure in [n/2, 3n/2] works well for PFSP.
    - Neighborhood: Full insertion neighborhood evaluated each iteration.

Tabu Search vs SA:
    - TS is deterministic given a neighborhood ordering
    - TS always moves to the best neighbor (intensification-oriented)
    - SA randomly accepts/rejects (exploration-oriented)
    - TS uses explicit memory; SA uses temperature as implicit memory

Notation: Fm | prmu | Cmax
Complexity: O(n^2 * m) per iteration (evaluate all insertion neighbors)
Reference: Nowicki, E. & Smutnicki, C. (1996). "A Fast Taboo Search Algorithm
           for the Permutation Flow-Shop Problem"
           European Journal of Operational Research, 91(1):160-175.

           Grabowski, J. & Wodecki, M. (2004). "A Very Fast Tabu Search Algorithm
           for the Permutation Flow Shop Problem with Makespan Criterion"
           Computers & Operations Research, 31(11):1891-1909.

           Glover, F. (1989). "Tabu Search — Part I"
           ORSA Journal on Computing, 1(3):190-206.
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def tabu_search(
    instance: FlowShopInstance,
    tenure_min: int | None = None,
    tenure_max: int | None = None,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    neighborhood: str = "insertion",
    seed: int | None = None,
    verbose: bool = False,
) -> FlowShopSolution:
    """
    Apply Tabu Search to the permutation flow shop.

    Args:
        instance: A FlowShopInstance.
        tenure_min: Minimum tabu tenure. Default: n // 2.
        tenure_max: Maximum tabu tenure. Default: 3 * n // 2.
        time_limit: Maximum wall-clock seconds.
        max_iterations: Maximum iterations. Default: 1000.
        neighborhood: "insertion" (stronger) or "swap". Default: "insertion".
        seed: Random seed for tenure randomization.
        verbose: Print progress.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    if tenure_min is None:
        tenure_min = max(n // 2, 3)
    if tenure_max is None:
        tenure_max = max(3 * n // 2, tenure_min + 2)

    # Initialize with NEH
    neh_sol = neh(instance)
    current_perm = list(neh_sol.permutation)
    current_ms = neh_sol.makespan
    best_perm = list(current_perm)
    best_ms = current_ms

    # Tabu matrix: tabu[job][position] = iteration until which this move is tabu
    tabu_matrix = np.zeros((n, n), dtype=int)

    start_time = time.time()
    no_improve_count = 0

    if verbose:
        print(f"TS: tenure=[{tenure_min},{tenure_max}], "
              f"neighborhood={neighborhood}")
        print(f"TS: Initial makespan={current_ms} (NEH)")

    for iteration in range(1, max_iterations + 1):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Find the best admissible neighbor
        best_neighbor = None
        best_neighbor_ms = float("inf")
        best_move = None  # (job, old_pos, new_pos)

        if neighborhood == "insertion":
            # Full insertion neighborhood: remove job at pos i, insert at pos j
            for i in range(n):
                job = current_perm[i]
                remaining = current_perm[:i] + current_perm[i + 1:]

                for j in range(n):
                    if j == i:
                        continue

                    candidate = remaining[:j] + [job] + remaining[j:]
                    ms = compute_makespan(instance, candidate)

                    # Check if move is tabu
                    is_tabu = tabu_matrix[job, j] >= iteration

                    # Aspiration: accept if new global best regardless of tabu
                    if ms < best_ms:
                        best_neighbor = candidate
                        best_neighbor_ms = ms
                        best_move = (job, i, j)
                    elif not is_tabu and ms < best_neighbor_ms:
                        best_neighbor = candidate
                        best_neighbor_ms = ms
                        best_move = (job, i, j)
        else:
            # Swap neighborhood
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = list(current_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    ms = compute_makespan(instance, candidate)

                    job_i = current_perm[i]
                    job_j = current_perm[j]

                    # Tabu check: is placing job_i at pos j or job_j at pos i tabu?
                    is_tabu = (tabu_matrix[job_i, j] >= iteration or
                               tabu_matrix[job_j, i] >= iteration)

                    if ms < best_ms:
                        best_neighbor = candidate
                        best_neighbor_ms = ms
                        best_move = (job_i, i, j)
                    elif not is_tabu and ms < best_neighbor_ms:
                        best_neighbor = candidate
                        best_neighbor_ms = ms
                        best_move = (job_i, i, j)

        if best_neighbor is None:
            # All moves are tabu and none improves global best — very rare
            break

        # Make the move
        current_perm = best_neighbor
        current_ms = best_neighbor_ms

        # Update tabu list: forbid reverse move
        if best_move is not None:
            job, old_pos, new_pos = best_move
            tenure = int(rng.integers(tenure_min, tenure_max + 1))
            tabu_matrix[job, old_pos] = iteration + tenure

        # Update global best
        if current_ms < best_ms:
            best_perm = list(current_perm)
            best_ms = current_ms
            no_improve_count = 0
        else:
            no_improve_count += 1

    if verbose:
        elapsed = time.time() - start_time
        print(f"TS: Best={best_ms}, iterations={iteration}, "
              f"time={elapsed:.2f}s")

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


if __name__ == "__main__":
    print("=== Tabu Search for PFSP ===\n")

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.neh import neh as neh_heuristic
    neh_sol = neh_heuristic(instance)
    print(f"NEH baseline:       {neh_sol.makespan}")

    # Test TS with insertion neighborhood
    sol_ts = tabu_search(
        instance, neighborhood="insertion",
        time_limit=0.5, seed=42, verbose=True
    )
    print(f"TS (insert, 0.5s):  {sol_ts.makespan}\n")

    # Test with swap
    sol_ts_swap = tabu_search(
        instance, neighborhood="swap",
        time_limit=0.5, seed=42, verbose=True
    )
    print(f"TS (swap, 0.5s):    {sol_ts_swap.makespan}\n")

    # Longer run
    sol_ts_long = tabu_search(
        instance, neighborhood="insertion",
        time_limit=2.0, seed=42, verbose=True
    )
    print(f"TS (insert, 2.0s):  {sol_ts_long.makespan}\n")

    # Compare with SA and IG
    from metaheuristics.simulated_annealing import simulated_annealing
    from metaheuristics.iterated_greedy import iterated_greedy

    sol_sa = simulated_annealing(instance, time_limit=0.5, seed=42)
    sol_ig = iterated_greedy(instance, time_limit=0.5, seed=42)
    print(f"SA (0.5s):          {sol_sa.makespan}")
    print(f"IG (0.5s):          {sol_ig.makespan}")

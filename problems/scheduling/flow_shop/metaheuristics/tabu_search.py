"""
Tabu Search (TS) — Classic Metaheuristic for Fm | prmu | Cmax

A fast tabu search algorithm for the permutation flow shop scheduling problem.
Maintains a short-term memory (tabu list) that forbids recently visited moves,
preventing cycling and encouraging exploration of new regions of the search space.

Algorithm:
    1. Generate initial solution using NEH.
    2. Repeat until termination:
       a. Generate all neighbors using insertion moves.
       b. Select the best non-tabu neighbor (or override tabu if it improves
          the global best — aspiration criterion).
       c. Record the reverse move in the tabu list.
       d. Update the best solution found.
    3. Return the best solution found.

Key parameters:
    - tabu_tenure: Number of iterations a move stays tabu (typically n/2 to n).
    - max_iterations: Maximum number of TS iterations.
    - neighborhood: Insertion-based (strongest single-job neighborhood for PFSP).

The tabu list stores (job, position) pairs representing the reverse of the
last applied move. A move that inserts job j at position p is tabu if
(j, original_position) is in the tabu list.

Notation: Fm | prmu | Cmax
Complexity: O(n^2 * m) per iteration (evaluate all insertion neighbors)
Reference: Nowicki, E. & Smutnicki, C. (1996). "A Fast Taboo Search Algorithm
           for the Permutation Flow-Shop Problem"
           European Journal of Operational Research, 91(1):160-175.
           DOI: 10.1016/0377-2217(95)00037-2

           Grabowski, J. & Wodecki, M. (2004). "A Very Fast Tabu Search Algorithm
           for the Permutation Flow Shop Problem with Makespan Criterion"
           Computers & Operations Research, 31(11):1891-1909.
           DOI: 10.1016/S0305-0548(03)00145-X
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
    tabu_tenure: int | None = None,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    neighborhood: str = "insertion",
    seed: int | None = None,
) -> FlowShopSolution:
    """
    Apply Tabu Search to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance.
        tabu_tenure: Number of iterations a move remains tabu.
            Default: max(5, n // 2).
        time_limit: Maximum runtime in seconds. If None, uses max_iterations.
        max_iterations: Maximum number of TS iterations (if no time_limit).
        neighborhood: Neighborhood type — "insertion" or "swap".
        seed: Random seed for tie-breaking.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Default tabu tenure
    if tabu_tenure is None:
        tabu_tenure = max(5, n // 2)

    # Initial solution via NEH
    initial = neh(instance)
    current_perm = list(initial.permutation)
    current_ms = initial.makespan

    best_perm = list(current_perm)
    best_ms = current_ms

    # Tabu list: maps (job, position) -> iteration when tabu expires
    tabu_list: dict[tuple[int, int], int] = {}

    if neighborhood == "insertion":
        search_fn = _insertion_neighborhood
    else:
        search_fn = _swap_neighborhood

    start_time = time.time()

    for iteration in range(max_iterations):
        # Check time limit
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Evaluate neighborhood and find best admissible move
        best_neighbor = None
        best_neighbor_ms = float('inf')
        best_move = None

        for neighbor_perm, move, reverse_move in search_fn(current_perm):
            neighbor_ms = compute_makespan(instance, neighbor_perm)

            # Check if move is tabu
            is_tabu = reverse_move in tabu_list and tabu_list[reverse_move] > iteration

            # Aspiration criterion: override tabu if it improves global best
            if is_tabu and neighbor_ms >= best_ms:
                continue

            if neighbor_ms < best_neighbor_ms:
                best_neighbor = neighbor_perm
                best_neighbor_ms = neighbor_ms
                best_move = move

        if best_neighbor is None:
            # All moves are tabu — accept the least-tabu move
            best_neighbor, best_neighbor_ms, best_move = _least_tabu_move(
                instance, current_perm, search_fn, tabu_list, iteration
            )

        # Apply the move
        current_perm = best_neighbor
        current_ms = best_neighbor_ms

        # Record the reverse move as tabu
        if best_move is not None:
            tabu_list[best_move] = iteration + tabu_tenure

        # Update global best
        if current_ms < best_ms:
            best_perm = list(current_perm)
            best_ms = current_ms

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


def _insertion_neighborhood(permutation: list[int]):
    """
    Generate all insertion neighbors.

    For each job at position i, remove it and try inserting at every other
    position j. Yields (new_permutation, move, reverse_move) tuples.

    The move is recorded as (job, new_position) and the reverse move is
    (job, original_position) — used for tabu tracking.

    Args:
        permutation: Current permutation.

    Yields:
        Tuples of (neighbor_perm, move, reverse_move).
    """
    n = len(permutation)
    for i in range(n):
        job = permutation[i]
        remaining = permutation[:i] + permutation[i + 1:]

        for j in range(n):
            if j == i:
                continue
            neighbor = remaining[:j] + [job] + remaining[j:]
            move = (job, j)
            reverse_move = (job, i)
            yield neighbor, move, reverse_move


def _swap_neighborhood(permutation: list[int]):
    """
    Generate all swap neighbors.

    Swaps jobs at positions i and j (i < j). The move is (i, j) and the
    reverse is also (i, j) since swaps are self-inverse, but we track
    (job_i, pos_j) for asymmetric tabu.

    Args:
        permutation: Current permutation.

    Yields:
        Tuples of (neighbor_perm, move, reverse_move).
    """
    n = len(permutation)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = list(permutation)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            move = (permutation[i], j)
            reverse_move = (permutation[j], i)
            yield neighbor, move, reverse_move


def _least_tabu_move(
    instance: FlowShopInstance,
    permutation: list[int],
    search_fn,
    tabu_list: dict[tuple[int, int], int],
    iteration: int,
) -> tuple[list[int], int, tuple[int, int] | None]:
    """
    When all moves are tabu, select the one whose tabu expires soonest.

    Args:
        instance: A FlowShopInstance.
        permutation: Current permutation.
        search_fn: Neighborhood generator function.
        tabu_list: Current tabu list.
        iteration: Current iteration number.

    Returns:
        Tuple of (best_neighbor, best_ms, best_move).
    """
    best_neighbor = list(permutation)
    best_ms = compute_makespan(instance, permutation)
    best_move = None
    earliest_expiry = float('inf')

    for neighbor_perm, move, reverse_move in search_fn(permutation):
        expiry = tabu_list.get(reverse_move, 0)
        neighbor_ms = compute_makespan(instance, neighbor_perm)

        # Prefer moves that expire soonest; break ties by makespan
        if expiry < earliest_expiry or (expiry == earliest_expiry and neighbor_ms < best_ms):
            best_neighbor = neighbor_perm
            best_ms = neighbor_ms
            best_move = move
            earliest_expiry = expiry

    return best_neighbor, best_ms, best_move


if __name__ == "__main__":
    print("=" * 60)
    print("Tabu Search — Permutation Flow Shop")
    print("=" * 60)

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.cds import cds
    sol_cds = cds(instance)
    sol_neh = neh(instance)

    print(f"\nCDS  Makespan:  {sol_cds.makespan}")
    print(f"NEH  Makespan:  {sol_neh.makespan}")

    # Tabu Search with insertion neighborhood
    sol_ts = tabu_search(instance, max_iterations=500, seed=42)
    print(f"TS   Makespan:  {sol_ts.makespan}")

    # Compare with other metaheuristics
    from metaheuristics.iterated_greedy import iterated_greedy
    from metaheuristics.simulated_annealing import simulated_annealing

    sol_ig = iterated_greedy(instance, max_iterations=500, seed=42)
    sol_sa = simulated_annealing(instance, max_iterations=5000, seed=42)

    print(f"IG   Makespan:  {sol_ig.makespan}")
    print(f"SA   Makespan:  {sol_sa.makespan}")

    # Larger instance with time limit
    print("\n" + "=" * 60)
    print("Larger Instance: 50x10")
    print("=" * 60)

    large_instance = FlowShopInstance.random(n=50, m=10, seed=123)
    sol_neh_lg = neh(large_instance)
    print(f"NEH Makespan:   {sol_neh_lg.makespan}")

    t0 = time.time()
    sol_ts_lg = tabu_search(large_instance, time_limit=2.0, seed=42)
    elapsed = time.time() - t0
    print(f"TS  Makespan:   {sol_ts_lg.makespan}  ({elapsed:.1f}s)")

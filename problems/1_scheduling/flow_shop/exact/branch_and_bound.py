"""
Branch and Bound — Exact Algorithm for Fm | prmu | Cmax

Explores the search tree of partial permutations, using lower bounds
to prune branches that cannot improve on the best known solution.

Lower Bound Strategy:
    Machine-based bound (Taillard, 1993): For each machine i, compute:
        LB_i = C_{i,last_scheduled} + sum of p_{i,j} for unscheduled jobs
               + min over unscheduled jobs of (sum of p_{k,j} for k > i)
    The overall lower bound is max over all machines i of LB_i.

    This bound considers: (1) the current partial schedule's completion,
    (2) all remaining work on machine i, and (3) the minimum time to
    get from machine i to the last machine.

Branching: Extend the partial permutation by trying each unscheduled job.
Pruning: Skip branches where lower_bound >= best_known.
Initial upper bound: NEH solution (warm start).

Notation: Fm | prmu | Cmax
Complexity: O(n!) worst case, but pruning makes it practical for n <= ~20
Reference: Taillard, E. (1993). "Benchmarks for Basic Scheduling Problems"
           Ignall, E. & Schrage, L. (1965). "Application of the Branch and
           Bound Technique to Some Flow-Shop Scheduling Problems"
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def branch_and_bound(
    instance: FlowShopInstance,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> FlowShopSolution:
    """
    Solve PFSP to optimality using Branch and Bound.

    Args:
        instance: A FlowShopInstance.
        time_limit: Maximum runtime in seconds (default: 60s).
        verbose: Print search progress.

    Returns:
        FlowShopSolution. If time limit is reached, returns the best
        solution found (may not be provably optimal).
    """
    n = instance.n
    m = instance.m
    p = instance.processing_times  # shape (m, n)

    # Warm start with NEH
    neh_sol = neh(instance)
    best_makespan = neh_sol.makespan
    best_perm = list(neh_sol.permutation)

    # Precompute: for each job, the sum of processing times from machine i
    # to the last machine. Used in the lower bound calculation.
    # tail_sum[i][j] = sum of p[k][j] for k = i..m-1
    tail_sum = np.zeros((m, n), dtype=int)
    for j in range(n):
        tail_sum[m - 1, j] = p[m - 1, j]
        for i in range(m - 2, -1, -1):
            tail_sum[i, j] = p[i, j] + tail_sum[i + 1, j]

    # Precompute: for each machine i, the minimum tail_sum[i+1][j]
    # over all jobs. Used for the lower bound "exit" component.
    # (The minimum time to get from machine i to the last machine.)
    min_tail_after = np.zeros(m, dtype=int)
    for i in range(m - 1):
        min_tail_after[i] = np.min(tail_sum[i + 1, :])
    min_tail_after[m - 1] = 0

    # Search statistics
    nodes_explored = 0
    nodes_pruned = 0
    start_time = time.time()
    timed_out = False

    def lower_bound(completion: np.ndarray,
                    unscheduled: list[int]) -> int:
        """
        Compute machine-based lower bound for the partial schedule.

        Args:
            completion: Array of shape (m,) — completion time on each
                       machine after the last scheduled job.
            unscheduled: List of unscheduled job indices.

        Returns:
            Lower bound on the makespan achievable from this partial state.
        """
        if not unscheduled:
            return int(completion[-1])

        lb = 0
        remaining_on_machine = np.zeros(m, dtype=int)
        min_exit = np.full(m, dtype=int, fill_value=np.iinfo(int).max)

        for j in unscheduled:
            for i in range(m):
                remaining_on_machine[i] += p[i, j]
            for i in range(m - 1):
                min_exit[i] = min(min_exit[i], tail_sum[i + 1, j])
        min_exit[m - 1] = 0

        for i in range(m):
            # Time to finish all unscheduled jobs on machine i,
            # starting from when machine i becomes available
            bound_i = int(completion[i]) + remaining_on_machine[i] + min_exit[i]
            lb = max(lb, bound_i)

        return lb

    def search(partial_perm: list[int],
               completion: np.ndarray,
               unscheduled: list[int]) -> None:
        nonlocal best_makespan, best_perm, nodes_explored, nodes_pruned
        nonlocal timed_out

        # Time check (every 1000 nodes)
        if nodes_explored % 1000 == 0:
            if time.time() - start_time >= time_limit:
                timed_out = True
                return

        if not unscheduled:
            # Complete solution
            ms = int(completion[-1])
            if ms < best_makespan:
                best_makespan = ms
                best_perm = list(partial_perm)
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  New best: {ms} at node {nodes_explored} "
                          f"({elapsed:.1f}s)")
            return

        # Try each unscheduled job as the next in the sequence
        for job in sorted(unscheduled):
            if timed_out:
                return

            nodes_explored += 1

            # Compute completion times if we add this job
            new_completion = np.copy(completion)
            new_completion[0] += p[0, job]
            for i in range(1, m):
                new_completion[i] = (max(new_completion[i - 1],
                                         new_completion[i])
                                     + p[i, job])

            # Compute lower bound
            remaining = [j for j in unscheduled if j != job]
            lb = lower_bound(new_completion, remaining)

            if lb >= best_makespan:
                nodes_pruned += 1
                continue  # Prune

            # Recurse
            partial_perm.append(job)
            search(partial_perm, new_completion, remaining)
            partial_perm.pop()

    if verbose:
        print(f"Branch and Bound for {n}×{m} instance")
        print(f"NEH warm start: {best_makespan}")
        print(f"Time limit: {time_limit}s")

    # Start search from empty partial schedule
    initial_completion = np.zeros(m, dtype=int)
    search([], initial_completion, list(range(n)))

    if verbose:
        elapsed = time.time() - start_time
        status = "OPTIMAL" if not timed_out else "TIME LIMIT"
        print(f"\n[{status}] Makespan: {best_makespan}")
        print(f"Nodes explored: {nodes_explored:,}")
        print(f"Nodes pruned:   {nodes_pruned:,}")
        print(f"Prune ratio:    {nodes_pruned/(nodes_explored+1)*100:.1f}%")
        print(f"Time:           {elapsed:.2f}s")

    return FlowShopSolution(permutation=best_perm, makespan=best_makespan)


if __name__ == "__main__":
    # Test on small instances
    print("=" * 50)
    print("B&B on 8×3 random instance")
    print("=" * 50)
    instance = FlowShopInstance.random(n=8, m=3, seed=42)
    sol = branch_and_bound(instance, verbose=True)
    print(f"\nOptimal permutation: {sol.permutation}")

    # Verify against NEH
    neh_sol = neh(instance)
    print(f"\nNEH makespan: {neh_sol.makespan}")
    print(f"B&B makespan: {sol.makespan}")
    print(f"Gap: {neh_sol.makespan - sol.makespan}")

    # Slightly larger
    print("\n" + "=" * 50)
    print("B&B on 10×3 random instance")
    print("=" * 50)
    instance2 = FlowShopInstance.random(n=10, m=3, seed=42)
    sol2 = branch_and_bound(instance2, time_limit=10.0, verbose=True)

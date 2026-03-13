"""Kernighan-Lin style greedy swap heuristic for Graph Partitioning.

Algorithm: Start with a random balanced partition. Iteratively find the
pair of vertices (one from each partition) whose swap yields the largest
reduction in edge cut. Lock swapped vertices and repeat for one pass.
Keep the best partition seen. Repeat until no improvement.

For k > 2, apply pairwise KL refinement between each pair of partitions.

Complexity: O(n^2 * k^2) per pass.

References:
    Kernighan, B. W., & Lin, S. (1970). An efficient heuristic procedure
    for partitioning graphs. The Bell System Technical Journal, 49(2), 291-307.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "gp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
GraphPartitioningInstance = _inst.GraphPartitioningInstance
GraphPartitioningSolution = _inst.GraphPartitioningSolution


def _initial_partition(n: int, k: int, seed: int = 42) -> list[int]:
    """Create a balanced initial partition."""
    rng = np.random.default_rng(seed)
    partition = []
    for p in range(k):
        count = n // k + (1 if p < n % k else 0)
        partition.extend([p] * count)
    rng.shuffle(partition)
    return list(partition)


def _swap_gain(adj: np.ndarray, partition: list[int], u: int, v: int) -> float:
    """Compute gain of swapping vertices u and v between their partitions.

    Gain > 0 means the swap reduces the edge cut.
    """
    n = len(partition)
    pu, pv = partition[u], partition[v]
    gain = 0.0

    for w in range(n):
        if w == u or w == v:
            continue
        pw = partition[w]
        # Effect on u: moving u from pu to pv
        if adj[u, w] > 0:
            if pw == pu:
                gain -= adj[u, w]  # was internal, now external
            elif pw == pv:
                gain += adj[u, w]  # was external, now internal
        # Effect on v: moving v from pv to pu
        if adj[v, w] > 0:
            if pw == pv:
                gain -= adj[v, w]  # was internal, now external
            elif pw == pu:
                gain += adj[v, w]  # was external, now internal

    # Edge between u and v: does not change (both swap)
    return gain


def greedy_kl(instance: GraphPartitioningInstance,
              seed: int = 42, max_passes: int = 50) -> GraphPartitioningSolution:
    """Kernighan-Lin style heuristic for balanced k-way graph partitioning.

    Args:
        instance: A GraphPartitioningInstance.
        seed: Random seed for initial partition.
        max_passes: Maximum number of KL passes.

    Returns:
        A GraphPartitioningSolution.
    """
    partition = _initial_partition(instance.n, instance.k, seed)
    best_cut = instance.edge_cut(partition)
    best_partition = partition.copy()

    for _ in range(max_passes):
        improved = False

        # Try all pairs of partitions
        for pa in range(instance.k):
            for pb in range(pa + 1, instance.k):
                vertices_a = [i for i in range(instance.n) if partition[i] == pa]
                vertices_b = [i for i in range(instance.n) if partition[i] == pb]

                locked = set()
                pass_gains = []
                pass_swaps = []

                n_swaps = min(len(vertices_a), len(vertices_b))
                for _ in range(n_swaps):
                    best_gain = -float('inf')
                    best_u, best_v = -1, -1

                    for u in vertices_a:
                        if u in locked:
                            continue
                        for v in vertices_b:
                            if v in locked:
                                continue
                            gain = _swap_gain(instance.adjacency, partition, u, v)
                            if gain > best_gain:
                                best_gain = gain
                                best_u, best_v = u, v

                    if best_u == -1:
                        break

                    # Perform swap
                    partition[best_u], partition[best_v] = partition[best_v], partition[best_u]
                    locked.add(best_u)
                    locked.add(best_v)
                    pass_gains.append(best_gain)
                    pass_swaps.append((best_u, best_v))

                # Find best prefix of swaps
                if pass_gains:
                    cumulative = np.cumsum(pass_gains)
                    best_prefix = int(np.argmax(cumulative))

                    if cumulative[best_prefix] > 0:
                        # Undo swaps after best prefix
                        for idx in range(len(pass_swaps) - 1, best_prefix, -1):
                            u, v = pass_swaps[idx]
                            partition[u], partition[v] = partition[v], partition[u]
                        improved = True
                    else:
                        # Undo all swaps
                        for u, v in reversed(pass_swaps):
                            partition[u], partition[v] = partition[v], partition[u]

        current_cut = instance.edge_cut(partition)
        if current_cut < best_cut:
            best_cut = current_cut
            best_partition = partition.copy()

        if not improved:
            break

    return GraphPartitioningSolution(
        partition=best_partition,
        edge_cut=best_cut,
    )


if __name__ == "__main__":
    inst = GraphPartitioningInstance.random(n=20, k=3, density=0.4)
    sol = greedy_kl(inst)
    print(f"Instance: {inst.n} vertices, k={inst.k}")
    print(f"Solution: {sol}")
    print(f"Balanced: {inst.is_balanced(sol.partition)}")

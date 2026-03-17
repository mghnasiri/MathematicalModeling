"""
Maximum Weight Matching Problem — Instance and Solution.

Given a weighted bipartite graph with n workers and m tasks, find a matching
(subset of edges with no shared endpoints) that maximizes total weight.
Unlike the standard assignment, not all workers/tasks need to be matched,
and n need not equal m.

Complexity: Polynomial — O(n^3) via Hungarian method on augmented matrix.

References:
    Kuhn, H.W. (1955). The Hungarian method for the assignment problem.
    Naval Research Logistics Quarterly, 2(1-2), 83-97.
    https://doi.org/10.1002/nav.3800020109
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MaxMatchingInstance:
    """Maximum Weight Matching instance.

    Attributes:
        n_workers: Number of workers.
        n_tasks: Number of tasks.
        weights: Weight matrix, shape (n_workers, n_tasks). Zero means no edge.
        name: Optional instance name.
    """

    n_workers: int
    n_tasks: int
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)

    @classmethod
    def random(
        cls,
        n_workers: int = 6,
        n_tasks: int = 8,
        weight_range: tuple[int, int] = (0, 20),
        density: float = 0.7,
        seed: int | None = None,
    ) -> MaxMatchingInstance:
        rng = np.random.default_rng(seed)
        weights = rng.integers(weight_range[0], weight_range[1] + 1,
                               size=(n_workers, n_tasks)).astype(float)
        # Sparsify
        mask = rng.random((n_workers, n_tasks)) > density
        weights[mask] = 0.0
        return cls(n_workers=n_workers, n_tasks=n_tasks, weights=weights,
                   name=f"random_{n_workers}x{n_tasks}")

    def matching_weight(self, matching: list[tuple[int, int]]) -> float:
        """Total weight of a matching."""
        return float(sum(self.weights[w][t] for w, t in matching))


@dataclass
class MaxMatchingSolution:
    """Maximum Weight Matching solution.

    Attributes:
        matching: List of (worker, task) pairs.
        total_weight: Total weight of the matching.
    """

    matching: list[tuple[int, int]]
    total_weight: float

    def __repr__(self) -> str:
        return f"MaxMatchingSolution(pairs={len(self.matching)}, weight={self.total_weight:.1f})"


def validate_solution(
    instance: MaxMatchingInstance, solution: MaxMatchingSolution
) -> tuple[bool, list[str]]:
    errors = []
    workers_used = set()
    tasks_used = set()
    for w, t in solution.matching:
        if w < 0 or w >= instance.n_workers:
            errors.append(f"Worker {w} out of range")
        if t < 0 or t >= instance.n_tasks:
            errors.append(f"Task {t} out of range")
        if w in workers_used:
            errors.append(f"Worker {w} matched twice")
        if t in tasks_used:
            errors.append(f"Task {t} matched twice")
        workers_used.add(w)
        tasks_used.add(t)
    actual = instance.matching_weight(solution.matching)
    if abs(actual - solution.total_weight) > 1e-4:
        errors.append(f"Weight mismatch: {solution.total_weight:.2f} != {actual:.2f}")
    return len(errors) == 0, errors


def small_matching_4x5() -> MaxMatchingInstance:
    return MaxMatchingInstance(
        n_workers=4,
        n_tasks=5,
        weights=np.array([
            [8, 0, 5, 3, 7],
            [0, 6, 4, 0, 9],
            [3, 7, 0, 8, 0],
            [5, 0, 6, 0, 4],
        ], dtype=float),
        name="small_4x5",
    )


if __name__ == "__main__":
    inst = small_matching_4x5()
    print(f"{inst.name}: {inst.n_workers} workers, {inst.n_tasks} tasks")

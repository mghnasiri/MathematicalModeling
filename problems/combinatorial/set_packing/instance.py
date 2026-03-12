"""
Maximum Weight Set Packing Problem — Instance and Solution definitions.

Given a universe U of elements and a collection of m subsets S_i with
weights w_i, find a maximum-weight subcollection of pairwise disjoint
sets (no two selected sets share an element).

Complexity: NP-hard. Greedy achieves 1/k-approximation where k is
the maximum set size.

References:
    Hurkens, C.A.J. & Schrijver, A. (1989). On the size of systems of
    sets every t of which have an SDR, with an application to the worst-case
    ratio of heuristics for packing problems. SIAM Journal on Discrete
    Mathematics, 2(1), 68-72.
    https://doi.org/10.1137/0402008
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SetPackingInstance:
    """Maximum weight set packing instance.

    Attributes:
        n_elements: Number of elements in the universe.
        sets: List of sets, each a frozenset of element indices.
        weights: Weight of each set, shape (m,).
        name: Optional instance name.
    """

    n_elements: int
    sets: list[frozenset[int]]
    weights: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (len(self.sets),):
            raise ValueError(
                f"weights shape {self.weights.shape} != ({len(self.sets)},)"
            )
        for i, s in enumerate(self.sets):
            for e in s:
                if e < 0 or e >= self.n_elements:
                    raise ValueError(
                        f"Set {i} contains invalid element {e}"
                    )

    @property
    def m(self) -> int:
        """Number of sets."""
        return len(self.sets)

    def are_disjoint(self, indices: list[int]) -> bool:
        """Check if selected sets are pairwise disjoint."""
        used = set()
        for i in indices:
            if used & self.sets[i]:
                return False
            used |= self.sets[i]
        return True

    def total_weight(self, indices: list[int]) -> float:
        """Compute total weight of selected sets."""
        return float(sum(self.weights[i] for i in indices))

    @classmethod
    def random(
        cls,
        n_elements: int = 10,
        m: int = 8,
        max_set_size: int = 4,
        seed: int | None = None,
    ) -> SetPackingInstance:
        """Generate a random set packing instance.

        Args:
            n_elements: Universe size.
            m: Number of sets.
            max_set_size: Maximum elements per set.
            seed: Random seed.

        Returns:
            A random SetPackingInstance.
        """
        rng = np.random.default_rng(seed)
        sets = []
        for _ in range(m):
            size = rng.integers(1, max_set_size + 1)
            elements = frozenset(rng.choice(n_elements, size=size, replace=False).tolist())
            sets.append(elements)
        weights = np.round(rng.uniform(1.0, 20.0, size=m), 1)
        return cls(n_elements=n_elements, sets=sets, weights=weights,
                   name=f"random_sp_{m}")


@dataclass
class SetPackingSolution:
    """Solution to a set packing instance.

    Attributes:
        selected: List of selected set indices.
        total_weight: Total weight of selected sets.
    """

    selected: list[int]
    total_weight: float

    def __repr__(self) -> str:
        return (
            f"SetPackingSolution(weight={self.total_weight:.1f}, "
            f"selected={self.selected})"
        )


# -- Benchmark instances ------------------------------------------------------


def small_sp_3() -> SetPackingInstance:
    """3 sets on 5 elements. Sets: {0,1}, {2,3}, {1,4}. Optimal = {0,1} (w=15)."""
    return SetPackingInstance(
        n_elements=5,
        sets=[frozenset({0, 1}), frozenset({2, 3}), frozenset({1, 4})],
        weights=np.array([10.0, 5.0, 8.0]),
        name="small_3",
    )


def disjoint_4() -> SetPackingInstance:
    """4 pairwise disjoint sets. All can be selected."""
    return SetPackingInstance(
        n_elements=8,
        sets=[frozenset({0, 1}), frozenset({2, 3}),
              frozenset({4, 5}), frozenset({6, 7})],
        weights=np.array([5.0, 8.0, 3.0, 7.0]),
        name="disjoint_4",
    )


def conflict_5() -> SetPackingInstance:
    """5 sets with heavy conflicts."""
    return SetPackingInstance(
        n_elements=4,
        sets=[
            frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3}),
            frozenset({0, 3}), frozenset({0, 1, 2, 3}),
        ],
        weights=np.array([10.0, 8.0, 6.0, 4.0, 20.0]),
        name="conflict_5",
    )


if __name__ == "__main__":
    inst = small_sp_3()
    print(f"{inst.name}: {inst.m} sets, {inst.n_elements} elements")

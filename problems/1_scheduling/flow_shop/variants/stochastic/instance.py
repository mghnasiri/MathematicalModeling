"""
Stochastic Flow Shop (Fm | prmu, stoch | E[Cmax]) — Instance and Solution.

Extends the permutation flow shop by modeling processing times as random
variables (typically normal or uniform distributions). The objective is
to minimize the expected makespan.

Complexity: NP-hard (generalizes deterministic PFSP).

References:
    Gourgand, M., Grangeon, N. & Norre, S. (2000). A review of the static
    stochastic flow-shop scheduling problem. Journal of Decision Systems,
    9(2), 1-31. https://doi.org/10.1080/12460125.2000.9736710

    Framinan, J.M. & Perez-Gonzalez, P. (2015). On heuristic solutions
    for the stochastic flowshop scheduling problem. European Journal of
    Operational Research, 246(2), 413-420.
    https://doi.org/10.1016/j.ejor.2015.05.006
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class StochasticFlowShopInstance:
    """Stochastic Flow Shop instance.

    Processing times are defined by mean and standard deviation matrices.
    Samples are drawn from truncated normal distributions (non-negative).

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        mean_times: Mean processing times, shape (n, m).
        std_times: Standard deviation of processing times, shape (n, m).
        name: Optional instance name.
    """

    n: int
    m: int
    mean_times: np.ndarray
    std_times: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.mean_times = np.asarray(self.mean_times, dtype=float)
        self.std_times = np.asarray(self.std_times, dtype=float)

    def sample_times(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample processing times from truncated normal distributions.

        Args:
            rng: Random generator for sampling.

        Returns:
            Sampled processing times, shape (n, m), all non-negative.
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = rng.normal(self.mean_times, self.std_times)
        return np.maximum(samples, 0.1)  # Ensure positive

    def makespan(self, perm: list[int], processing_times: np.ndarray) -> float:
        """Compute makespan for a permutation with given processing times.

        Args:
            perm: Job permutation.
            processing_times: Realized processing times, shape (n, m).

        Returns:
            Makespan value.
        """
        nj = len(perm)
        C = np.zeros((nj, self.m))
        for i, job in enumerate(perm):
            for k in range(self.m):
                prev_job = C[i - 1][k] if i > 0 else 0
                prev_machine = C[i][k - 1] if k > 0 else 0
                C[i][k] = max(prev_job, prev_machine) + processing_times[job][k]
        return float(C[-1][-1])

    def expected_makespan(
        self, perm: list[int], num_samples: int = 100,
        seed: int | None = None
    ) -> float:
        """Estimate expected makespan by Monte Carlo sampling.

        Args:
            perm: Job permutation.
            num_samples: Number of Monte Carlo samples.
            seed: Random seed for reproducibility.

        Returns:
            Estimated expected makespan.
        """
        rng = np.random.default_rng(seed)
        total = 0.0
        for _ in range(num_samples):
            times = self.sample_times(rng)
            total += self.makespan(perm, times)
        return total / num_samples

    def deterministic_makespan(self, perm: list[int]) -> float:
        """Makespan using mean processing times (deterministic proxy)."""
        return self.makespan(perm, self.mean_times)

    @classmethod
    def random(
        cls,
        n: int = 6,
        m: int = 3,
        mean_range: tuple[int, int] = (10, 50),
        cv: float = 0.2,
        seed: int | None = None,
    ) -> StochasticFlowShopInstance:
        """Generate random stochastic flow shop instance.

        Args:
            n: Number of jobs.
            m: Number of machines.
            mean_range: Range for mean processing times.
            cv: Coefficient of variation (std/mean).
            seed: Random seed.

        Returns:
            StochasticFlowShopInstance.
        """
        rng = np.random.default_rng(seed)
        mean_times = rng.integers(mean_range[0], mean_range[1] + 1,
                                   size=(n, m)).astype(float)
        std_times = mean_times * cv
        return cls(n=n, m=m, mean_times=mean_times, std_times=std_times,
                   name=f"stoch_{n}x{m}")


@dataclass
class StochasticFlowShopSolution:
    """Stochastic flow shop solution.

    Attributes:
        permutation: Job permutation.
        expected_makespan: Estimated expected makespan.
    """

    permutation: list[int]
    expected_makespan: float

    def __repr__(self) -> str:
        return f"StochFSSolution(E[Cmax]={self.expected_makespan:.1f})"


def validate_solution(
    instance: StochasticFlowShopInstance, solution: StochasticFlowShopSolution
) -> tuple[bool, list[str]]:
    errors = []

    if sorted(solution.permutation) != list(range(instance.n)):
        errors.append("Invalid permutation")

    return len(errors) == 0, errors


def small_stoch_fs_4x3() -> StochasticFlowShopInstance:
    """Small stochastic flow shop instance with 4 jobs and 3 machines."""
    mean_times = np.array([
        [20, 15, 25],
        [30, 10, 20],
        [15, 25, 30],
        [25, 20, 15],
    ], dtype=float)
    std_times = mean_times * 0.2
    return StochasticFlowShopInstance(
        n=4, m=3, mean_times=mean_times, std_times=std_times,
        name="small_stoch_4x3",
    )


if __name__ == "__main__":
    inst = small_stoch_fs_4x3()
    print(f"{inst.name}: n={inst.n}, m={inst.m}")
    perm = list(range(inst.n))
    print(f"Deterministic makespan: {inst.deterministic_makespan(perm):.1f}")
    print(f"Expected makespan (100 samples): {inst.expected_makespan(perm, seed=42):.1f}")

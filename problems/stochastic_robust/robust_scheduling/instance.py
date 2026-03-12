"""
Robust Single Machine Scheduling (Min-Max Regret)

Given n jobs with uncertain processing times (discrete scenarios),
find a schedule minimizing the worst-case regret of the total
weighted completion time or makespan.

Notation: 1 | uncertain p_j | min max-regret gamma

Two robustness criteria:
    - Min-Max Cost: min_{pi} max_{s} Cmax(pi, s)
    - Min-Max Regret: min_{pi} max_{s} [gamma(pi,s) - gamma(pi*_s, s)]

Complexity:
    - Min-max Cmax on a single machine: trivially optimal (any order,
      same Cmax) when only processing times are uncertain.
    - Min-max regret ΣCj: NP-hard for general interval data.
    - Min-max regret ΣwjCj: NP-hard.

References:
    - Kouvelis, P. & Yu, G. (1997). Robust Discrete Optimization.
      Springer. https://doi.org/10.1007/978-1-4757-2620-6
    - Lebedev, V. & Averbakh, I. (2006). Complexity of minimizing the
      total flow time with interval data. DAM, 154(15), 2167-2177.
      https://doi.org/10.1016/j.dam.2005.04.015
    - Kasperski, A. & Zielinski, P. (2008). A 2-approximation algorithm
      for interval data minmax regret sequencing problems. Oper. Res.
      Lett., 36(5), 561-564. https://doi.org/10.1016/j.orl.2008.07.004
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class RobustSchedulingInstance:
    """Robust single machine scheduling with scenario-dependent processing times.

    Args:
        n: Number of jobs.
        processing_scenarios: Processing times per scenario (S, n).
        weights: Job weights for weighted objectives (n,). All-ones if None.
        due_dates: Job due dates (n,). None if not relevant.
        probabilities: Scenario probabilities (S,). Uniform if None.
    """
    n: int
    processing_scenarios: np.ndarray
    weights: np.ndarray | None = None
    due_dates: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.processing_scenarios = np.asarray(self.processing_scenarios, dtype=float)
        if self.weights is None:
            self.weights = np.ones(self.n)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
        if self.due_dates is not None:
            self.due_dates = np.asarray(self.due_dates, dtype=float)
        if self.probabilities is None:
            S = self.processing_scenarios.shape[0]
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_scenarios(self) -> int:
        return self.processing_scenarios.shape[0]

    @property
    def mean_processing(self) -> np.ndarray:
        """Expected processing time of each job."""
        return np.dot(self.probabilities, self.processing_scenarios)

    def total_weighted_completion(self, permutation: list[int],
                                  scenario: int) -> float:
        """Compute ΣwjCj for a permutation under a scenario.

        Args:
            permutation: Job processing order (0-indexed).
            scenario: Scenario index.

        Returns:
            Total weighted completion time.
        """
        p = self.processing_scenarios[scenario]
        w = self.weights
        C = 0.0
        twc = 0.0
        for j in permutation:
            C += p[j]
            twc += w[j] * C
        return twc

    def makespan(self, permutation: list[int], scenario: int) -> float:
        """Compute Cmax for a permutation under a scenario.

        On a single machine, Cmax = sum of processing times (order-independent).
        """
        return float(self.processing_scenarios[scenario].sum())

    def total_completion(self, permutation: list[int], scenario: int) -> float:
        """Compute ΣCj for a permutation under a scenario."""
        p = self.processing_scenarios[scenario]
        C = 0.0
        tc = 0.0
        for j in permutation:
            C += p[j]
            tc += C
        return tc

    def max_regret_twc(self, permutation: list[int],
                        optimal_values: list[float] | None = None) -> float:
        """Compute max regret of ΣwjCj across scenarios.

        Args:
            permutation: Job order.
            optimal_values: Pre-computed optimal ΣwjCj per scenario.
                If None, computed internally (expensive).

        Returns:
            Maximum regret across scenarios.
        """
        S = self.n_scenarios
        if optimal_values is None:
            optimal_values = self._compute_optimal_twc()

        max_reg = 0.0
        for s in range(S):
            val = self.total_weighted_completion(permutation, s)
            reg = val - optimal_values[s]
            max_reg = max(max_reg, reg)
        return max_reg

    def _compute_optimal_twc(self) -> list[float]:
        """Compute optimal ΣwjCj per scenario via WSPT rule."""
        optimal = []
        for s in range(self.n_scenarios):
            p = self.processing_scenarios[s]
            ratios = p / np.maximum(self.weights, 1e-9)
            perm = list(np.argsort(ratios))
            optimal.append(self.total_weighted_completion(perm, s))
        return optimal

    @classmethod
    def random(cls, n: int = 8, n_scenarios: int = 10,
               seed: int = 42) -> RobustSchedulingInstance:
        """Generate a random robust scheduling instance."""
        rng = np.random.default_rng(seed)
        base_p = rng.uniform(3, 20, n)
        processing_scenarios = np.array([
            np.maximum(1, base_p + rng.normal(0, 4, n))
            for _ in range(n_scenarios)
        ])
        weights = rng.uniform(1, 10, n)
        return cls(n=n, processing_scenarios=processing_scenarios, weights=weights)


@dataclass
class RobustSchedulingSolution:
    """Solution to the robust scheduling problem.

    Args:
        permutation: Job processing order (0-indexed).
        max_regret: Worst-case regret across scenarios.
        expected_twc: Expected ΣwjCj across scenarios.
        scenario_values: ΣwjCj under each scenario.
    """
    permutation: list[int]
    max_regret: float
    expected_twc: float
    scenario_values: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"RobustSchedulingSolution(perm={self.permutation}, "
                f"max_regret={self.max_regret:.2f}, "
                f"E[ΣwjCj]={self.expected_twc:.2f})")

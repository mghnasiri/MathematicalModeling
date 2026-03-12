"""
Stochastic Knapsack Problem

Items have deterministic values but random weights drawn from known
distributions (discrete scenarios). The goal is to select items
maximizing expected value while satisfying the capacity constraint
with high probability or in expectation.

Two main variants:
    - Expected capacity: E[sum w_i x_i] <= W
    - Chance-constrained: P(sum w_i x_i <= W) >= 1 - alpha

Complexity: NP-hard (generalizes deterministic knapsack).

References:
    - Kleinberg, J., Rabani, Y. & Tardos, E. (1997). Allocating bandwidth
      for bursty connections. STOC, 664-673.
      https://doi.org/10.1145/258533.258661
    - Dean, B.C., Goemans, M.X. & Vondrák, J. (2008). Approximating the
      stochastic knapsack problem. Math. Oper. Res., 33(1), 1-14.
      https://doi.org/10.1287/moor.1070.0285
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class StochasticKnapsackInstance:
    """Stochastic knapsack with scenario-dependent weights.

    Args:
        n: Number of items.
        values: Deterministic value of each item (n,).
        weight_scenarios: Weight of each item under each scenario (S, n).
        capacity: Knapsack capacity W.
        probabilities: Scenario probabilities (S,). Uniform if None.
    """
    n: int
    values: np.ndarray
    weight_scenarios: np.ndarray
    capacity: float
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.weight_scenarios = np.asarray(self.weight_scenarios, dtype=float)
        if self.probabilities is None:
            S = self.weight_scenarios.shape[0]
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_scenarios(self) -> int:
        return self.weight_scenarios.shape[0]

    @property
    def mean_weights(self) -> np.ndarray:
        """Expected weight of each item."""
        return np.dot(self.probabilities, self.weight_scenarios)

    def solution_value(self, selection: np.ndarray) -> float:
        """Total value of selected items."""
        return float(np.dot(self.values, selection))

    def feasibility_probability(self, selection: np.ndarray) -> float:
        """Probability that the selection fits in the knapsack."""
        sel = np.asarray(selection, dtype=float)
        weights_per_scenario = self.weight_scenarios @ sel
        feasible = weights_per_scenario <= self.capacity + 1e-9
        return float(np.dot(feasible, self.probabilities))

    def expected_weight(self, selection: np.ndarray) -> float:
        """Expected total weight of selected items."""
        sel = np.asarray(selection, dtype=float)
        return float(np.dot(self.mean_weights, sel))

    @classmethod
    def random(cls, n: int = 10, n_scenarios: int = 20,
               seed: int = 42) -> StochasticKnapsackInstance:
        """Generate a random stochastic knapsack instance."""
        rng = np.random.default_rng(seed)
        values = rng.uniform(5, 50, n)
        base_weights = rng.uniform(3, 20, n)
        weight_scenarios = np.array([
            np.maximum(1, base_weights + rng.normal(0, 3, n))
            for _ in range(n_scenarios)
        ])
        capacity = float(base_weights.sum() * 0.5)
        return cls(n=n, values=values, weight_scenarios=weight_scenarios,
                   capacity=capacity)


@dataclass
class StochasticKnapsackSolution:
    """Solution to the stochastic knapsack problem.

    Args:
        selection: Binary vector of selected items.
        total_value: Sum of values of selected items.
        feasibility_prob: P(total weight <= capacity).
        expected_weight: E[total weight].
    """
    selection: np.ndarray
    total_value: float
    feasibility_prob: float
    expected_weight: float

    def __repr__(self) -> str:
        n_sel = int(self.selection.sum())
        return (f"StochasticKnapsackSolution(items={n_sel}, "
                f"value={self.total_value:.1f}, "
                f"P(feasible)={self.feasibility_prob:.3f}, "
                f"E[weight]={self.expected_weight:.1f})")

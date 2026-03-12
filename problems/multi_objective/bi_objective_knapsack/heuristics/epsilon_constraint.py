"""Epsilon-Constraint method for Bi-Objective 0-1 Knapsack.

Solves a series of single-objective knapsack problems, each maximizing
objective 1 subject to objective 2 >= epsilon, for varying epsilon values.
Collects all non-dominated solutions to approximate the Pareto front.

Also includes a greedy Pareto heuristic using weighted-sum scalarization.

Complexity: O(G * n * W) where G = number of epsilon grid points.

References:
    Ehrgott, M. (2005). Multicriteria Optimization (2nd ed.). Springer.
    Haimes, Y. Y. (1971). On a bicriterion formulation of the problems of
    integrated system identification and system optimization. IEEE Trans.
    Systems, Man, and Cybernetics, 1(3), 296-297.
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
    "bok_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
BiObjectiveKnapsackInstance = _inst.BiObjectiveKnapsackInstance
BiObjectiveKnapsackSolution = _inst.BiObjectiveKnapsackSolution


def _solve_knapsack_dp(weights: np.ndarray, values: np.ndarray,
                        capacity: int) -> np.ndarray:
    """Solve single-objective 0-1 knapsack via DP.

    Args:
        weights: Item weights.
        values: Item values.
        capacity: Capacity.

    Returns:
        Binary selection vector.
    """
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1), dtype=float)

    for i in range(1, n + 1):
        w = int(weights[i - 1])
        v = float(values[i - 1])
        for c in range(capacity + 1):
            dp[i, c] = dp[i - 1, c]
            if w <= c and dp[i - 1, c - w] + v > dp[i, c]:
                dp[i, c] = dp[i - 1, c - w] + v

    # Backtrack
    x = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            x[i - 1] = 1
            c -= int(weights[i - 1])

    return x


def _is_dominated(point: tuple[float, float],
                   front: list[tuple[float, float]]) -> bool:
    """Check if point is dominated by any point in front."""
    for fp in front:
        if fp[0] >= point[0] and fp[1] >= point[1] and fp != point:
            return True
    return False


def _filter_pareto(points: list[tuple[float, float]],
                   solutions: list[np.ndarray]
                   ) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    """Filter to non-dominated solutions."""
    filtered_pts = []
    filtered_sols = []
    for i, pt in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j and other[0] >= pt[0] and other[1] >= pt[1] and other != pt:
                dominated = True
                break
        if not dominated:
            # Check for duplicates
            is_dup = False
            for fp in filtered_pts:
                if abs(fp[0] - pt[0]) < 1e-9 and abs(fp[1] - pt[1]) < 1e-9:
                    is_dup = True
                    break
            if not is_dup:
                filtered_pts.append(pt)
                filtered_sols.append(solutions[i])
    return filtered_pts, filtered_sols


def epsilon_constraint(instance: BiObjectiveKnapsackInstance,
                       n_points: int = 20) -> BiObjectiveKnapsackSolution:
    """Epsilon-constraint method for bi-objective knapsack.

    Varies epsilon on objective 2, solving max v1 s.t. v2 >= epsilon
    for each epsilon value.

    Args:
        instance: A BiObjectiveKnapsackInstance.
        n_points: Number of epsilon grid points.

    Returns:
        A BiObjectiveKnapsackSolution with approximate Pareto front.
    """
    # Determine range for objective 2
    v2_max = float(instance.values2.sum())
    epsilons = np.linspace(0, v2_max, n_points)

    all_points = []
    all_solutions = []

    for eps in epsilons:
        # Solve: max v1 s.t. weight <= W AND v2 >= eps
        # We handle the v2 constraint by penalizing infeasible items
        # Simple approach: DP with modified capacity
        # Actually, use a combined approach: solve max v1 with items
        # that can contribute to v2 >= eps

        # Brute force for small n, DP otherwise
        if instance.n <= 20:
            best_x = None
            best_v1 = -1.0
            for mask in range(1 << instance.n):
                x = np.array([(mask >> i) & 1 for i in range(instance.n)])
                if not instance.is_feasible(x):
                    continue
                v1, v2 = instance.evaluate(x)
                if v2 >= eps - 1e-9 and v1 > best_v1:
                    best_v1 = v1
                    best_x = x.copy()
            if best_x is not None:
                v1, v2 = instance.evaluate(best_x)
                all_points.append((v1, v2))
                all_solutions.append(best_x)
        else:
            # For larger instances, use weighted-sum scalarization
            alpha = eps / (v2_max + 1e-9)
            combined = (1 - alpha) * instance.values1 + alpha * instance.values2
            x = _solve_knapsack_dp(instance.weights, combined,
                                    instance.capacity)
            v1, v2 = instance.evaluate(x)
            all_points.append((v1, v2))
            all_solutions.append(x)

    if not all_points:
        return BiObjectiveKnapsackSolution(
            pareto_front=[], pareto_solutions=[], n_solutions=0
        )

    front, sols = _filter_pareto(all_points, all_solutions)
    # Sort by objective 1
    paired = sorted(zip(front, sols), key=lambda x: x[0])
    front = [p[0] for p in paired]
    sols = [p[1] for p in paired]

    return BiObjectiveKnapsackSolution(
        pareto_front=front, pareto_solutions=sols, n_solutions=len(front)
    )


def weighted_sum_pareto(instance: BiObjectiveKnapsackInstance,
                        n_weights: int = 11) -> BiObjectiveKnapsackSolution:
    """Weighted-sum scalarization for bi-objective knapsack.

    Solves max alpha * v1 + (1 - alpha) * v2 for varying alpha in [0, 1].

    Args:
        instance: A BiObjectiveKnapsackInstance.
        n_weights: Number of weight values to try.

    Returns:
        A BiObjectiveKnapsackSolution with supported Pareto front.
    """
    all_points = []
    all_solutions = []

    for alpha in np.linspace(0, 1, n_weights):
        combined = alpha * instance.values1 + (1 - alpha) * instance.values2
        x = _solve_knapsack_dp(instance.weights, combined.astype(float),
                                instance.capacity)
        v1, v2 = instance.evaluate(x)
        all_points.append((v1, v2))
        all_solutions.append(x)

    front, sols = _filter_pareto(all_points, all_solutions)
    paired = sorted(zip(front, sols), key=lambda x: x[0])
    front = [p[0] for p in paired]
    sols = [p[1] for p in paired]

    return BiObjectiveKnapsackSolution(
        pareto_front=front, pareto_solutions=sols, n_solutions=len(front)
    )


if __name__ == "__main__":
    inst = BiObjectiveKnapsackInstance.random(n=8, seed=42)
    print(f"Instance: {inst.n} items, capacity={inst.capacity}")

    sol_ec = epsilon_constraint(inst, n_points=15)
    print(f"\nEpsilon-constraint: {sol_ec.n_solutions} Pareto points")
    for pt in sol_ec.pareto_front:
        print(f"  v1={pt[0]:.0f}, v2={pt[1]:.0f}")

    sol_ws = weighted_sum_pareto(inst, n_weights=11)
    print(f"\nWeighted-sum: {sol_ws.n_solutions} Pareto points")
    for pt in sol_ws.pareto_front:
        print(f"  v1={pt[0]:.0f}, v2={pt[1]:.0f}")

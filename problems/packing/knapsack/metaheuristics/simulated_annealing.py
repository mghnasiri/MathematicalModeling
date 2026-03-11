"""
Simulated Annealing for 0-1 Knapsack Problem.

Problem: KP01 (0-1 Knapsack)

Uses a bit-flip neighborhood: at each iteration, toggle one item
(add if not selected, remove if selected). Infeasible solutions
(overweight) are penalized to guide the search back to feasibility.

Warm-started with greedy value-density heuristic.

Complexity: O(iterations * n) per run.

References:
    Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization
    by simulated annealing. Science, 220(4598), 671-680.
    https://doi.org/10.1126/science.220.4598.671

    Drexl, A. (1988). A simulated annealing approach to the multiconstraint
    zero-one knapsack problem. Computing, 40(1), 1-8.
    https://doi.org/10.1007/BF02242185
"""

from __future__ import annotations

import os
import math
import time
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_module(
    "knapsack_instance_sa", os.path.join(_parent_dir, "instance.py")
)
KnapsackInstance = _inst.KnapsackInstance
KnapsackSolution = _inst.KnapsackSolution


def simulated_annealing(
    instance: KnapsackInstance,
    max_iterations: int = 10000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.999,
    penalty_factor: float = 2.0,
    time_limit: float | None = None,
    seed: int | None = None,
) -> KnapsackSolution:
    """Solve 0-1 Knapsack using Simulated Annealing.

    Args:
        instance: Knapsack instance.
        max_iterations: Maximum iterations.
        initial_temp: Initial temperature. If None, auto-calibrated.
        cooling_rate: Geometric cooling factor.
        penalty_factor: Multiplier for overweight penalty.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best feasible KnapsackSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()
    weights = instance.weights
    values = instance.values
    capacity = instance.capacity

    # ── Greedy initialization (value density) ────────────────────────────
    density = values / np.maximum(weights, 1e-10)
    order = np.argsort(-density)

    selected = np.zeros(n, dtype=bool)
    total_weight = 0.0
    total_value = 0.0

    for i in order:
        if total_weight + weights[i] <= capacity:
            selected[i] = True
            total_weight += weights[i]
            total_value += values[i]

    best_selected = selected.copy()
    best_value = total_value
    best_weight = total_weight

    def _penalized_obj(val: float, wt: float) -> float:
        """Higher is better; penalize overweight."""
        if wt <= capacity + 1e-10:
            return val
        return val - penalty_factor * (wt - capacity) * max(density)

    current_obj = _penalized_obj(total_value, total_weight)

    if initial_temp is None:
        initial_temp = max(1.0, total_value * 0.1) if total_value > 0 else 10.0

    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Bit-flip: toggle a random item
        i = rng.integers(0, n)

        if selected[i]:
            # Remove item
            new_value = total_value - values[i]
            new_weight = total_weight - weights[i]
        else:
            # Add item
            new_value = total_value + values[i]
            new_weight = total_weight + weights[i]

        new_obj = _penalized_obj(new_value, new_weight)
        delta = new_obj - current_obj

        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-10)):
            selected[i] = not selected[i]
            total_value = new_value
            total_weight = new_weight
            current_obj = new_obj

            # Update best (feasible only)
            if total_weight <= capacity + 1e-10 and total_value > best_value:
                best_selected = selected.copy()
                best_value = total_value
                best_weight = total_weight

        temp *= cooling_rate

    items = [i for i in range(n) if best_selected[i]]
    return KnapsackSolution(
        items=items,
        value=float(best_value),
        weight=float(best_weight),
    )


if __name__ == "__main__":
    from instance import small_knapsack_4, medium_knapsack_8

    print("=== SA on small4 (optimal=35) ===")
    inst = small_knapsack_4()
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: value={sol.value}, weight={sol.weight}, items={sol.items}")

    print("\n=== SA on medium8 (optimal=300) ===")
    inst = medium_knapsack_8()
    sol = simulated_annealing(inst, seed=42)
    print(f"SA: value={sol.value}, weight={sol.weight}, items={sol.items}")

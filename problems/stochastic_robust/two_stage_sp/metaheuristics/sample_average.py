"""
Sample Average Approximation (SAA) for Two-Stage Stochastic Programs

Instead of solving the full deterministic equivalent with all scenarios,
SAA draws a random sample of N scenarios and solves the resulting
smaller problem. Multiple replications provide confidence intervals.

Complexity: Each replication solves an LP with n1 + N*n2 variables.

References:
    - Kleywegt, A.J., Shapiro, A. & Homem-de-Mello, T. (2002). The sample
      average approximation method for stochastic discrete optimization.
      SIAM J. Optim., 12(2), 479-502. https://doi.org/10.1137/S1052623499363220
    - Shapiro, A. (2003). Monte Carlo sampling methods. In Handbooks in OR&MS,
      Vol. 10, 353-425. https://doi.org/10.1016/S0927-0507(03)10006-0
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

import importlib.util

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("sp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
_de = _load_parent("sp_de", os.path.join(os.path.dirname(__file__), "..", "heuristics", "deterministic_equivalent.py"))
TwoStageSPInstance = _inst.TwoStageSPInstance
TwoStageSPSolution = _inst.TwoStageSPSolution
solve_deterministic_equivalent = _de.solve_deterministic_equivalent


@dataclass
class SAASolution:
    """Result of SAA with multiple replications.

    Args:
        best_solution: Best TwoStageSPSolution across replications.
        objective_values: Objective from each replication.
        mean_objective: Mean across replications.
        std_objective: Standard deviation across replications.
        n_replications: Number of replications solved.
    """
    best_solution: TwoStageSPSolution
    objective_values: list[float]
    mean_objective: float
    std_objective: float
    n_replications: int

    def __repr__(self) -> str:
        return (f"SAASolution(mean={self.mean_objective:.2f}, "
                f"std={self.std_objective:.2f}, "
                f"best={self.best_solution.total_cost:.2f}, "
                f"reps={self.n_replications})")


def sample_average_approximation(
    instance: TwoStageSPInstance,
    sample_size: int = 20,
    n_replications: int = 5,
    seed: int = 42,
) -> SAASolution:
    """Solve 2SSP via Sample Average Approximation.

    For each replication:
    1. Draw sample_size scenarios (with replacement) from the original set.
    2. Solve the deterministic equivalent on the sampled scenarios.

    Args:
        instance: TwoStageSPInstance.
        sample_size: Number of scenarios per SAA replication.
        n_replications: Number of independent replications.
        seed: Random seed.

    Returns:
        SAASolution with statistics across replications.
    """
    rng = np.random.default_rng(seed)
    S = instance.n_scenarios
    objectives = []
    best_sol = None
    best_obj = np.inf

    for rep in range(n_replications):
        # Sample scenarios with replacement
        indices = rng.choice(S, size=sample_size, replace=True)
        sampled_scenarios = [instance.scenarios[i] for i in indices]
        sampled_probs = np.full(sample_size, 1.0 / sample_size)

        saa_instance = TwoStageSPInstance(
            c=instance.c,
            A=instance.A,
            b=instance.b,
            scenarios=sampled_scenarios,
            probabilities=sampled_probs,
        )

        sol = solve_deterministic_equivalent(saa_instance)
        if sol is None:
            continue

        # Evaluate on all original scenarios for true objective
        true_cost = _evaluate_first_stage(instance, sol.x)
        objectives.append(true_cost)

        if true_cost < best_obj:
            best_obj = true_cost
            best_sol = TwoStageSPSolution(
                x=sol.x,
                first_stage_cost=sol.first_stage_cost,
                expected_recourse_cost=true_cost - sol.first_stage_cost,
                total_cost=true_cost,
            )

    if best_sol is None:
        # Fallback: return trivial solution
        x0 = np.zeros(instance.n1)
        best_sol = TwoStageSPSolution(
            x=x0,
            first_stage_cost=0.0,
            expected_recourse_cost=0.0,
            total_cost=0.0,
        )

    obj_arr = np.array(objectives) if objectives else np.array([0.0])

    return SAASolution(
        best_solution=best_sol,
        objective_values=objectives,
        mean_objective=float(obj_arr.mean()),
        std_objective=float(obj_arr.std()),
        n_replications=len(objectives),
    )


def _evaluate_first_stage(instance: TwoStageSPInstance, x: np.ndarray) -> float:
    """Evaluate a first-stage decision x on all original scenarios.

    For each scenario, solve the second-stage LP:
        min q^T y  s.t. W y <= h - T x, y >= 0

    Args:
        instance: Original TwoStageSPInstance.
        x: First-stage decision vector.

    Returns:
        Total expected cost (first-stage + expected recourse).
    """
    from scipy.optimize import linprog

    first_stage_cost = float(np.dot(instance.c, x))
    expected_recourse = 0.0

    for s, sc in enumerate(instance.scenarios):
        rhs = sc["h"] - sc["T"] @ x
        n2 = len(sc["q"])
        result = linprog(
            sc["q"],
            A_ub=sc["W"],
            b_ub=rhs,
            bounds=[(0, None)] * n2,
            method="highs",
        )
        if result.success:
            expected_recourse += instance.probabilities[s] * result.fun

    return first_stage_cost + expected_recourse


if __name__ == "__main__":
    inst = TwoStageSPInstance.capacity_planning(n_facilities=3, n_scenarios=10)
    saa_sol = sample_average_approximation(inst, sample_size=5, n_replications=5)
    print(f"SAA result: {saa_sol}")

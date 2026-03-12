"""Greedy heuristic for Maximum Satisfiability (MAX-SAT).

Algorithm: Assign variables one at a time, choosing the value (True/False)
that maximizes the weight of newly satisfied clauses.

Complexity: O(n_vars * n_clauses * max_clause_len)

References:
    Johnson, D. S. (1974). Approximation algorithms for combinatorial
    problems. Journal of Computer and System Sciences, 9(3), 256-278.
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
    "maxsat_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MaxSATInstance = _inst.MaxSATInstance
MaxSATSolution = _inst.MaxSATSolution


def greedy_maxsat(instance: MaxSATInstance) -> MaxSATSolution:
    """Greedy variable assignment for MAX-SAT.

    For each variable (in order), try both True and False and pick
    the assignment that satisfies the most additional clause weight.

    Args:
        instance: A MaxSATInstance.

    Returns:
        A MaxSATSolution.
    """
    assignment = [False] * instance.n_vars
    # Track which clauses are already satisfied
    satisfied = [False] * len(instance.clauses)

    for var_idx in range(instance.n_vars):
        var_num = var_idx + 1  # 1-indexed

        # Try True
        gain_true = 0.0
        for c_idx, clause in enumerate(instance.clauses):
            if satisfied[c_idx]:
                continue
            for lit in clause:
                if abs(lit) == var_num:
                    if (lit > 0 and True) or (lit < 0 and not True):
                        gain_true += instance.weights[c_idx]
                    break

        # Try False
        gain_false = 0.0
        for c_idx, clause in enumerate(instance.clauses):
            if satisfied[c_idx]:
                continue
            for lit in clause:
                if abs(lit) == var_num:
                    if (lit > 0 and False) or (lit < 0 and not False):
                        gain_false += instance.weights[c_idx]
                    break

        # Choose better value
        if gain_true >= gain_false:
            assignment[var_idx] = True
        else:
            assignment[var_idx] = False

        # Update satisfied clauses
        chosen = assignment[var_idx]
        for c_idx, clause in enumerate(instance.clauses):
            if satisfied[c_idx]:
                continue
            for lit in clause:
                if abs(lit) == var_num:
                    if (lit > 0 and chosen) or (lit < 0 and not chosen):
                        satisfied[c_idx] = True
                    break

    total_weight, n_satisfied = instance.evaluate(assignment)
    return MaxSATSolution(
        assignment=assignment,
        satisfied_weight=total_weight,
        n_satisfied=n_satisfied,
    )


def random_assignment(instance: MaxSATInstance,
                      seed: int = 42) -> MaxSATSolution:
    """Random truth assignment baseline.

    Args:
        instance: A MaxSATInstance.
        seed: Random seed.

    Returns:
        A MaxSATSolution.
    """
    rng = np.random.default_rng(seed)
    assignment = [bool(x) for x in rng.choice([True, False], size=instance.n_vars)]
    total_weight, n_satisfied = instance.evaluate(assignment)
    return MaxSATSolution(
        assignment=assignment,
        satisfied_weight=total_weight,
        n_satisfied=n_satisfied,
    )


if __name__ == "__main__":
    inst = MaxSATInstance.random(n_vars=10, n_clauses=20)
    sol = greedy_maxsat(inst)
    rand_sol = random_assignment(inst)
    print(f"Instance: {inst.n_vars} vars, {len(inst.clauses)} clauses, "
          f"total weight={inst.total_weight():.1f}")
    print(f"Greedy: {sol}")
    print(f"Random: {rand_sol}")

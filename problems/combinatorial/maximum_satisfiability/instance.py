"""Maximum Satisfiability Problem (MAX-SAT).

Problem: Given a CNF formula with weighted clauses, find a truth assignment
that maximizes the total weight of satisfied clauses.

Complexity: NP-hard. MAX-2SAT is APX-complete. Random assignment gives
expected weight >= W/2 (1/2-approximation).

References:
    Garey, M. R., & Johnson, D. S. (1979). Computers and Intractability:
    A Guide to the Theory of NP-Completeness.

    Goemans, M. X., & Williamson, D. P. (1994). New 3/4-approximation
    algorithms for the maximum satisfiability problem. SIAM Journal on
    Discrete Mathematics, 7(4), 656-666.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MaxSATInstance:
    """MAX-SAT problem instance.

    A clause is a list of integers. Positive integer k means variable k
    appears positive; negative integer -k means variable k appears negated.
    Variables are 1-indexed.

    Args:
        n_vars: Number of Boolean variables (1-indexed).
        clauses: List of clauses. Each clause is a list of signed integers.
        weights: Weight for each clause. len(weights) == len(clauses).
    """
    n_vars: int
    clauses: list[list[int]]
    weights: np.ndarray

    @classmethod
    def random(cls, n_vars: int = 10, n_clauses: int = 20,
               max_clause_len: int = 3, seed: int = 42) -> MaxSATInstance:
        """Generate a random MAX-SAT instance.

        Args:
            n_vars: Number of variables.
            n_clauses: Number of clauses.
            max_clause_len: Maximum literals per clause.
            seed: Random seed.

        Returns:
            A random MaxSATInstance.
        """
        rng = np.random.default_rng(seed)
        clauses = []
        for _ in range(n_clauses):
            clause_len = rng.integers(1, max_clause_len + 1)
            vars_in_clause = rng.choice(n_vars, size=clause_len, replace=False) + 1
            signs = rng.choice([-1, 1], size=clause_len)
            clause = (vars_in_clause * signs).tolist()
            clauses.append(clause)
        weights = rng.integers(1, 11, size=n_clauses).astype(float)
        return cls(n_vars=n_vars, clauses=clauses, weights=weights)

    def evaluate(self, assignment: list[bool]) -> tuple[float, int]:
        """Evaluate a truth assignment.

        Args:
            assignment: List of length n_vars. assignment[i] is True/False
                for variable i+1.

        Returns:
            Tuple of (total weight of satisfied clauses, number satisfied).
        """
        total_weight = 0.0
        n_satisfied = 0
        for idx, clause in enumerate(self.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                val = assignment[var_idx]
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
            if satisfied:
                total_weight += self.weights[idx]
                n_satisfied += 1
        return total_weight, n_satisfied

    def total_weight(self) -> float:
        """Return total weight of all clauses."""
        return float(self.weights.sum())


@dataclass
class MaxSATSolution:
    """Solution to a MAX-SAT problem.

    Args:
        assignment: Truth assignment (list of bool, 0-indexed).
        satisfied_weight: Total weight of satisfied clauses.
        n_satisfied: Number of satisfied clauses.
    """
    assignment: list[bool]
    satisfied_weight: float
    n_satisfied: int

    def __repr__(self) -> str:
        return (f"MaxSATSolution(weight={self.satisfied_weight:.1f}, "
                f"satisfied={self.n_satisfied})")

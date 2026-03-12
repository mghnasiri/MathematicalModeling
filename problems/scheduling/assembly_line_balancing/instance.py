"""Simple Assembly Line Balancing Problem (SALBP-1) instance and solution.

Problem: Given a set of tasks with processing times and precedence constraints,
and a fixed cycle time C, assign tasks to the minimum number of workstations
such that precedence is respected and no station exceeds the cycle time.

Notation: SALBP-1
Complexity: NP-hard (Wee & Magazine, 1982)

References:
    Scholl, A., & Becker, C. (2006). State-of-the-art exact and heuristic
    solution procedures for simple assembly line balancing. European Journal
    of Operational Research, 168(3), 666-693.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SALBPInstance:
    """SALBP-1 instance.

    Attributes:
        n_tasks: Number of tasks.
        processing_times: Processing time for each task.
        precedences: List of (predecessor, successor) pairs.
        cycle_time: Maximum time allowed per station.
    """
    n_tasks: int
    processing_times: np.ndarray
    precedences: list[tuple[int, int]]
    cycle_time: float

    def successors(self) -> dict[int, list[int]]:
        """Return dict mapping each task to its immediate successors."""
        succ: dict[int, list[int]] = {i: [] for i in range(self.n_tasks)}
        for pred, s in self.precedences:
            succ[pred].append(s)
        return succ

    def predecessors(self) -> dict[int, list[int]]:
        """Return dict mapping each task to its immediate predecessors."""
        pred_map: dict[int, list[int]] = {i: [] for i in range(self.n_tasks)}
        for p, s in self.precedences:
            pred_map[s].append(p)
        return pred_map

    @classmethod
    def random(cls, n_tasks: int = 10, cycle_time: float = 20.0,
               seed: int = 42) -> SALBPInstance:
        """Generate a random SALBP-1 instance.

        Args:
            n_tasks: Number of tasks.
            cycle_time: Cycle time constraint.
            seed: Random seed.

        Returns:
            A random SALBPInstance.
        """
        rng = np.random.default_rng(seed)
        processing_times = rng.uniform(1.0, cycle_time * 0.6, size=n_tasks)

        # Generate precedences as a DAG (chain + some extra edges)
        precedences = []
        for i in range(n_tasks - 1):
            if rng.random() < 0.4:
                precedences.append((i, i + 1))

        return cls(n_tasks=n_tasks, processing_times=processing_times,
                   precedences=precedences, cycle_time=cycle_time)


@dataclass
class SALBPSolution:
    """Solution to a SALBP-1 instance.

    Attributes:
        assignment: Dict mapping task index to station index.
        n_stations: Number of stations used.
        station_times: Total processing time at each station.
        feasible: Whether precedence and cycle time are respected.
    """
    assignment: dict[int, int]
    n_stations: int
    station_times: list[float]
    feasible: bool

    def __repr__(self) -> str:
        return (f"SALBPSolution(n_stations={self.n_stations}, "
                f"feasible={self.feasible})")

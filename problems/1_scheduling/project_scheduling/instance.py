"""Multi-Project Scheduling Problem (MPSP).

Given P projects, each with a set of activities, precedence constraints,
and resource requirements, schedule all activities across shared renewable
resources to minimize total project delays (sum of project makespans or
total weighted tardiness).

Each project has its own precedence graph and deadline. Resources are shared
across projects.

Complexity: NP-hard (extends RCPSP).

References:
    Lova, A., Tormos, P., Cervantes, M., & Barber, F. (2009). An efficient
    hybrid genetic algorithm for scheduling projects with resource constraints
    and multiple execution modes. International Journal of Production Economics,
    117(2), 302-316.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Project:
    """A single project within the multi-project instance.

    Attributes:
        project_id: Project identifier.
        n_activities: Number of activities (including dummy source/sink).
        durations: Processing time for each activity.
        predecessors: predecessors[j] = list of predecessor activities.
        resource_requirements: resource_requirements[j][k] = units of resource k
                               needed by activity j, shape (n_activities, n_resources).
        deadline: Target completion time.
        weight: Priority weight.
    """

    project_id: int
    n_activities: int
    durations: np.ndarray
    predecessors: list[list[int]]
    resource_requirements: np.ndarray
    deadline: int
    weight: float


@dataclass
class MultiProjectInstance:
    """Multi-project scheduling instance.

    Attributes:
        projects: List of Project objects.
        n_resources: Number of renewable resource types.
        resource_capacities: Available units per resource type.
    """

    projects: list[Project]
    n_resources: int
    resource_capacities: np.ndarray

    @property
    def n_projects(self) -> int:
        return len(self.projects)

    @classmethod
    def random(cls, n_projects: int = 3, n_activities: int = 6,
               n_resources: int = 2, seed: int | None = None
               ) -> MultiProjectInstance:
        """Generate a random multi-project instance.

        Args:
            n_projects: Number of projects.
            n_activities: Activities per project (including source/sink).
            n_resources: Number of resource types.
            seed: Random seed.

        Returns:
            A random MultiProjectInstance.
        """
        rng = np.random.default_rng(seed)
        resource_caps = rng.integers(3, 8, size=n_resources)
        projects = []

        for p in range(n_projects):
            durations = np.zeros(n_activities, dtype=int)
            # Source (0) and sink (n_activities-1) have zero duration
            durations[1:-1] = rng.integers(1, 10, size=n_activities - 2)

            # Build precedence: chain-like with some parallel activities
            predecessors: list[list[int]] = [[] for _ in range(n_activities)]
            for j in range(1, n_activities):
                if j == n_activities - 1:
                    # Sink: all non-sink activities with no successor
                    has_succ = set()
                    for jj in range(1, n_activities - 1):
                        for pred_list in predecessors:
                            if jj in []:
                                pass
                    # Simple: make sink depend on last few activities
                    predecessors[j] = list(range(max(1, n_activities - 3),
                                                  n_activities - 1))
                else:
                    # Each activity depends on at least one earlier activity
                    if j == 1:
                        predecessors[j] = [0]
                    else:
                        n_preds = rng.integers(1, min(3, j) + 1)
                        preds = rng.choice(range(max(0, j - 3), j),
                                           size=min(int(n_preds), j),
                                           replace=False).tolist()
                        predecessors[j] = sorted(preds)

            # Resource requirements
            req = np.zeros((n_activities, n_resources), dtype=int)
            req[1:-1] = rng.integers(0, 3, size=(n_activities - 2, n_resources))

            # Deadline: sum of durations (generous)
            deadline = int(durations.sum()) + rng.integers(0, 10)
            weight = float(rng.uniform(0.5, 2.0))

            projects.append(Project(
                project_id=p, n_activities=n_activities,
                durations=durations, predecessors=predecessors,
                resource_requirements=req, deadline=deadline,
                weight=weight,
            ))

        return cls(projects=projects, n_resources=n_resources,
                   resource_capacities=resource_caps)


@dataclass
class MultiProjectSolution:
    """Solution to a multi-project scheduling problem.

    Attributes:
        start_times: List of arrays, start_times[p][j] = start time of
                     activity j in project p.
        project_makespans: Makespan for each project.
        objective: Total weighted tardiness.
    """

    start_times: list[np.ndarray]
    project_makespans: list[int]
    objective: float

    def __repr__(self) -> str:
        return (f"MultiProjectSolution(makespans={self.project_makespans}, "
                f"objective={self.objective:.2f})")

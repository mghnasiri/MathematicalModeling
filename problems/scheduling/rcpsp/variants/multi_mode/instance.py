"""
Multi-Mode Resource-Constrained Project Scheduling (MRCPSP).

Problem notation: MPS | prec | Cmax

Each activity can be executed in one of several modes, each with
different duration and resource requirements. Activities have
precedence constraints. Minimize project makespan.

Complexity: Strongly NP-hard.

References:
    Talbot, F.B. (1982). Resource-constrained project scheduling with
    time-resource tradeoffs: the nonpreemptive case. Management Science,
    28(10), 1197-1210. https://doi.org/10.1287/mnsc.28.10.1197

    Hartmann, S. & Briskorn, D. (2010). A survey of variants and
    extensions of the resource-constrained project scheduling problem.
    European Journal of Operational Research, 207(1), 1-14.
    https://doi.org/10.1016/j.ejor.2009.11.005
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MRCPSPInstance:
    """Multi-Mode RCPSP instance.

    Attributes:
        n: Number of real activities (0=source, n+1=sink are dummies).
        num_resources: Number of renewable resource types.
        resource_capacities: Capacity per resource type, shape (num_resources,).
        modes: modes[j] = list of (duration, resource_reqs) per mode for activity j.
               resource_reqs shape: (num_resources,).
        successors: successors[j] = list of successor activities.
        name: Optional instance name.
    """

    n: int
    num_resources: int
    resource_capacities: np.ndarray
    modes: list[list[tuple[int, np.ndarray]]]
    successors: list[list[int]]
    name: str = ""

    def __post_init__(self):
        self.resource_capacities = np.asarray(self.resource_capacities, dtype=int)

    @classmethod
    def random(
        cls,
        n: int = 8,
        num_resources: int = 2,
        max_modes: int = 3,
        seed: int | None = None,
    ) -> MRCPSPInstance:
        rng = np.random.default_rng(seed)
        caps = rng.integers(5, 15, size=num_resources)

        # Generate modes for n+2 activities (0=source, n+1=sink)
        modes: list[list[tuple[int, np.ndarray]]] = []
        # Source: 1 mode, duration 0
        modes.append([(0, np.zeros(num_resources, dtype=int))])
        for j in range(1, n + 1):
            nm = rng.integers(1, max_modes + 1)
            job_modes = []
            for _ in range(nm):
                dur = int(rng.integers(1, 15))
                reqs = rng.integers(0, np.maximum(caps // 2, 1) + 1, size=num_resources)
                job_modes.append((dur, reqs))
            modes.append(job_modes)
        # Sink: 1 mode, duration 0
        modes.append([(0, np.zeros(num_resources, dtype=int))])

        # Generate random precedence DAG
        successors: list[list[int]] = [[] for _ in range(n + 2)]
        # Ensure connectivity: each real activity has at least source as predecessor
        for j in range(1, n + 1):
            if rng.random() < 0.3 and j > 1:
                pred = rng.integers(1, j)
                if j not in successors[pred]:
                    successors[pred].append(j)
            else:
                if j not in successors[0]:
                    successors[0].append(j)
        # Ensure all lead to sink
        for j in range(1, n + 1):
            if not successors[j]:
                successors[j].append(n + 1)

        return cls(n=n, num_resources=num_resources,
                   resource_capacities=caps, modes=modes,
                   successors=successors, name=f"random_{n}")

    def predecessors(self) -> list[list[int]]:
        """Compute predecessor list from successors."""
        preds: list[list[int]] = [[] for _ in range(self.n + 2)]
        for j in range(self.n + 2):
            for s in self.successors[j]:
                preds[s].append(j)
        return preds


@dataclass
class MRCPSPSolution:
    """MRCPSP solution.

    Attributes:
        mode_assignments: mode_assignments[j] = chosen mode index for activity j.
        start_times: start_times[j] = start time of activity j.
        makespan: Project makespan.
    """

    mode_assignments: list[int]
    start_times: list[int]
    makespan: int

    def __repr__(self) -> str:
        return f"MRCPSPSolution(makespan={self.makespan})"


def validate_solution(
    instance: MRCPSPInstance, solution: MRCPSPSolution
) -> tuple[bool, list[str]]:
    errors = []
    n = instance.n

    # Check mode assignments are valid
    for j in range(n + 2):
        m = solution.mode_assignments[j]
        if m < 0 or m >= len(instance.modes[j]):
            errors.append(f"Activity {j}: invalid mode {m}")

    # Check precedence
    for j in range(n + 2):
        for s in instance.successors[j]:
            dur_j = instance.modes[j][solution.mode_assignments[j]][0]
            if solution.start_times[s] < solution.start_times[j] + dur_j:
                errors.append(
                    f"Precedence violated: {j}(end={solution.start_times[j]+dur_j}) -> "
                    f"{s}(start={solution.start_times[s]})")

    # Check resource constraints at each time point
    if not errors:
        max_t = solution.makespan
        for t in range(max_t):
            usage = np.zeros(instance.num_resources, dtype=int)
            for j in range(n + 2):
                m = solution.mode_assignments[j]
                dur, reqs = instance.modes[j][m]
                if solution.start_times[j] <= t < solution.start_times[j] + dur:
                    usage += reqs
            for r in range(instance.num_resources):
                if usage[r] > instance.resource_capacities[r]:
                    errors.append(
                        f"Resource {r} exceeded at t={t}: {usage[r]} > "
                        f"{instance.resource_capacities[r]}")
                    break
            if errors:
                break

    # Check makespan
    actual_ms = 0
    for j in range(n + 2):
        m = solution.mode_assignments[j]
        dur = instance.modes[j][m][0]
        actual_ms = max(actual_ms, solution.start_times[j] + dur)
    if actual_ms != solution.makespan:
        errors.append(f"Makespan {solution.makespan} != actual {actual_ms}")

    return len(errors) == 0, errors


def small_mrcpsp_4() -> MRCPSPInstance:
    """Small instance: 4 real activities, 2 resources, multiple modes."""
    modes = [
        [(0, np.array([0, 0]))],                              # source
        [(3, np.array([2, 1])), (5, np.array([1, 1]))],       # act 1
        [(4, np.array([3, 0])), (2, np.array([2, 2]))],       # act 2
        [(2, np.array([1, 2])), (4, np.array([1, 1]))],       # act 3
        [(3, np.array([2, 1])), (6, np.array([1, 0]))],       # act 4
        [(0, np.array([0, 0]))],                              # sink
    ]
    successors = [
        [1, 2],    # source -> 1, 2
        [3],       # 1 -> 3
        [4],       # 2 -> 4
        [5],       # 3 -> sink
        [5],       # 4 -> sink
        [],        # sink
    ]
    return MRCPSPInstance(
        n=4, num_resources=2,
        resource_capacities=np.array([4, 3]),
        modes=modes, successors=successors,
        name="small_4",
    )


if __name__ == "__main__":
    inst = small_mrcpsp_4()
    print(f"{inst.name}: n={inst.n}, resources={inst.num_resources}")

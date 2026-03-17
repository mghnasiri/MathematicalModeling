"""
Resource-Constrained Project Scheduling — Instance and Solution Definitions

The RCPSP schedules n activities subject to precedence constraints and
renewable resource constraints. Activities 0 (source) and n+1 (sink) are
dummy activities with zero duration.

Notation: PS | prec | Cmax
    - PS: project scheduling with resources
    - prec: precedence constraints
    - Cmax: minimize makespan (project duration)

Reference: Kolisch, R. & Hartmann, S. (2006).
           "Experimental investigation of heuristics for RCPSP."
           European Journal of Operational Research, 169(1), 16-37.
           https://doi.org/10.1016/j.ejor.2004.01.035
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RCPSPInstance:
    """
    A resource-constrained project scheduling instance.

    Activities are numbered 0..n+1 where 0=source, n+1=sink.

    Attributes:
        n: Number of non-dummy activities (activities 1..n).
        num_resources: Number of renewable resource types.
        durations: Duration of each activity, shape (n+2,).
        resource_demands: Resource demands, shape (n+2, num_resources).
        resource_capacities: Capacity of each resource, shape (num_resources,).
        successors: Dict activity -> list of successor activities.
        predecessors: Dict activity -> list of predecessor activities.
    """
    n: int
    num_resources: int
    durations: np.ndarray
    resource_demands: np.ndarray
    resource_capacities: np.ndarray
    successors: dict[int, list[int]]
    predecessors: dict[int, list[int]]

    def __post_init__(self):
        total = self.n + 2
        assert self.durations.shape == (total,), (
            f"durations shape {self.durations.shape} != ({total},)"
        )
        assert self.resource_demands.shape == (total, self.num_resources), (
            f"resource_demands shape {self.resource_demands.shape} != "
            f"({total}, {self.num_resources})"
        )
        assert self.resource_capacities.shape == (self.num_resources,)
        assert self.durations[0] == 0, "Source dummy must have 0 duration"
        assert self.durations[self.n + 1] == 0, "Sink dummy must have 0 duration"

    @classmethod
    def random(
        cls,
        n: int,
        num_resources: int = 2,
        seed: int | None = None,
        max_duration: int = 10,
        resource_factor: float = 0.5,
        resource_strength: float = 0.5,
    ) -> RCPSPInstance:
        """
        Generate a random RCPSP instance.

        Args:
            n: Number of non-dummy activities.
            num_resources: Number of resource types.
            seed: Random seed.
            max_duration: Maximum activity duration.
            resource_factor: Fraction of resources used by each activity.
            resource_strength: Controls resource tightness (0=tight, 1=loose).

        Returns:
            A random RCPSPInstance.
        """
        rng = np.random.default_rng(seed)
        total = n + 2

        # Durations
        durations = np.zeros(total, dtype=int)
        durations[1:n + 1] = rng.integers(1, max_duration + 1, size=n)

        # Resource demands
        resource_demands = np.zeros((total, num_resources), dtype=int)
        for j in range(1, n + 1):
            for k in range(num_resources):
                if rng.random() < resource_factor:
                    resource_demands[j, k] = rng.integers(1, 6)

        # Resource capacities
        max_demand = resource_demands[1:n + 1].max(axis=0)
        avg_demand = resource_demands[1:n + 1].mean(axis=0)
        resource_capacities = np.maximum(
            max_demand,
            (resource_strength * max_demand +
             (1 - resource_strength) * avg_demand * n / max(1, n // 3)).astype(int),
        )

        # Precedence: random DAG
        successors: dict[int, list[int]] = {i: [] for i in range(total)}
        predecessors: dict[int, list[int]] = {i: [] for i in range(total)}

        # Generate random precedence (order-based)
        order = rng.permutation(np.arange(1, n + 1)).tolist()
        n_edges = max(n - 1, int(n * 1.5))

        for i in range(min(n_edges, n - 1)):
            src = order[i]
            tgt = order[i + 1]
            if tgt not in successors[src]:
                successors[src].append(tgt)
                predecessors[tgt].append(src)

        # Add some random edges
        for _ in range(n_edges - (n - 1)):
            i_idx = rng.integers(0, n - 1)
            j_idx = rng.integers(i_idx + 1, n)
            src, tgt = order[i_idx], order[j_idx]
            if tgt not in successors[src]:
                successors[src].append(tgt)
                predecessors[tgt].append(src)

        # Connect source and sink
        for j in range(1, n + 1):
            if not predecessors[j]:
                successors[0].append(j)
                predecessors[j].append(0)
            if not successors[j]:
                successors[j].append(n + 1)
                predecessors[n + 1].append(j)

        return cls(
            n=n,
            num_resources=num_resources,
            durations=durations,
            resource_demands=resource_demands,
            resource_capacities=resource_capacities,
            successors=successors,
            predecessors=predecessors,
        )

    @classmethod
    def from_arrays(
        cls,
        durations: list[int],
        resource_demands: list[list[int]],
        resource_capacities: list[int],
        successors: dict[int, list[int]],
    ) -> RCPSPInstance:
        """
        Create instance from arrays.

        Args:
            durations: List of n+2 durations (including dummies).
            resource_demands: (n+2) x K resource demands.
            resource_capacities: K resource capacities.
            successors: Dict activity -> list of successors.

        Returns:
            An RCPSPInstance.
        """
        dur = np.array(durations, dtype=int)
        n = len(dur) - 2
        rd = np.array(resource_demands, dtype=int)
        rc = np.array(resource_capacities, dtype=int)
        num_resources = len(resource_capacities)

        # Build predecessors
        predecessors: dict[int, list[int]] = {i: [] for i in range(n + 2)}
        for i, succs in successors.items():
            for j in succs:
                predecessors[j].append(i)

        return cls(
            n=n,
            num_resources=num_resources,
            durations=dur,
            resource_demands=rd,
            resource_capacities=rc,
            successors=successors,
            predecessors=predecessors,
        )

    def topological_order(self) -> list[int]:
        """
        Compute a topological ordering of activities.

        Returns:
            List of activity indices in topological order.
        """
        total = self.n + 2
        in_degree = {i: len(self.predecessors.get(i, [])) for i in range(total)}
        queue = [i for i in range(total) if in_degree[i] == 0]
        order = []

        while queue:
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for succ in self.successors.get(node, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def critical_path_length(self) -> int:
        """
        Compute the critical path length (lower bound, ignoring resources).

        Returns:
            Length of the longest path from source to sink.
        """
        total = self.n + 2
        earliest = np.zeros(total, dtype=int)
        for act in self.topological_order():
            for succ in self.successors.get(act, []):
                earliest[succ] = max(
                    earliest[succ],
                    earliest[act] + self.durations[act],
                )
        return int(earliest[self.n + 1])

    def earliest_start_times(self) -> np.ndarray:
        """Compute earliest start times (ignoring resources)."""
        total = self.n + 2
        es = np.zeros(total, dtype=int)
        for act in self.topological_order():
            for succ in self.successors.get(act, []):
                es[succ] = max(es[succ], es[act] + self.durations[act])
        return es

    def latest_start_times(self, makespan: int | None = None) -> np.ndarray:
        """Compute latest start times (backward pass)."""
        total = self.n + 2
        if makespan is None:
            makespan = self.critical_path_length()
        ls = np.full(total, makespan, dtype=int)
        ls[self.n + 1] = makespan

        order = self.topological_order()
        for act in reversed(order):
            for succ in self.successors.get(act, []):
                ls[act] = min(ls[act], ls[succ] - self.durations[act])
        return ls


@dataclass
class RCPSPSolution:
    """
    A solution to an RCPSP instance.

    Attributes:
        start_times: Start time of each activity, shape (n+2,).
        makespan: Project makespan (= start_times[n+1]).
    """
    start_times: np.ndarray
    makespan: int

    def __repr__(self) -> str:
        return f"RCPSPSolution(makespan={self.makespan})"


def compute_makespan_from_starts(
    instance: RCPSPInstance,
    start_times: np.ndarray,
) -> int:
    """Compute makespan from start times."""
    return int(start_times[instance.n + 1])


def validate_solution(
    instance: RCPSPInstance,
    start_times: np.ndarray,
) -> tuple[bool, list[str]]:
    """
    Validate an RCPSP solution.

    Checks:
    1. All activities scheduled.
    2. Precedence constraints satisfied.
    3. Resource constraints satisfied at all time points.

    Returns:
        (is_valid, list of violations).
    """
    violations = []
    total = instance.n + 2

    if len(start_times) != total:
        violations.append(
            f"Expected {total} start times, got {len(start_times)}"
        )
        return False, violations

    # Check precedence
    for act in range(total):
        for succ in instance.successors.get(act, []):
            if start_times[act] + instance.durations[act] > start_times[succ]:
                violations.append(
                    f"Precedence: act {act} ends at "
                    f"{start_times[act] + instance.durations[act]} > "
                    f"act {succ} starts at {start_times[succ]}"
                )

    # Check resources
    if not violations:
        makespan = int(start_times[instance.n + 1])
        for t in range(makespan):
            for k in range(instance.num_resources):
                usage = 0
                for j in range(total):
                    if (start_times[j] <= t <
                            start_times[j] + instance.durations[j]):
                        usage += instance.resource_demands[j, k]
                if usage > instance.resource_capacities[k]:
                    violations.append(
                        f"Resource {k} at time {t}: usage {usage} > "
                        f"capacity {instance.resource_capacities[k]}"
                    )

    return len(violations) == 0, violations


if __name__ == "__main__":
    print("=== RCPSP Instance Demo ===\n")

    inst = RCPSPInstance.random(n=8, num_resources=2, seed=42)
    print(f"Activities: {inst.n} (+ 2 dummies)")
    print(f"Resources: {inst.num_resources}")
    print(f"Capacities: {inst.resource_capacities}")
    print(f"Critical path length: {inst.critical_path_length()}")

    topo = inst.topological_order()
    print(f"Topological order: {topo}")

    es = inst.earliest_start_times()
    print(f"Earliest starts: {es}")

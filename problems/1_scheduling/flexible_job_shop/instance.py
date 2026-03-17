"""
Flexible Job Shop Scheduling — Problem Instance and Solution Definitions

The FJSP extends the classical Job Shop by allowing each operation to be
processed on any machine from a set of eligible machines. This introduces
a routing sub-problem (machine assignment) on top of sequencing.

Notation: FJm | | Cmax
    - FJm: m machines, flexible routing
    - Cmax: makespan objective

Variants:
    - Total FJSP (T-FJSP): every operation can run on any machine
    - Partial FJSP (P-FJSP): each operation has a subset of eligible machines

Reference: Brucker, P. & Schlie, R. (1990).
           "Job-shop scheduling with multi-purpose machines."
           Computing, 45(4), 369-375.
           https://doi.org/10.1007/BF02238804
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FlexibleOperation:
    """
    A single operation in a flexible job shop problem.

    Attributes:
        job: Job index.
        position: Position within its job (0-indexed).
        eligible_machines: Dict mapping machine_id -> processing_time.
    """
    job: int
    position: int
    eligible_machines: dict[int, int]


@dataclass
class FlexibleJobShopInstance:
    """
    A flexible job shop scheduling instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        jobs: List of n lists, where jobs[j] is a list of dicts.
              Each dict maps machine_id -> processing_time for that operation.
    """
    n: int
    m: int
    jobs: list[list[dict[int, int]]]

    def __post_init__(self):
        assert len(self.jobs) == self.n
        for j, job_ops in enumerate(self.jobs):
            for k, eligible in enumerate(job_ops):
                assert len(eligible) > 0, (
                    f"Job {j}, op {k}: no eligible machines"
                )
                for mach, pt in eligible.items():
                    assert 0 <= mach < self.m, (
                        f"Job {j}, op {k}: machine {mach} out of range"
                    )
                    assert pt > 0, (
                        f"Job {j}, op {k}: non-positive processing time {pt}"
                    )

    @classmethod
    def random(
        cls,
        n: int,
        m: int,
        max_ops: int | None = None,
        flexibility: float = 0.5,
        p_low: int = 1,
        p_high: int = 99,
        seed: int | None = None,
    ) -> FlexibleJobShopInstance:
        """
        Generate a random flexible job shop instance.

        Args:
            n: Number of jobs.
            m: Number of machines.
            max_ops: Max operations per job (default: m).
            flexibility: Fraction of machines eligible per operation (0, 1].
            p_low: Min processing time.
            p_high: Max processing time.
            seed: Random seed.

        Returns:
            A random FlexibleJobShopInstance.
        """
        rng = np.random.default_rng(seed)
        if max_ops is None:
            max_ops = m

        n_eligible = max(1, int(m * flexibility))
        jobs = []
        for _ in range(n):
            num_ops = rng.integers(max(1, max_ops // 2), max_ops + 1)
            ops = []
            for _ in range(num_ops):
                machines = rng.choice(m, size=n_eligible, replace=False).tolist()
                times = rng.integers(p_low, p_high + 1, size=n_eligible).tolist()
                ops.append(dict(zip(machines, times)))
            jobs.append(ops)
        return cls(n=n, m=m, jobs=jobs)

    @classmethod
    def random_total(
        cls,
        n: int,
        m: int,
        max_ops: int | None = None,
        p_low: int = 1,
        p_high: int = 99,
        seed: int | None = None,
    ) -> FlexibleJobShopInstance:
        """Generate a Total FJSP (all machines eligible for every operation)."""
        return cls.random(n=n, m=m, max_ops=max_ops, flexibility=1.0,
                          p_low=p_low, p_high=p_high, seed=seed)

    @classmethod
    def from_standard_format(cls, text: str) -> FlexibleJobShopInstance:
        """
        Parse a standard FJSP instance format.

        Format:
            n m avg_machines_per_op
            num_ops  n_eligible_1 m1 p1 m2 p2 ...  n_eligible_2 m1 p1 ...
            ...

        Args:
            text: Instance text.

        Returns:
            A FlexibleJobShopInstance.
        """
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        header = lines[0].split()
        n, m = int(header[0]), int(header[1])

        jobs = []
        for j in range(n):
            values = list(map(int, lines[j + 1].split()))
            idx = 0
            num_ops = values[idx]; idx += 1
            ops = []
            for _ in range(num_ops):
                n_eligible = values[idx]; idx += 1
                eligible = {}
                for _ in range(n_eligible):
                    mach = values[idx]; idx += 1
                    pt = values[idx]; idx += 1
                    eligible[mach] = pt
                ops.append(eligible)
            jobs.append(ops)
        return cls(n=n, m=m, jobs=jobs)

    @classmethod
    def from_job_shop(
        cls,
        n: int,
        m: int,
        jobs: list[list[tuple[int, int]]],
    ) -> FlexibleJobShopInstance:
        """
        Convert a classical JSP instance to FJSP (each operation has one
        eligible machine).
        """
        fjsp_jobs = []
        for job_ops in jobs:
            ops = [{mach: pt} for mach, pt in job_ops]
            fjsp_jobs.append(ops)
        return cls(n=n, m=m, jobs=fjsp_jobs)

    def get_operation(self, job: int, position: int) -> FlexibleOperation:
        """Get operation at given position."""
        return FlexibleOperation(
            job=job,
            position=position,
            eligible_machines=self.jobs[job][position],
        )

    def num_operations(self, job: int) -> int:
        """Number of operations in a given job."""
        return len(self.jobs[job])

    def total_operations(self) -> int:
        """Total number of operations."""
        return sum(len(j) for j in self.jobs)

    def is_total(self) -> bool:
        """Check if this is a Total FJSP (all machines eligible everywhere)."""
        for job_ops in self.jobs:
            for eligible in job_ops:
                if len(eligible) != self.m:
                    return False
        return True


@dataclass
class FlexibleJobShopSolution:
    """
    A flexible job shop scheduling solution.

    Attributes:
        assignments: Dict (job, position) -> machine assigned.
        start_times: Dict (job, position) -> start time.
        makespan: The Cmax value.
    """
    assignments: dict[tuple[int, int], int]
    start_times: dict[tuple[int, int], int]
    makespan: int

    def __repr__(self) -> str:
        return f"FlexibleJobShopSolution(makespan={self.makespan})"


def compute_makespan(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
    start_times: dict[tuple[int, int], int],
) -> int:
    """Compute makespan from assignments and start times."""
    cmax = 0
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            if (j, k) in start_times and (j, k) in assignments:
                mach = assignments[(j, k)]
                pt = instance.jobs[j][k][mach]
                end = start_times[(j, k)] + pt
                cmax = max(cmax, end)
    return cmax


def validate_solution(
    instance: FlexibleJobShopInstance,
    assignments: dict[tuple[int, int], int],
    start_times: dict[tuple[int, int], int],
) -> tuple[bool, list[str]]:
    """
    Validate an FJSP solution.

    Checks:
    1. All operations scheduled and assigned.
    2. Each operation assigned to an eligible machine.
    3. Precedence constraints satisfied.
    4. No machine conflicts.

    Returns:
        (is_valid, list of violations).
    """
    violations = []

    # Check completeness and eligibility
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            if (j, k) not in assignments:
                violations.append(f"Op ({j},{k}) not assigned to a machine")
            elif assignments[(j, k)] not in instance.jobs[j][k]:
                violations.append(
                    f"Op ({j},{k}) assigned to ineligible machine "
                    f"{assignments[(j, k)]}"
                )
            if (j, k) not in start_times:
                violations.append(f"Op ({j},{k}) has no start time")

    if violations:
        return False, violations

    # Check precedence
    for j in range(instance.n):
        for k in range(len(instance.jobs[j]) - 1):
            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            if start_times[(j, k)] + pt > start_times[(j, k + 1)]:
                violations.append(
                    f"Precedence: job {j}, op {k} ends at "
                    f"{start_times[(j, k)] + pt} > op {k+1} at "
                    f"{start_times[(j, k + 1)]}"
                )

    # Check machine conflicts
    machine_ops: dict[int, list[tuple[int, int, int]]] = {}
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            s = start_times[(j, k)]
            machine_ops.setdefault(mach, []).append((s, s + pt, j))

    for mach, ops in machine_ops.items():
        ops.sort()
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0]:
                violations.append(
                    f"Machine {mach}: job {ops[i][2]} ends at {ops[i][1]} > "
                    f"job {ops[i+1][2]} starts at {ops[i+1][0]}"
                )

    return len(violations) == 0, violations


if __name__ == "__main__":
    print("=== Flexible Job Shop Instance Demo ===\n")

    inst = FlexibleJobShopInstance.random(n=4, m=3, flexibility=0.6, seed=42)
    print(f"FJSP: {inst.n} jobs, {inst.m} machines")
    print(f"Total FJSP: {inst.is_total()}")
    print(f"Total operations: {inst.total_operations()}")

    for j in range(inst.n):
        print(f"\nJob {j}:")
        for k, eligible in enumerate(inst.jobs[j]):
            machines_str = ", ".join(
                f"M{m}({p})" for m, p in sorted(eligible.items())
            )
            print(f"  Op {k}: [{machines_str}]")

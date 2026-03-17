"""
Job Shop Scheduling — Problem Instance and Solution Definitions

Defines dataclasses for job shop scheduling problems (JSP).
Each job has a sequence of operations, each requiring a specific machine
for a given processing time. Different jobs may visit machines in different orders.

Notation: Jm | | Cmax
    - Jm: m machines, job-specific routing
    - Cmax: makespan objective

The disjunctive graph representation (Roy & Sussmann, 1964) is the
standard model for JSP.

Reference: Pinedo, M.L. (2016). "Scheduling: Theory, Algorithms, and Systems"
           5th Edition, Springer. Chapter 7.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Operation:
    """
    A single operation in a job shop problem.

    Attributes:
        job: Job index this operation belongs to.
        position: Position of this operation within its job (0-indexed).
        machine: Machine this operation must be processed on.
        processing_time: Processing time of this operation.
    """
    job: int
    position: int
    machine: int
    processing_time: int


@dataclass
class JobShopInstance:
    """
    A job shop scheduling problem instance.

    Attributes:
        n: Number of jobs.
        m: Number of machines.
        jobs: List of n lists, where jobs[j] is a list of (machine, processing_time)
              tuples representing the operations of job j in order.
    """
    n: int
    m: int
    jobs: list[list[tuple[int, int]]]

    def __post_init__(self):
        assert len(self.jobs) == self.n, (
            f"Expected {self.n} jobs, got {len(self.jobs)}"
        )
        for j, job_ops in enumerate(self.jobs):
            for pos, (machine, pt) in enumerate(job_ops):
                assert 0 <= machine < self.m, (
                    f"Job {j}, op {pos}: machine {machine} out of range [0, {self.m})"
                )
                assert pt >= 0, (
                    f"Job {j}, op {pos}: negative processing time {pt}"
                )

    @classmethod
    def random(
        cls,
        n: int,
        m: int,
        p_low: int = 1,
        p_high: int = 99,
        seed: int | None = None,
    ) -> JobShopInstance:
        """
        Generate a random job shop instance.

        Each job visits all m machines exactly once in a random order
        (classical JSP with full routing).

        Args:
            n: Number of jobs.
            m: Number of machines.
            p_low: Minimum processing time.
            p_high: Maximum processing time.
            seed: Random seed.

        Returns:
            A random JobShopInstance.
        """
        rng = np.random.default_rng(seed)
        jobs = []
        for _ in range(n):
            route = rng.permutation(m).tolist()
            times = rng.integers(p_low, p_high + 1, size=m).tolist()
            jobs.append(list(zip(route, times)))
        return cls(n=n, m=m, jobs=jobs)

    @classmethod
    def from_standard_format(
        cls,
        text: str,
    ) -> JobShopInstance:
        """
        Parse a standard JSP instance (OR-Library / Taillard format).

        Format:
            n m
            machine_0 time_0 machine_1 time_1 ... (for job 0)
            machine_0 time_0 machine_1 time_1 ... (for job 1)
            ...

        Args:
            text: Instance text.

        Returns:
            A JobShopInstance.
        """
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        first_line = lines[0].split()
        n, m = int(first_line[0]), int(first_line[1])
        jobs = []
        for j in range(n):
            values = list(map(int, lines[j + 1].split()))
            ops = []
            for k in range(m):
                machine = values[2 * k]
                pt = values[2 * k + 1]
                ops.append((machine, pt))
            jobs.append(ops)
        return cls(n=n, m=m, jobs=jobs)

    @classmethod
    def from_arrays(
        cls,
        machines: list[list[int]],
        processing_times: list[list[int]],
    ) -> JobShopInstance:
        """
        Create instance from separate machine and processing time arrays.

        Args:
            machines: machines[j][k] = machine for operation k of job j.
            processing_times: processing_times[j][k] = processing time.

        Returns:
            A JobShopInstance.
        """
        n = len(machines)
        m = max(max(row) for row in machines) + 1
        jobs = []
        for j in range(n):
            ops = list(zip(machines[j], processing_times[j]))
            jobs.append(ops)
        return cls(n=n, m=m, jobs=jobs)

    def get_operation(self, job: int, position: int) -> Operation:
        """Get operation at given position in given job."""
        machine, pt = self.jobs[job][position]
        return Operation(job=job, position=position, machine=machine,
                         processing_time=pt)

    def num_operations(self, job: int) -> int:
        """Number of operations in a given job."""
        return len(self.jobs[job])

    def total_operations(self) -> int:
        """Total number of operations across all jobs."""
        return sum(len(j) for j in self.jobs)

    def processing_time_matrix(self) -> np.ndarray:
        """
        Build processing time lookup matrix.

        Returns:
            Array of shape (n, m) where entry [j][k] is the processing
            time of the k-th operation of job j.
        """
        max_ops = max(len(j) for j in self.jobs)
        pt = np.zeros((self.n, max_ops), dtype=int)
        for j, ops in enumerate(self.jobs):
            for k, (_, p) in enumerate(ops):
                pt[j, k] = p
        return pt

    def machine_matrix(self) -> np.ndarray:
        """
        Build machine assignment lookup matrix.

        Returns:
            Array of shape (n, max_ops) where entry [j][k] is the machine
            of the k-th operation of job j.
        """
        max_ops = max(len(j) for j in self.jobs)
        mm = np.full((self.n, max_ops), -1, dtype=int)
        for j, ops in enumerate(self.jobs):
            for k, (mach, _) in enumerate(ops):
                mm[j, k] = mach
        return mm


@dataclass
class JobShopSolution:
    """
    A job shop scheduling solution.

    Attributes:
        start_times: Dict mapping (job, position) to start time.
        makespan: The Cmax value.
        machine_sequences: Optional dict mapping machine -> ordered list
                          of (job, position) on that machine.
    """
    start_times: dict[tuple[int, int], int]
    makespan: int
    machine_sequences: dict[int, list[tuple[int, int]]] | None = None

    def __repr__(self) -> str:
        return f"JobShopSolution(makespan={self.makespan})"


def compute_makespan(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
) -> int:
    """
    Compute makespan from start times.

    Args:
        instance: JSP instance.
        start_times: Dict (job, position) -> start time.

    Returns:
        The makespan.
    """
    cmax = 0
    for j in range(instance.n):
        for k, (_, pt) in enumerate(instance.jobs[j]):
            if (j, k) in start_times:
                end = start_times[(j, k)] + pt
                cmax = max(cmax, end)
    return cmax


def validate_solution(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
) -> tuple[bool, list[str]]:
    """
    Validate a JSP solution for feasibility.

    Checks:
    1. All operations scheduled.
    2. Precedence constraints satisfied (within each job).
    3. No machine conflicts (no two operations overlap on same machine).

    Args:
        instance: JSP instance.
        start_times: Dict (job, position) -> start time.

    Returns:
        (is_valid, list of violation messages).
    """
    violations = []

    # Check all operations scheduled
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            if (j, k) not in start_times:
                violations.append(f"Operation ({j},{k}) not scheduled")

    if violations:
        return False, violations

    # Check precedence
    for j in range(instance.n):
        for k in range(len(instance.jobs[j]) - 1):
            _, pt = instance.jobs[j][k]
            if start_times[(j, k)] + pt > start_times[(j, k + 1)]:
                violations.append(
                    f"Precedence violated: job {j}, op {k} ends at "
                    f"{start_times[(j, k)] + pt} > op {k+1} starts at "
                    f"{start_times[(j, k + 1)]}"
                )

    # Check machine conflicts
    machine_ops: dict[int, list[tuple[int, int, int]]] = {}
    for j in range(instance.n):
        for k, (mach, pt) in enumerate(instance.jobs[j]):
            start = start_times[(j, k)]
            machine_ops.setdefault(mach, []).append((start, start + pt, j))

    for mach, ops in machine_ops.items():
        ops.sort()
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0]:
                violations.append(
                    f"Machine {mach} conflict: job {ops[i][2]} ends at "
                    f"{ops[i][1]} > job {ops[i+1][2]} starts at {ops[i+1][0]}"
                )

    return len(violations) == 0, violations


def build_machine_sequences(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Build ordered machine sequences from start times.

    Args:
        instance: JSP instance.
        start_times: Dict (job, position) -> start time.

    Returns:
        Dict machine -> list of (job, position) ordered by start time.
    """
    machine_ops: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for j in range(instance.n):
        for k, (mach, _) in enumerate(instance.jobs[j]):
            if (j, k) in start_times:
                machine_ops.setdefault(mach, []).append(
                    (start_times[(j, k)], (j, k))
                )
    return {
        m: [op for _, op in sorted(ops)]
        for m, ops in machine_ops.items()
    }


# ---------------------------------------------------------------------------
# Classic benchmark instances
# ---------------------------------------------------------------------------

def ft06() -> JobShopInstance:
    """
    Fisher & Thompson 6x6 instance (ft06). Optimal makespan = 55.

    Reference: Fisher, H. & Thompson, G.L. (1963).
    """
    machines = [
        [2, 0, 1, 3, 5, 4],
        [1, 2, 4, 5, 0, 3],
        [2, 3, 5, 0, 1, 4],
        [1, 0, 2, 3, 4, 5],
        [2, 1, 4, 5, 0, 3],
        [1, 3, 5, 0, 4, 2],
    ]
    times = [
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ]
    return JobShopInstance.from_arrays(machines, times)


def ft10() -> JobShopInstance:
    """
    Fisher & Thompson 10x10 instance (ft10). Optimal makespan = 930.
    Remained unsolved for 26 years (1963-1989).

    Reference: Fisher, H. & Thompson, G.L. (1963).
    """
    machines = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
        [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
        [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
        [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
        [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
        [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
        [2, 0, 1, 5, 4, 6, 8, 9, 3, 7],
        [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
        [1, 0, 2, 6, 8, 9, 5, 3, 4, 7],
    ]
    times = [
        [29, 78, 9, 36, 49, 11, 62, 56, 44, 21],
        [43, 90, 75, 11, 69, 28, 46, 46, 72, 30],
        [91, 85, 39, 74, 90, 10, 12, 89, 45, 33],
        [81, 95, 71, 99, 9, 52, 85, 98, 22, 43],
        [14, 6, 22, 61, 26, 69, 21, 49, 72, 53],
        [84, 2, 52, 95, 48, 72, 47, 65, 6, 25],
        [46, 37, 61, 13, 32, 21, 32, 89, 30, 55],
        [31, 86, 46, 74, 32, 88, 19, 48, 36, 79],
        [76, 69, 76, 51, 85, 11, 40, 89, 26, 74],
        [85, 13, 61, 7, 64, 76, 47, 52, 90, 45],
    ]
    return JobShopInstance.from_arrays(machines, times)


if __name__ == "__main__":
    print("=== Job Shop Instance Demo ===\n")

    # ft06 benchmark
    inst = ft06()
    print(f"ft06: {inst.n} jobs, {inst.m} machines")
    print(f"Total operations: {inst.total_operations()}")
    for j in range(inst.n):
        ops_str = " -> ".join(
            f"M{m}({p})" for m, p in inst.jobs[j]
        )
        print(f"  Job {j}: {ops_str}")

    # Random instance
    print("\nRandom 5x3 instance:")
    rand_inst = JobShopInstance.random(n=5, m=3, seed=42)
    for j in range(rand_inst.n):
        ops_str = " -> ".join(
            f"M{m}({p})" for m, p in rand_inst.jobs[j]
        )
        print(f"  Job {j}: {ops_str}")

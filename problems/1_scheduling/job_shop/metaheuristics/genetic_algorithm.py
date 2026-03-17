"""
Genetic Algorithm for Job Shop Scheduling (Jm || Cmax)

Implements a GA using operation-based encoding (Bierwirth, 1995). Each
chromosome is a permutation of all operations (with job repetitions),
decoded into a schedule by iterating through genes and scheduling each
job's next operation as early as possible.

Algorithm:
    1. Initialize population from dispatching rules + random.
    2. Select parents via binary tournament.
    3. Apply Precedence Preserving Crossover (PPX) or Job-Order Crossover (JOX).
    4. Mutate by swapping two random genes.
    5. Decode chromosome and evaluate makespan.
    6. Replace worst individual if offspring is better.

Encoding (Bierwirth, 1995):
    A chromosome is a list of length n*m containing each job index exactly
    m times. The k-th occurrence of job j refers to the k-th operation of
    job j. Decoding processes genes left-to-right, scheduling each operation
    as early as possible given job precedence and machine constraints.

Notation: Jm || Cmax
Complexity: O(generations * pop_size * n * m) per run.

Reference:
    Bierwirth, C. (1995). A generalized permutation approach to job shop
    scheduling with genetic algorithms. OR Spektrum, 17(2), 87-92.
    https://doi.org/10.1007/BF01719250

    Cheng, R., Gen, M. & Tsujimura, Y. (1999). A tutorial survey of
    job-shop scheduling problems using genetic algorithms. Computers &
    Industrial Engineering, 36(2), 343-364.
    https://doi.org/10.1016/S0360-8352(99)00136-9
"""

from __future__ import annotations

import sys
import os
import time
import importlib.util
from collections import deque

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("job_shop_instance_mod", os.path.join(_parent_dir, "instance.py"))
JobShopInstance = _inst.JobShopInstance
JobShopSolution = _inst.JobShopSolution
compute_makespan = _inst.compute_makespan
build_machine_sequences = _inst.build_machine_sequences

_disp = _load_mod(
    "job_shop_dispatching_mod",
    os.path.join(_parent_dir, "heuristics", "dispatching_rules.py"),
)
dispatching_rule = _disp.dispatching_rule


def _encode_from_start_times(
    instance: JobShopInstance,
    start_times: dict[tuple[int, int], int],
) -> list[int]:
    """Convert start times to operation-based chromosome.

    Sort all operations by start time, breaking ties by job index,
    then output job indices (each job appears m times).
    """
    ops = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            ops.append((start_times.get((j, k), 0), j, k))
    ops.sort()
    return [j for _, j, _ in ops]


def _decode(
    instance: JobShopInstance,
    chromosome: list[int],
    op_machine: dict[tuple[int, int], int],
    op_pt: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], int], int]:
    """Decode chromosome to start times and makespan.

    Process genes left-to-right. For each gene (job j), schedule
    the next unscheduled operation of that job.

    Returns:
        (start_times dict, makespan)
    """
    job_count = [0] * instance.n
    machine_end = [0] * instance.m
    job_end = [0] * instance.n
    start_times: dict[tuple[int, int], int] = {}
    makespan = 0

    for gene in chromosome:
        j = gene
        k = job_count[j]
        job_count[j] += 1

        mach = op_machine[(j, k)]
        pt = op_pt[(j, k)]

        start = max(machine_end[mach], job_end[j])
        end = start + pt

        start_times[(j, k)] = start
        machine_end[mach] = end
        job_end[j] = end
        makespan = max(makespan, end)

    return start_times, makespan


def _jox_crossover(
    parent1: list[int],
    parent2: list[int],
    n_jobs: int,
    rng: np.random.Generator,
) -> list[int]:
    """Job-Order Crossover (JOX).

    Select a random subset of jobs. Positions of those jobs are inherited
    from parent1; remaining positions filled from parent2 preserving order.
    """
    n_select = max(1, n_jobs // 2)
    selected_jobs = set(rng.choice(n_jobs, size=n_select, replace=False).tolist())

    child = [None] * len(parent1)

    # Copy selected job positions from parent1
    for i, gene in enumerate(parent1):
        if gene in selected_jobs:
            child[i] = gene

    # Fill remaining from parent2
    p2_remaining = [g for g in parent2 if g not in selected_jobs]
    idx = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = p2_remaining[idx]
            idx += 1

    return child


def _mutate_swap(
    chromosome: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Swap two random genes."""
    result = list(chromosome)
    n = len(result)
    i, j = rng.choice(n, size=2, replace=False)
    result[i], result[j] = result[j], result[i]
    return result


def genetic_algorithm(
    instance: JobShopInstance,
    population_size: int = 30,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    time_limit: float | None = None,
    seed: int | None = None,
) -> JobShopSolution:
    """Solve JSP using a Genetic Algorithm with operation-based encoding.

    Args:
        instance: Job shop instance.
        population_size: Number of individuals.
        generations: Maximum generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation.
        time_limit: Time limit in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Best JobShopSolution found.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    n, m = instance.n, instance.m
    pop_size = population_size
    total_ops = n * m  # chromosome length

    # Precompute operation info
    op_machine: dict[tuple[int, int], int] = {}
    op_pt: dict[tuple[int, int], int] = {}
    for j in range(n):
        for k, (mach, pt) in enumerate(instance.jobs[j]):
            op_machine[(j, k)] = mach
            op_pt[(j, k)] = pt

    # ── Initialize population ────────────────────────────────────────────
    population: list[list[int]] = []
    fitnesses: list[int] = []

    # Seed from dispatching rules
    for rule in ["spt", "mwr", "lpt", "lwr"]:
        sol = dispatching_rule(instance, rule=rule)
        chrom = _encode_from_start_times(instance, sol.start_times)
        _, ms = _decode(instance, chrom, op_machine, op_pt)
        population.append(chrom)
        fitnesses.append(ms)

    # Fill rest with random chromosomes
    while len(population) < pop_size:
        chrom = []
        for j in range(n):
            chrom.extend([j] * len(instance.jobs[j]))
        rng.shuffle(chrom)
        _, ms = _decode(instance, chrom, op_machine, op_pt)
        population.append(chrom)
        fitnesses.append(ms)

    gbest_idx = int(np.argmin(fitnesses))
    gbest_chrom = list(population[gbest_idx])
    gbest_ms = fitnesses[gbest_idx]

    for gen in range(generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Binary tournament selection
        def tournament():
            a, b = rng.choice(pop_size, size=2, replace=False)
            return a if fitnesses[a] <= fitnesses[b] else b

        p1_idx = tournament()
        p2_idx = tournament()

        # Crossover
        if rng.random() < crossover_rate:
            child = _jox_crossover(
                population[p1_idx], population[p2_idx], n, rng,
            )
        else:
            child = list(population[p1_idx])

        # Mutation
        if rng.random() < mutation_rate:
            child = _mutate_swap(child, rng)

        # Decode and evaluate
        _, child_ms = _decode(instance, child, op_machine, op_pt)

        # Replace worst
        worst_idx = int(np.argmax(fitnesses))
        if child_ms < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = child_ms

            if child_ms < gbest_ms:
                gbest_ms = child_ms
                gbest_chrom = list(child)

    # Build final solution
    final_st, final_ms = _decode(instance, gbest_chrom, op_machine, op_pt)
    machine_sequences = build_machine_sequences(instance, final_st)

    return JobShopSolution(
        start_times=final_st,
        makespan=final_ms,
        machine_sequences=machine_sequences,
    )


if __name__ == "__main__":
    from instance import ft06

    print("=== Genetic Algorithm on ft06 (optimal=55) ===\n")
    inst = ft06()
    sol = genetic_algorithm(inst, generations=500, seed=42)
    print(f"GA makespan: {sol.makespan}")

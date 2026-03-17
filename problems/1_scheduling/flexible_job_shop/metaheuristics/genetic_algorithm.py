"""
Genetic Algorithm for Flexible Job Shop Scheduling (FJm || Cmax)

Implements a GA with integrated routing + sequencing encoding.
Each chromosome consists of:
1. Machine assignment vector (which machine for each operation)
2. Operation sequence (priority order for decoding)

Uses uniform crossover for machine assignment and order crossover
for the operation sequence.

Complexity: O(generations * pop_size * total_ops * m).

Reference:
    Pezzella, F., Morganti, G. & Ciaschetti, G. (2008).
    "A genetic algorithm for the Flexible Job-shop Scheduling Problem."
    Computers & Operations Research, 35(10), 3202-3212.
    https://doi.org/10.1016/j.cor.2007.02.014
"""

from __future__ import annotations
import sys
import os
import importlib.util
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

_inst = _load_mod("fjsp_instance_mod", os.path.join(_parent_dir, "instance.py"))
FlexibleJobShopInstance = _inst.FlexibleJobShopInstance
FlexibleJobShopSolution = _inst.FlexibleJobShopSolution
compute_makespan = _inst.compute_makespan
validate_solution = _inst.validate_solution


def genetic_algorithm(
    instance: FlexibleJobShopInstance,
    pop_size: int = 50,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    seed: int | None = None,
) -> FlexibleJobShopSolution:
    """
    Genetic Algorithm for FJm||Cmax.

    Args:
        instance: FJSP instance.
        pop_size: Population size.
        generations: Number of generations.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation.
        seed: Random seed.

    Returns:
        Best FlexibleJobShopSolution found.
    """
    rng = np.random.default_rng(seed)

    # Build operation list: (job, position) for all operations
    op_list = []
    for j in range(instance.n):
        for k in range(len(instance.jobs[j])):
            op_list.append((j, k))

    total_ops = len(op_list)

    # Initialize population
    population = [_random_chromosome(instance, op_list, rng) for _ in range(pop_size)]

    # Evaluate
    fitness = [_evaluate(instance, chromo, op_list) for chromo in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_chromo = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for gen in range(generations):
        new_pop = []

        # Elitism: keep best
        new_pop.append(best_chromo.copy())

        while len(new_pop) < pop_size:
            # Tournament selection
            p1 = _tournament(population, fitness, rng)
            p2 = _tournament(population, fitness, rng)

            if rng.random() < crossover_rate:
                c1, c2 = _crossover(p1, p2, total_ops, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.random() < mutation_rate:
                _mutate(instance, c1, op_list, rng)
            if rng.random() < mutation_rate:
                _mutate(instance, c2, op_list, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitness = [_evaluate(instance, chromo, op_list) for chromo in population]

        gen_best = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[gen_best] < best_fitness:
            best_fitness = fitness[gen_best]
            best_chromo = population[gen_best].copy()

    # Decode best solution
    assignments, start_times, makespan = _decode(instance, best_chromo, op_list)

    return FlexibleJobShopSolution(
        assignments=assignments,
        start_times=start_times,
        makespan=makespan,
    )


def _random_chromosome(
    instance: FlexibleJobShopInstance,
    op_list: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a random chromosome.

    Layout: [machine_assignments... | operation_permutation...]
    First total_ops entries: machine index for each operation.
    Next total_ops entries: permutation of operation indices.
    """
    total_ops = len(op_list)
    chromo = np.zeros(2 * total_ops, dtype=int)

    # Machine assignments
    for i, (j, k) in enumerate(op_list):
        eligible = list(instance.jobs[j][k].keys())
        chromo[i] = rng.choice(eligible)

    # Operation permutation (using job-based encoding)
    # Create a sequence using random job order
    perm = rng.permutation(total_ops)
    chromo[total_ops:] = perm

    return chromo


def _evaluate(
    instance: FlexibleJobShopInstance,
    chromo: np.ndarray,
    op_list: list[tuple[int, int]],
) -> int:
    """Evaluate chromosome fitness (makespan)."""
    _, _, makespan = _decode(instance, chromo, op_list)
    return makespan


def _decode(
    instance: FlexibleJobShopInstance,
    chromo: np.ndarray,
    op_list: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], int]:
    """
    Decode chromosome into a schedule.

    Returns:
        (assignments, start_times, makespan).
    """
    total_ops = len(op_list)
    assignments: dict[tuple[int, int], int] = {}
    start_times: dict[tuple[int, int], int] = {}

    # Extract machine assignments
    for i, (j, k) in enumerate(op_list):
        mach = int(chromo[i])
        # Validate machine is eligible
        if mach not in instance.jobs[j][k]:
            mach = min(instance.jobs[j][k].keys())
        assignments[(j, k)] = mach

    # Extract operation order
    perm = chromo[total_ops:].copy()

    # Decode using the permutation order
    machine_available = [0] * instance.m
    job_available = [0] * instance.n
    next_op = [0] * instance.n

    # Schedule operations in permutation order, but respecting precedence
    # Build a priority from the permutation
    op_priority = np.zeros(total_ops, dtype=int)
    for rank, idx in enumerate(perm):
        op_priority[idx] = rank

    # Sort operations by priority, but only schedule when precedence allows
    sorted_ops = sorted(range(total_ops), key=lambda i: op_priority[i])

    scheduled = set()
    remaining = list(sorted_ops)

    while remaining:
        progress = False
        for idx in list(remaining):
            j, k = op_list[idx]
            # Check precedence
            if k > 0 and (j, k - 1) not in scheduled:
                continue

            mach = assignments[(j, k)]
            pt = instance.jobs[j][k][mach]
            start = max(machine_available[mach], job_available[j])

            start_times[(j, k)] = start
            machine_available[mach] = start + pt
            job_available[j] = start + pt
            scheduled.add((j, k))
            remaining.remove(idx)
            progress = True

        if not progress:
            break

    makespan = 0
    for (j, k), s in start_times.items():
        mach = assignments[(j, k)]
        pt = instance.jobs[j][k][mach]
        makespan = max(makespan, s + pt)

    return assignments, start_times, makespan


def _tournament(
    population: list[np.ndarray],
    fitness: list[int],
    rng: np.random.Generator,
    size: int = 3,
) -> np.ndarray:
    """Tournament selection."""
    indices = rng.choice(len(population), size=size, replace=False)
    best = min(indices, key=lambda i: fitness[i])
    return population[best].copy()


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    total_ops: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover for machine assignment, order crossover for sequence."""
    c1 = p1.copy()
    c2 = p2.copy()

    # Uniform crossover on machine assignments
    mask = rng.random(total_ops) < 0.5
    c1[:total_ops] = np.where(mask, p1[:total_ops], p2[:total_ops])
    c2[:total_ops] = np.where(mask, p2[:total_ops], p1[:total_ops])

    # For operation permutation, use PMX-like crossover
    seq1 = p1[total_ops:].copy()
    seq2 = p2[total_ops:].copy()

    # Two-point crossover points
    pts = sorted(rng.choice(total_ops, size=2, replace=False))
    pt1, pt2 = pts[0], pts[1]

    # Simple: just swap segment
    c1_seq = _order_crossover(seq1, seq2, pt1, pt2, total_ops)
    c2_seq = _order_crossover(seq2, seq1, pt1, pt2, total_ops)

    c1[total_ops:] = c1_seq
    c2[total_ops:] = c2_seq

    return c1, c2


def _order_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    pt1: int,
    pt2: int,
    n: int,
) -> np.ndarray:
    """Order crossover (OX) for permutation part."""
    child = np.full(n, -1, dtype=int)
    # Copy segment from p1
    child[pt1:pt2] = p1[pt1:pt2]
    used = set(child[pt1:pt2])

    # Fill remaining from p2 in order
    pos = pt2
    for val in np.concatenate([p2[pt2:], p2[:pt2]]):
        if val not in used:
            child[pos % n] = val
            used.add(val)
            pos += 1

    return child


def _mutate(
    instance: FlexibleJobShopInstance,
    chromo: np.ndarray,
    op_list: list[tuple[int, int]],
    rng: np.random.Generator,
):
    """Mutate: reassign a random operation's machine or swap in sequence."""
    total_ops = len(op_list)

    if rng.random() < 0.5:
        # Machine reassignment
        idx = rng.integers(0, total_ops)
        j, k = op_list[idx]
        eligible = list(instance.jobs[j][k].keys())
        chromo[idx] = rng.choice(eligible)
    else:
        # Swap two positions in operation sequence
        i, j_swap = rng.choice(total_ops, size=2, replace=False)
        seq_start = total_ops
        chromo[seq_start + i], chromo[seq_start + j_swap] = (
            chromo[seq_start + j_swap], chromo[seq_start + i]
        )


if __name__ == "__main__":
    print("=== GA for Flexible Job Shop ===\n")

    inst = FlexibleJobShopInstance.random(n=5, m=3, flexibility=0.6, seed=42)
    sol = genetic_algorithm(inst, pop_size=30, generations=100, seed=42)
    valid, _ = validate_solution(inst, sol.assignments, sol.start_times)
    print(f"GA makespan: {sol.makespan}, valid: {valid}")

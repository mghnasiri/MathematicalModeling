"""
Genetic Algorithm (GA) — Population-Based Metaheuristic for Pm || Cmax

A steady-state genetic algorithm for the parallel machine makespan problem.
Uses an integer-vector encoding where each gene represents the machine
assignment of a job.

Algorithm:
    1. Initialize population: LPT solution + random assignments.
    2. Repeat until termination:
       a. SELECTION: Binary tournament.
       b. CROSSOVER: Uniform crossover — each gene inherited from a random parent.
       c. MUTATION: Reassign a random job to the least loaded machine.
       d. LOCAL SEARCH (optional): Move jobs from most to least loaded machine.
       e. REPLACEMENT: Replace worst if offspring is better.
    3. Return the best individual found.

Encoding: Integer vector of length n, where gene[j] = i means job j is
assigned to machine i.

Notation: Pm || Cmax (generalizes to Rm || Cmax)
Complexity: O(generations * population_size * n * m)
Reference: Cheng, R. & Gen, M. (1997). "Parallel Machine Scheduling Problems
           Using Memetic Algorithms"
           Computers & Industrial Engineering, 33(3-4):761-764.
           DOI: 10.1016/S0360-8352(97)00234-2
"""

from __future__ import annotations
import sys
import os
import time
import importlib.util
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))

_instance_path = os.path.join(_this_dir, "..", "instance.py")
_spec = importlib.util.spec_from_file_location("pm_instance", _instance_path)
_pm_instance = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("pm_instance", _pm_instance)
_spec.loader.exec_module(_pm_instance)

ParallelMachineInstance = _pm_instance.ParallelMachineInstance
ParallelMachineSolution = _pm_instance.ParallelMachineSolution
compute_makespan = _pm_instance.compute_makespan
compute_machine_loads = _pm_instance.compute_machine_loads

_lpt_path = os.path.join(_this_dir, "..", "heuristics", "lpt.py")
_spec2 = importlib.util.spec_from_file_location("pm_lpt", _lpt_path)
_pm_lpt = importlib.util.module_from_spec(_spec2)
sys.modules.setdefault("pm_lpt", _pm_lpt)
_spec2.loader.exec_module(_pm_lpt)

lpt = _pm_lpt.lpt


def genetic_algorithm(
    instance: ParallelMachineInstance,
    population_size: int = 30,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    time_limit: float | None = None,
    max_generations: int = 500,
    use_local_search: bool = False,
    seed: int | None = None,
) -> ParallelMachineSolution:
    """
    Apply a Genetic Algorithm to a parallel machine instance.

    Args:
        instance: A ParallelMachineInstance.
        population_size: Number of individuals.
        crossover_rate: Probability of crossover.
        mutation_rate: Probability of mutation per offspring.
        time_limit: Maximum runtime in seconds.
        max_generations: Maximum generations (if no time_limit).
        use_local_search: If True, apply load-balancing local search.
        seed: Random seed for reproducibility.

    Returns:
        ParallelMachineSolution with the best assignment found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    m = instance.m

    # Initialize population
    population = _initialize_population(instance, population_size, rng)
    fitness = [_evaluate(instance, ind) for ind in population]

    best_idx = int(np.argmin(fitness))
    best_ind = population[best_idx].copy()
    best_ms = fitness[best_idx]

    start_time = time.time()

    for gen in range(max_generations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # Selection
        p1 = _tournament_select(population, fitness, rng)
        p2 = _tournament_select(population, fitness, rng)

        # Crossover
        if rng.random() < crossover_rate:
            offspring = _uniform_crossover(p1, p2, rng)
        else:
            offspring = p1.copy()

        # Mutation
        if rng.random() < mutation_rate:
            offspring = _mutation(instance, offspring, rng)

        # Optional local search
        if use_local_search:
            offspring = _load_balance(instance, offspring)

        # Evaluate
        offspring_ms = _evaluate(instance, offspring)

        # Steady-state replacement
        worst_idx = int(np.argmax(fitness))
        if offspring_ms < fitness[worst_idx]:
            population[worst_idx] = offspring
            fitness[worst_idx] = offspring_ms

        if offspring_ms < best_ms:
            best_ind = offspring.copy()
            best_ms = offspring_ms

    # Convert best individual to assignment
    assignment = _encoding_to_assignment(best_ind, m)
    loads = compute_machine_loads(instance, assignment)
    return ParallelMachineSolution(
        assignment=assignment, makespan=best_ms, machine_loads=loads
    )


def _encoding_to_assignment(
    encoding: np.ndarray, m: int
) -> list[list[int]]:
    """Convert integer-vector encoding to assignment lists."""
    assignment: list[list[int]] = [[] for _ in range(m)]
    for j, machine in enumerate(encoding):
        assignment[int(machine)].append(j)
    return assignment


def _evaluate(
    instance: ParallelMachineInstance,
    encoding: np.ndarray,
) -> float:
    """Evaluate makespan of an encoding."""
    assignment = _encoding_to_assignment(encoding, instance.m)
    return compute_makespan(instance, assignment)


def _initialize_population(
    instance: ParallelMachineInstance,
    size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create initial population with LPT seed and random perturbations."""
    n = instance.n
    m = instance.m
    population = []

    # First individual: LPT solution
    lpt_sol = lpt(instance)
    lpt_encoding = np.zeros(n, dtype=int)
    for i, jobs in enumerate(lpt_sol.assignment):
        for j in jobs:
            lpt_encoding[j] = i
    population.append(lpt_encoding)

    # Remaining: random perturbations of LPT
    for _ in range(size - 1):
        enc = lpt_encoding.copy()
        # Randomly reassign some jobs
        n_changes = max(1, n // 4)
        for _ in range(n_changes):
            j = rng.integers(0, n)
            enc[j] = rng.integers(0, m)
        population.append(enc)

    return population


def _tournament_select(
    population: list[np.ndarray],
    fitness: list[float],
    rng: np.random.Generator,
    k: int = 2,
) -> np.ndarray:
    """Binary tournament selection."""
    candidates = rng.choice(len(population), size=k, replace=False)
    winner = min(candidates, key=lambda idx: fitness[idx])
    return population[winner].copy()


def _uniform_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Uniform crossover: each gene from a random parent."""
    mask = rng.random(len(p1)) < 0.5
    offspring = np.where(mask, p1, p2)
    return offspring


def _mutation(
    instance: ParallelMachineInstance,
    encoding: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mutation: reassign a random job to the least loaded machine.
    """
    enc = encoding.copy()
    m = instance.m

    # Find loads
    loads = np.zeros(m)
    for j, machine in enumerate(enc):
        loads[int(machine)] += instance.get_processing_time(j, int(machine))

    # Pick a random job and reassign to least loaded machine
    j = rng.integers(0, len(enc))
    least_loaded = int(np.argmin(loads))
    enc[j] = least_loaded

    return enc


def _load_balance(
    instance: ParallelMachineInstance,
    encoding: np.ndarray,
) -> np.ndarray:
    """
    Load-balancing local search: move one job from the most loaded
    machine to the least loaded machine if it improves the makespan.
    """
    enc = encoding.copy()
    m = instance.m
    improved = True

    for _ in range(len(enc)):  # limit iterations
        if not improved:
            break
        improved = False

        # Compute loads
        assignment = _encoding_to_assignment(enc, m)
        loads = []
        for i, jobs in enumerate(assignment):
            load = sum(instance.get_processing_time(j, i) for j in jobs)
            loads.append(load)

        current_ms = max(loads)
        most_loaded = int(np.argmax(loads))
        least_loaded = int(np.argmin(loads))

        if most_loaded == least_loaded:
            break

        # Try moving each job from most to least loaded
        for j in assignment[most_loaded]:
            pt_from = instance.get_processing_time(j, most_loaded)
            pt_to = instance.get_processing_time(j, least_loaded)
            new_most = loads[most_loaded] - pt_from
            new_least = loads[least_loaded] + pt_to
            new_ms = max(new_most, new_least,
                         *(loads[k] for k in range(m)
                           if k != most_loaded and k != least_loaded))

            if new_ms < current_ms:
                enc[j] = least_loaded
                improved = True
                break

    return enc


if __name__ == "__main__":
    print("=" * 60)
    print("Genetic Algorithm — Parallel Machine Scheduling")
    print("=" * 60)

    instance = ParallelMachineInstance.random_identical(n=20, m=3, seed=42)

    sol_lpt = lpt(instance)
    print(f"\nLPT  Makespan: {sol_lpt.makespan:.0f}")

    sol_ga = genetic_algorithm(instance, max_generations=300, seed=42)
    print(f"GA   Makespan: {sol_ga.makespan:.0f}")

    sol_mga = genetic_algorithm(
        instance, max_generations=100, use_local_search=True, seed=42
    )
    print(f"MGA  Makespan: {sol_mga.makespan:.0f}")

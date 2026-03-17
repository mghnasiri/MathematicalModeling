"""
Genetic Algorithm (GA) — Population-Based Metaheuristic for Fm | prmu | Cmax

A steady-state genetic algorithm for the permutation flow shop scheduling
problem. Maintains a population of permutations that evolves through
selection, crossover, and mutation operators designed specifically for
permutation representations.

Algorithm:
    1. Initialize population: NEH solution + random perturbations.
    2. Repeat until termination:
       a. SELECTION: Binary tournament — pick two parents from population.
       b. CROSSOVER: Order Crossover (OX) — preserves relative ordering
          from one parent while inheriting a segment from the other.
       c. MUTATION: Insertion mutation — remove a random job and reinsert
          at a random position (aligned with the best PFSP neighborhood).
       d. LOCAL SEARCH (optional): Apply insertion-based local search to
          offspring for memetic behavior.
       e. REPLACEMENT: Replace worst individual if offspring is better
          than the worst in the population (steady-state).
    3. Return the best individual found.

Crossover operator — Order Crossover (OX):
    Given parents P1 and P2, select a random segment from P1. Place it in
    the same positions in the offspring. Fill remaining positions with jobs
    from P2 in their relative order. This preserves adjacency information
    from P1 and ordering information from P2.

    Reference: Davis, L. (1985). "Applying Adaptive Algorithms to Epistatic
               Domains" Proc. IJCAI, pp. 162-164.

Notation: Fm | prmu | Cmax
Reference: Reeves, C.R. (1995). "A Genetic Algorithm for Flowshop Sequencing"
           Computers & Operations Research, 22(1):5-13.
           DOI: 10.1016/0305-0548(93)E0014-K

           Ruiz, R., Maroto, C. & Alcaraz, J. (2006). "Two New Robust Genetic
           Algorithms for the Flowshop Scheduling Problem"
           Omega, 34(5):461-476.
           DOI: 10.1016/j.omega.2004.12.006
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instance import FlowShopInstance, FlowShopSolution, compute_makespan
from heuristics.neh import neh


def genetic_algorithm(
    instance: FlowShopInstance,
    population_size: int = 20,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    time_limit: float | None = None,
    max_generations: int = 500,
    use_local_search: bool = False,
    seed: int | None = None,
) -> FlowShopSolution:
    """
    Apply a Genetic Algorithm to a permutation flow shop instance.

    Args:
        instance: A FlowShopInstance.
        population_size: Number of individuals in the population.
        crossover_rate: Probability of applying crossover (vs. copying parent).
        mutation_rate: Probability of mutation per offspring.
        time_limit: Maximum runtime in seconds. If None, uses max_generations.
        max_generations: Maximum number of generations (if no time_limit).
        use_local_search: If True, apply insertion local search to each
            offspring (memetic GA). Significantly slower but higher quality.
        seed: Random seed for reproducibility.

    Returns:
        FlowShopSolution with the best permutation found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # --- Population Initialization ---
    population = _initialize_population(instance, population_size, rng)

    # Evaluate fitness (makespan, lower is better)
    fitness = [compute_makespan(instance, ind) for ind in population]

    best_idx = int(np.argmin(fitness))
    best_perm = list(population[best_idx])
    best_ms = fitness[best_idx]

    start_time = time.time()

    for generation in range(max_generations):
        # Check time limit
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        # --- Selection: Binary Tournament ---
        parent1 = _tournament_select(population, fitness, rng)
        parent2 = _tournament_select(population, fitness, rng)

        # --- Crossover: Order Crossover (OX) ---
        if rng.random() < crossover_rate:
            offspring = _order_crossover(parent1, parent2, rng)
        else:
            offspring = list(parent1)

        # --- Mutation: Insertion ---
        if rng.random() < mutation_rate:
            offspring = _insertion_mutation(offspring, rng)

        # --- Optional Local Search (memetic) ---
        if use_local_search:
            offspring = _local_search_insert(instance, offspring)

        # --- Evaluate offspring ---
        offspring_ms = compute_makespan(instance, offspring)

        # --- Steady-state replacement: replace worst if offspring is better ---
        worst_idx = int(np.argmax(fitness))
        if offspring_ms < fitness[worst_idx]:
            population[worst_idx] = offspring
            fitness[worst_idx] = offspring_ms

        # Update best
        if offspring_ms < best_ms:
            best_perm = list(offspring)
            best_ms = offspring_ms

    return FlowShopSolution(permutation=best_perm, makespan=best_ms)


def _initialize_population(
    instance: FlowShopInstance,
    size: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Create initial population with NEH seed and random perturbations.

    The first individual is the NEH solution (high quality). Remaining
    individuals are generated by randomly perturbing the NEH solution
    with a series of random insertion moves.

    Args:
        instance: A FlowShopInstance.
        size: Population size.
        rng: Random number generator.

    Returns:
        List of permutations (each a list of job indices).
    """
    n = instance.n
    population: list[list[int]] = []

    # First individual: NEH solution
    neh_sol = neh(instance)
    population.append(list(neh_sol.permutation))

    # Remaining: perturbed versions of NEH
    for _ in range(size - 1):
        perm = list(neh_sol.permutation)
        # Apply n/2 random insertion moves to diversify
        n_perturbations = max(2, n // 2)
        for _ in range(n_perturbations):
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            if j >= i:
                j += 1
            job = perm.pop(i)
            insert_pos = min(j, len(perm))
            perm.insert(insert_pos, job)
        population.append(perm)

    return population


def _tournament_select(
    population: list[list[int]],
    fitness: list[int],
    rng: np.random.Generator,
    tournament_size: int = 2,
) -> list[int]:
    """
    Binary tournament selection.

    Randomly pick tournament_size individuals and return the one with
    the lowest fitness (best makespan).

    Args:
        population: List of permutations.
        fitness: Makespan for each individual.
        rng: Random number generator.
        tournament_size: Number of candidates in each tournament.

    Returns:
        A copy of the selected individual.
    """
    candidates = rng.choice(len(population), size=tournament_size, replace=False)
    winner = min(candidates, key=lambda idx: fitness[idx])
    return list(population[winner])


def _order_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """
    Order Crossover (OX) for permutation representations.

    Selects a random segment from parent1 and preserves it in the offspring.
    Fills remaining positions with elements from parent2 in their relative
    order, wrapping around from the second cut point.

    Args:
        parent1: First parent permutation.
        parent2: Second parent permutation.
        rng: Random number generator.

    Returns:
        Offspring permutation.
    """
    n = len(parent1)
    if n <= 2:
        return list(parent1)

    # Select two random cut points
    cut1, cut2 = sorted(rng.choice(n, size=2, replace=False))

    # Copy segment from parent1
    offspring = [None] * n
    segment = set()
    for i in range(cut1, cut2 + 1):
        offspring[i] = parent1[i]
        segment.add(parent1[i])

    # Fill remaining positions from parent2 in order, starting after cut2
    p2_filtered = [j for j in parent2 if j not in segment]
    fill_pos = [(cut2 + 1 + i) % n for i in range(n - len(segment))]
    fill_pos.sort()

    for pos, job in zip(fill_pos, p2_filtered):
        offspring[pos] = job

    return offspring


def _insertion_mutation(
    permutation: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """
    Insertion mutation: remove a random job and reinsert at a random position.

    This is aligned with the insertion neighborhood, which is the most
    effective neighborhood for PFSP.

    Args:
        permutation: The permutation to mutate.
        rng: Random number generator.

    Returns:
        Mutated permutation.
    """
    n = len(permutation)
    perm = list(permutation)

    i = rng.integers(0, n)
    job = perm.pop(i)
    j = rng.integers(0, len(perm) + 1)
    perm.insert(j, job)

    return perm


def _local_search_insert(
    instance: FlowShopInstance,
    permutation: list[int],
) -> list[int]:
    """
    First-improvement insertion local search.

    For each job, try removing and reinserting at every other position.
    Accept the first improving move and restart.

    Args:
        instance: A FlowShopInstance.
        permutation: Starting permutation.

    Returns:
        Locally optimal permutation.
    """
    perm = list(permutation)
    current_ms = compute_makespan(instance, perm)
    improved = True

    while improved:
        improved = False
        for i in range(len(perm)):
            job = perm[i]
            remaining = perm[:i] + perm[i + 1:]

            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = compute_makespan(instance, candidate)
                if ms < current_ms:
                    perm = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break

    return perm


if __name__ == "__main__":
    print("=" * 60)
    print("Genetic Algorithm — Permutation Flow Shop")
    print("=" * 60)

    instance = FlowShopInstance.random(n=20, m=5, seed=42)

    from heuristics.cds import cds
    sol_cds = cds(instance)
    sol_neh = neh(instance)
    print(f"\nCDS  Makespan:  {sol_cds.makespan}")
    print(f"NEH  Makespan:  {sol_neh.makespan}")

    # Standard GA
    sol_ga = genetic_algorithm(
        instance, max_generations=500, seed=42
    )
    print(f"GA   Makespan:  {sol_ga.makespan}")

    # Memetic GA (with local search)
    sol_mga = genetic_algorithm(
        instance, max_generations=100, use_local_search=True, seed=42
    )
    print(f"MGA  Makespan:  {sol_mga.makespan}")

    # Compare with IG
    from metaheuristics.iterated_greedy import iterated_greedy
    sol_ig = iterated_greedy(instance, max_iterations=500, seed=42)
    print(f"IG   Makespan:  {sol_ig.makespan}")

    # Larger instance
    print("\n" + "=" * 60)
    print("Larger Instance: 50x10")
    print("=" * 60)

    large_instance = FlowShopInstance.random(n=50, m=10, seed=123)
    sol_neh_lg = neh(large_instance)
    print(f"NEH Makespan:   {sol_neh_lg.makespan}")

    t0 = time.time()
    sol_ga_lg = genetic_algorithm(
        large_instance, time_limit=2.0, seed=42
    )
    elapsed = time.time() - t0
    print(f"GA  Makespan:   {sol_ga_lg.makespan}  ({elapsed:.1f}s)")

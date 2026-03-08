"""
Benchmark Runner — Evaluate All PFSP Algorithms Against Taillard Instances

Downloads Taillard benchmark instances and evaluates every implemented
algorithm, reporting:
    - Makespan per instance
    - Average Relative Percentage Deviation (ARPD) from best known
    - Runtime per instance

ARPD is the standard metric in flow shop literature:
    RPD_i = 100 × (Heuristic_i - BKS_i) / BKS_i
    ARPD  = mean(RPD_i) over all instances in a class

Usage:
    python benchmark_runner.py                    # Run on tai20_5 (quick)
    python benchmark_runner.py --class 50_10      # Run on tai50_10
    python benchmark_runner.py --all              # Run all small classes
    python benchmark_runner.py --ig-time 1.0      # IG with 1s time limit
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..'))

from instance import FlowShopInstance, compute_makespan
from exact.johnsons_rule import johnsons_rule
from heuristics.palmers_slope import palmers_slope
from heuristics.cds import cds
from heuristics.neh import neh, neh_with_tiebreaking
from metaheuristics.iterated_greedy import iterated_greedy
from shared.parsers.taillard_parser import (
    load_taillard_instance,
    BEST_KNOWN_UPPER_BOUNDS,
)


def run_benchmark(
    n_jobs: int,
    n_machines: int,
    ig_time_limit: float = 0.5,
    ig_seed: int = 42,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Run all algorithms on a Taillard instance class.

    Args:
        n_jobs: Number of jobs in the class.
        n_machines: Number of machines in the class.
        ig_time_limit: Time limit in seconds for Iterated Greedy.
        ig_seed: Random seed for IG.
        verbose: Print per-instance results.

    Returns:
        Dict mapping algorithm name → list of RPD values.
    """
    # Define algorithms to benchmark
    algorithms: dict[str, object] = {}

    # Only include Johnson's Rule for 2-machine instances
    if n_machines == 2:
        algorithms["Johnson"] = lambda inst: johnsons_rule(inst)

    algorithms["Palmer"] = lambda inst: palmers_slope(inst)
    algorithms["CDS"] = lambda inst: cds(inst)
    algorithms["NEH"] = lambda inst: neh(inst)
    algorithms["NEH-TB"] = lambda inst: neh_with_tiebreaking(inst)
    algorithms["IG"] = lambda inst: iterated_greedy(
        inst, time_limit=ig_time_limit, seed=ig_seed
    )

    results: dict[str, list[float]] = {name: [] for name in algorithms}
    runtimes: dict[str, list[float]] = {name: [] for name in algorithms}

    if verbose:
        # Header
        alg_names = list(algorithms.keys())
        header = f"{'Instance':<14}"
        header += f"{'BKS':>6}"
        for name in alg_names:
            header += f"  {name:>8}"
        print(header)
        print("-" * len(header))

    for idx in range(10):
        instance_name = f"tai{n_jobs}_{n_machines}_{idx}"

        # Load instance
        try:
            p, info = load_taillard_instance(instance_name)
        except ConnectionError as e:
            print(f"  SKIP {instance_name}: {e}")
            continue

        instance = FlowShopInstance(
            n=info.n_jobs, m=info.n_machines, processing_times=p
        )
        bks = info.upper_bound

        if verbose:
            row = f"{instance_name:<14}{bks:>6}"

        for alg_name, alg_func in algorithms.items():
            t0 = time.time()
            sol = alg_func(instance)
            elapsed = time.time() - t0

            rpd = 100.0 * (sol.makespan - bks) / bks
            results[alg_name].append(rpd)
            runtimes[alg_name].append(elapsed)

            if verbose:
                row += f"  {sol.makespan:>5}({rpd:+.1f}%)"

        if verbose:
            print(row.replace("(+", " (+").replace("(-", " (-"))

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print(f"ARPD Summary — tai{n_jobs}_{n_machines} (10 instances)")
        print("=" * 60)

        summary_header = f"{'Algorithm':<12} {'ARPD':>8} {'Best RPD':>10} {'Worst RPD':>10} {'Avg Time':>10}"
        print(summary_header)
        print("-" * len(summary_header))

        for alg_name in algorithms:
            rpds = results[alg_name]
            times = runtimes[alg_name]
            if rpds:
                arpd = np.mean(rpds)
                best = np.min(rpds)
                worst = np.max(rpds)
                avg_time = np.mean(times)
                print(f"{alg_name:<12} {arpd:>7.2f}% {best:>9.2f}% {worst:>9.2f}% {avg_time:>9.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PFSP algorithms against Taillard instances"
    )
    parser.add_argument(
        "--class", dest="inst_class", type=str, default="20_5",
        help="Instance class as 'jobs_machines', e.g. '20_5', '50_10'"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all small instance classes (20×5 through 50×10)"
    )
    parser.add_argument(
        "--ig-time", type=float, default=0.5,
        help="Time limit in seconds for Iterated Greedy (default: 0.5)"
    )
    parser.add_argument(
        "--ig-seed", type=int, default=42,
        help="Random seed for Iterated Greedy (default: 42)"
    )
    args = parser.parse_args()

    if args.all:
        classes = [(20, 5), (20, 10), (20, 20), (50, 5), (50, 10)]
        all_results: dict[str, list[float]] = {}

        for n_jobs, n_machines in classes:
            print(f"\n{'='*60}")
            print(f"  Taillard Class: {n_jobs} jobs × {n_machines} machines")
            print(f"{'='*60}\n")

            results = run_benchmark(
                n_jobs, n_machines,
                ig_time_limit=args.ig_time,
                ig_seed=args.ig_seed,
            )

            for alg_name, rpds in results.items():
                if alg_name not in all_results:
                    all_results[alg_name] = []
                all_results[alg_name].extend(rpds)

        # Grand summary
        print(f"\n{'='*60}")
        print("  GRAND ARPD — All Classes Combined")
        print(f"{'='*60}")
        for alg_name, rpds in all_results.items():
            if rpds:
                print(f"  {alg_name:<12} {np.mean(rpds):>7.2f}%")

    else:
        parts = args.inst_class.split("_")
        n_jobs, n_machines = int(parts[0]), int(parts[1])
        run_benchmark(
            n_jobs, n_machines,
            ig_time_limit=args.ig_time,
            ig_seed=args.ig_seed,
        )


if __name__ == "__main__":
    main()

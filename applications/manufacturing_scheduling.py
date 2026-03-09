"""
Real-World Application: Manufacturing Production Scheduling.

Domain: Automotive paint shop / Pharmaceutical tablet production
Model: Permutation Flow Shop with Sequence-Dependent Setup Times (Fm | prmu, Ssd | Cmax)

Scenario:
    A pharmaceutical company produces 12 tablet formulations on a 4-stage
    production line: granulation → compression → coating → packaging.
    Changeover times between formulations depend on color, coating type,
    and allergen requirements (e.g., switching from a dark to light tablet
    requires a full line purge; same-color transitions are faster).

    Objective: Minimize makespan to maximize daily throughput.

Real-world considerations modeled:
    - Sequence-dependent setup times (color/allergen changeovers)
    - Processing time variation by product complexity
    - Line balancing across 4 production stages

Industry context:
    Setup time optimization in pharma can reduce changeover losses by
    15-30% (Allahverdi et al., 2008). The SDST flow shop model captures
    the key trade-off: grouping similar products reduces setups but may
    increase total processing time.

References:
    Allahverdi, A., Ng, C.T., Cheng, T.C.E. & Kovalyov, M.Y. (2008).
    A survey of scheduling problems with setup times or costs.
    European Journal of Operational Research, 187(3), 985-1032.
    https://doi.org/10.1016/j.ejor.2006.06.060

    Ruiz, R. & Stützle, T. (2008). An iterated greedy heuristic for the
    sequence dependent setup times flowshop problem with makespan and
    weighted tardiness objectives. European Journal of Operational
    Research, 187(3), 1143-1159.
    https://doi.org/10.1016/j.ejor.2006.07.029
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Domain Data ──────────────────────────────────────────────────────────────

# 12 tablet formulations
PRODUCTS = [
    "Aspirin 500mg (white)",
    "Ibuprofen 400mg (brown)",
    "Paracetamol 325mg (white)",
    "Vitamin D 1000IU (yellow)",
    "Omeprazole 20mg (pink)",
    "Metformin 500mg (white)",
    "Atorvastatin 10mg (white)",
    "Amoxicillin 250mg (red)",
    "Lisinopril 5mg (blue)",
    "Cetirizine 10mg (white)",
    "Ranitidine 150mg (green)",
    "Losartan 50mg (green)",
]

# 4 production stages
STAGES = ["Granulation", "Compression", "Coating", "Packaging"]

# Color groups (same color → faster changeover)
COLOR_GROUPS = {
    "white": [0, 2, 5, 6, 9],
    "brown": [1],
    "yellow": [3],
    "pink": [4],
    "red": [7],
    "blue": [8],
    "green": [10, 11],
}

# Allergen groups (allergen switch requires full purge)
ALLERGEN_FREE = {0, 2, 3, 5, 6, 9, 10, 11}
CONTAINS_ALLERGENS = {1, 4, 7, 8}


def _color_of(job: int) -> str:
    for color, jobs in COLOR_GROUPS.items():
        if job in jobs:
            return color
    return "unknown"


def create_pharma_instance() -> dict:
    """Create a pharmaceutical production scheduling instance.

    Returns:
        Dictionary with processing_times, setup_times, product names,
        and stage names.
    """
    rng = np.random.default_rng(42)
    n_jobs = len(PRODUCTS)
    n_machines = len(STAGES)

    # Processing times (minutes) vary by product complexity and stage
    # Granulation: 30-60 min, Compression: 15-40 min,
    # Coating: 20-50 min, Packaging: 10-30 min
    stage_ranges = [(30, 60), (15, 40), (20, 50), (10, 30)]

    processing_times = np.zeros((n_jobs, n_machines), dtype=int)
    for j in range(n_jobs):
        for m in range(n_machines):
            lo, hi = stage_ranges[m]
            processing_times[j][m] = rng.integers(lo, hi + 1)

    # Setup times: depend on color transition and allergen status
    # Setup times shape: (m, n+1, n) — row 0 is initial setup (from idle)
    setup_times = np.zeros((n_machines, n_jobs + 1, n_jobs), dtype=int)
    for m in range(n_machines):
        # Initial setup (from idle state to each job)
        stage_factor = [1.0, 0.8, 1.5, 0.6][m]
        for j in range(n_jobs):
            setup_times[m][0][j] = int(10 * stage_factor)

        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue

                base = 5  # minimum changeover

                # Same color → small additional setup
                if _color_of(i) == _color_of(j):
                    color_penalty = 0
                else:
                    # Dark to light requires full purge
                    dark = {"brown", "red", "green", "blue"}
                    if _color_of(i) in dark and _color_of(j) == "white":
                        color_penalty = 25  # full purge
                    else:
                        color_penalty = 12  # standard color change

                # Allergen transition penalty
                if i in CONTAINS_ALLERGENS and j in ALLERGEN_FREE:
                    allergen_penalty = 20  # allergen purge
                elif i in ALLERGEN_FREE and j in CONTAINS_ALLERGENS:
                    allergen_penalty = 5  # minor prep
                else:
                    allergen_penalty = 0

                # Row i+1 (offset by 1 for initial row)
                setup_times[m][i + 1][j] = int(
                    (base + color_penalty + allergen_penalty) * stage_factor
                )

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "processing_times": processing_times,
        "setup_times": setup_times,
        "products": PRODUCTS,
        "stages": STAGES,
    }


def solve_pharma_scheduling(verbose: bool = True) -> dict:
    """Solve the pharmaceutical scheduling problem.

    Uses the SDST flow shop variant with NEH-SDST and IG-SDST.

    Returns:
        Dictionary with results from each method.
    """
    data = create_pharma_instance()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdst_dir = os.path.join(
        base_dir, "problems", "scheduling", "flow_shop", "variants", "setup_times"
    )

    sdst_inst_mod = _load_mod(
        "sdst_inst_app", os.path.join(sdst_dir, "instance.py")
    )
    sdst_heur_mod = _load_mod(
        "sdst_heur_app", os.path.join(sdst_dir, "heuristics.py")
    )
    sdst_meta_mod = _load_mod(
        "sdst_meta_app", os.path.join(sdst_dir, "metaheuristics.py")
    )

    # SDST expects (m, n) shape for processing_times
    instance = sdst_inst_mod.SDSTFlowShopInstance(
        n=data["n_jobs"],
        m=data["n_machines"],
        processing_times=data["processing_times"].T,
        setup_times=data["setup_times"],
    )

    results = {}

    # NEH-SDST (constructive)
    neh_sol = sdst_heur_mod.neh_sdst(instance)
    results["NEH-SDST"] = {
        "makespan": neh_sol.makespan,
        "sequence": neh_sol.permutation,
    }

    # GRASP-SDST
    grasp_sol = sdst_heur_mod.grasp_sdst(instance, max_constructions=50, alpha=0.3, seed=42)
    results["GRASP-SDST"] = {
        "makespan": grasp_sol.makespan,
        "sequence": grasp_sol.permutation,
    }

    # Iterated Greedy for SDST
    ig_sol = sdst_meta_mod.iterated_greedy_sdst(
        instance, max_iterations=500, d=3, seed=42
    )
    results["IG-SDST"] = {
        "makespan": ig_sol.makespan,
        "sequence": ig_sol.permutation,
    }

    if verbose:
        print("=" * 70)
        print("PHARMACEUTICAL PRODUCTION SCHEDULING")
        print(f"  {data['n_jobs']} formulations × {data['n_machines']} stages")
        print(f"  Stages: {', '.join(data['stages'])}")
        print("=" * 70)

        for method, res in results.items():
            seq = res["sequence"]
            hours = res["makespan"] / 60
            print(f"\n{method}: makespan = {res['makespan']} min ({hours:.1f} hrs)")
            print("  Production order:")
            for rank, job_id in enumerate(seq):
                print(f"    {rank+1:2d}. {data['products'][job_id]}")

        # Analyze best solution setup patterns
        best_method = min(results, key=lambda k: results[k]["makespan"])
        best_seq = results[best_method]["sequence"]
        print(f"\n--- Best: {best_method} ---")
        print("  Color transitions:")
        for i in range(len(best_seq) - 1):
            j1, j2 = best_seq[i], best_seq[i + 1]
            c1, c2 = _color_of(j1), _color_of(j2)
            marker = "✓ same" if c1 == c2 else "✗ change"
            print(f"    {data['products'][j1][:20]:20s} → "
                  f"{data['products'][j2][:20]:20s}  [{c1}→{c2}] {marker}")

    return results


if __name__ == "__main__":
    solve_pharma_scheduling()

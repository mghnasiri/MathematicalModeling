"""
Real-World Application: Agricultural Pest Control Task Scheduling.

Domain: Pest control task scheduling during growing season
Model: Single Machine Scheduling (1 || ΣwjTj) — schedule pest treatment
       tasks on one specialized crop sprayer with urgency weights and
       treatment deadlines.

Scenario:
    A farm operates a single specialized crop sprayer that must treat
    10 different pest infestations across various fields during the growing
    season. Each treatment task has:
    - A spray duration (processing time in hours)
    - A deadline (hours from now before pest damage exceeds threshold)
    - An urgency weight reflecting crop value and pest severity

    The sprayer can only handle one treatment at a time. The goal is to
    schedule all treatments to minimize total weighted tardiness — late
    treatments cause proportionally more damage on higher-value crops.

Real-world considerations modeled:
    - Varying spray durations based on field size and pest type
    - Deadlines reflecting pest lifecycle and damage thresholds
    - Urgency weights combining crop value and infestation severity
    - Single specialized equipment constraint (one sprayer)
    - Trade-off between treating urgent pests first vs. quick treatments

Industry context:
    Delayed pest control reduces crop yield by 5-30% depending on pest
    type and crop growth stage. Aphid infestations can double every 2-3
    days, making timely treatment critical. Integrated Pest Management
    (IPM) programs schedule treatments within biological windows to
    maximize efficacy and minimize chemical use (Pedigo & Rice, 2009).

References:
    Pedigo, L.P. & Rice, M.E. (2009). Entomology and Pest Management.
    6th Edition, Pearson Prentice Hall.

    Higley, L.G. & Pedigo, L.P. (1996). Economic Thresholds for
    Integrated Pest Management. University of Nebraska Press.

    Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound
    algorithm for the total weighted tardiness problem. Operations
    Research, 33(2), 363-377.
    https://doi.org/10.1287/opre.33.2.363
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

# 10 pest treatment tasks across the farm
PEST_TASKS = [
    {"name": "Aphid spray — Wheat Field A",     "pest": "aphid",
     "crop": "wheat",     "field_ha": 12, "severity": "high"},
    {"name": "Corn borer treatment — Corn B",    "pest": "corn_borer",
     "crop": "corn",      "field_ha": 8,  "severity": "critical"},
    {"name": "Spider mite spray — Soybean C",    "pest": "spider_mite",
     "crop": "soybean",   "field_ha": 15, "severity": "moderate"},
    {"name": "Cutworm treatment — Corn D",       "pest": "cutworm",
     "crop": "corn",      "field_ha": 10, "severity": "high"},
    {"name": "Whitefly spray — Tomato E",        "pest": "whitefly",
     "crop": "tomato",    "field_ha": 3,  "severity": "critical"},
    {"name": "Grasshopper spray — Alfalfa F",    "pest": "grasshopper",
     "crop": "alfalfa",   "field_ha": 20, "severity": "moderate"},
    {"name": "Flea beetle treatment — Canola G", "pest": "flea_beetle",
     "crop": "canola",    "field_ha": 14, "severity": "low"},
    {"name": "Armyworm spray — Wheat H",         "pest": "armyworm",
     "crop": "wheat",     "field_ha": 11, "severity": "high"},
    {"name": "Thrips spray — Onion I",           "pest": "thrips",
     "crop": "onion",     "field_ha": 5,  "severity": "critical"},
    {"name": "Leaf miner treatment — Potato J",  "pest": "leaf_miner",
     "crop": "potato",    "field_ha": 7,  "severity": "moderate"},
]

# Crop value per hectare ($/ha) — determines economic weight
CROP_VALUES = {
    "wheat": 800,
    "corn": 1000,
    "soybean": 900,
    "tomato": 5000,
    "alfalfa": 600,
    "canola": 750,
    "onion": 4000,
    "potato": 2500,
}

# Severity multipliers for urgency weight
SEVERITY_MULTIPLIERS = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 5,
}

# Spray rate: hours per hectare (depends on pest type)
SPRAY_RATES = {
    "aphid": 0.4,
    "corn_borer": 0.5,
    "spider_mite": 0.35,
    "cutworm": 0.45,
    "whitefly": 0.6,
    "grasshopper": 0.3,
    "flea_beetle": 0.35,
    "armyworm": 0.4,
    "thrips": 0.5,
    "leaf_miner": 0.45,
}

# Deadline in hours from now (before pest damage exceeds economic threshold)
DEADLINES_HOURS = [36, 18, 48, 30, 12, 60, 72, 24, 15, 42]


def create_pest_control_instance() -> dict:
    """Create a pest control scheduling instance.

    Returns:
        Dictionary with instance data and metadata.
    """
    n = len(PEST_TASKS)

    # Processing times: field_ha * spray_rate (rounded to integer hours)
    processing_times = np.array([
        max(1, int(round(task["field_ha"] * SPRAY_RATES[task["pest"]])))
        for task in PEST_TASKS
    ])

    # Urgency weights: (crop_value / 1000) * severity_multiplier
    weights = np.array([
        max(1, int(round(
            CROP_VALUES[task["crop"]] / 1000.0
            * SEVERITY_MULTIPLIERS[task["severity"]]
        )))
        for task in PEST_TASKS
    ])

    due_dates = np.array(DEADLINES_HOURS)

    return {
        "n": n,
        "processing_times": processing_times,
        "weights": weights,
        "due_dates": due_dates,
        "tasks": PEST_TASKS,
    }


def solve_pest_control(verbose: bool = True) -> dict:
    """Solve the pest control scheduling problem.

    Compares EDD, WSPT, and ATC dispatching rules for scheduling
    pest treatments on a single crop sprayer.

    Returns:
        Dictionary with results from each method.
    """
    data = create_pest_control_instance()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sm_dir = os.path.join(base_dir, "problems", "scheduling", "single_machine")

    sm_inst_mod = _load_mod(
        "sm_inst_pest", os.path.join(sm_dir, "instance.py")
    )
    sm_disp_mod = _load_mod(
        "sm_disp_pest",
        os.path.join(sm_dir, "heuristics", "dispatching_rules.py"),
    )
    sm_atc_mod = _load_mod(
        "sm_atc_pest",
        os.path.join(sm_dir, "heuristics", "apparent_tardiness_cost.py"),
    )

    instance = sm_inst_mod.SingleMachineInstance.from_arrays(
        processing_times=data["processing_times"],
        weights=data["weights"],
        due_dates=data["due_dates"],
    )

    results = {}

    # EDD — Earliest Due Date (optimal for Lmax, heuristic for ΣwjTj)
    edd_sol = sm_disp_mod.edd(instance)
    edd_wt = sm_inst_mod.compute_weighted_tardiness(instance, edd_sol.sequence)
    results["EDD"] = {
        "sequence": edd_sol.sequence,
        "weighted_tardiness": edd_wt,
    }

    # WSPT — Weighted Shortest Processing Time (optimal for ΣwjCj)
    wspt_sol = sm_disp_mod.wspt(instance)
    wspt_wt = sm_inst_mod.compute_weighted_tardiness(instance, wspt_sol.sequence)
    results["WSPT"] = {
        "sequence": wspt_sol.sequence,
        "weighted_tardiness": wspt_wt,
    }

    # ATC — Apparent Tardiness Cost (best dispatching rule for ΣwjTj)
    for k_val in [1.0, 2.0, 3.0]:
        atc_sol = sm_atc_mod.atc(instance, K=k_val)
        results[f"ATC(K={k_val})"] = {
            "sequence": atc_sol.sequence,
            "weighted_tardiness": atc_sol.objective_value,
        }

    if verbose:
        print("=" * 70)
        print("AGRICULTURAL PEST CONTROL SCHEDULING")
        print(f"  {data['n']} pest treatment tasks, 1 crop sprayer")
        print(f"  Total spray time: {data['processing_times'].sum()} hours")
        print("=" * 70)

        print("\n  Task details:")
        for i, task in enumerate(data["tasks"]):
            print(f"    {i:2d}. {task['name']}")
            print(f"        Duration: {data['processing_times'][i]}h, "
                  f"Deadline: {data['due_dates'][i]}h, "
                  f"Weight: {data['weights'][i]}")

        best_method = min(results, key=lambda k: results[k]["weighted_tardiness"])

        for method, res in results.items():
            marker = " ** BEST **" if method == best_method else ""
            print(f"\n  {method}: weighted tardiness = {res['weighted_tardiness']}{marker}")
            seq = res["sequence"]
            current_time = 0
            for rank, job_id in enumerate(seq):
                p = data["processing_times"][job_id]
                current_time += p
                d = data["due_dates"][job_id]
                tardy = max(0, current_time - d)
                status = f"LATE by {tardy}h" if tardy > 0 else "on time"
                print(f"    {rank+1:2d}. [{job_id:2d}] {data['tasks'][job_id]['name'][:40]:40s} "
                      f"done@{current_time:3d}h (due {d:3d}h) {status}")

        print(f"\n  Best method: {best_method} "
              f"(ΣwjTj = {results[best_method]['weighted_tardiness']})")

        # Yield loss analysis
        best_seq = results[best_method]["sequence"]
        current_time = 0
        total_loss = 0.0
        print(f"\n  Estimated yield loss (best schedule):")
        for job_id in best_seq:
            task = data["tasks"][job_id]
            p = data["processing_times"][job_id]
            current_time += p
            d = data["due_dates"][job_id]
            tardy = max(0, current_time - d)
            if tardy > 0:
                # Estimate yield loss: 2% per hour late, capped at 30%
                loss_pct = min(30.0, tardy * 2.0)
                field_value = CROP_VALUES[task["crop"]] * task["field_ha"]
                loss = field_value * loss_pct / 100.0
                total_loss += loss
                print(f"    {task['name'][:45]:45s} {loss_pct:.0f}% loss = ${loss:,.0f}")
        print(f"    Total estimated yield loss: ${total_loss:,.0f}")

    return results


if __name__ == "__main__":
    solve_pest_control()

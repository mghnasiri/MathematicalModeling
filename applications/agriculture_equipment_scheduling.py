"""
Real-World Application: Farm Equipment Scheduling During Planting Season.

Domain: Agricultural mechanization / Precision planting operations
Model: Parallel Machine Scheduling (Pm || Cmax)

Scenario:
    A large-scale grain farm operates 4 identical tractors (each with
    planting attachments) during the spring planting window. There are
    15 planting tasks across different fields, each with a duration
    determined by field size, crop type, and soil preparation needs.

    The planting window is weather-constrained: typically 2-3 weeks
    of suitable soil conditions. Finishing all planting tasks as early
    as possible (minimizing makespan) ensures crops are in the ground
    before the window closes. Each day of delay can reduce yield by
    approximately 1% due to shortened growing season.

    The 4 tractors are modeled as identical parallel machines. Each
    planting task is a non-preemptive job with known duration. The
    goal is to assign tasks to tractors to minimize the completion
    time of the last task (makespan).

Real-world considerations modeled:
    - Variable task durations (field size, crop type, tillage needs)
    - Non-preemptive scheduling (cannot interrupt mid-field)
    - Identical equipment capability (interchangeable tractors)
    - Makespan minimization (race against weather window)
    - Workload balancing across equipment

Industry context:
    The planting window for corn in the US Midwest is typically
    April 15 - May 15, with each day of delay after May 1 reducing
    yield by 1-2% (Nafziger, 2008). Efficient equipment scheduling
    can save 2-4 days of planting time on a large operation,
    translating to $15,000-$50,000 in yield preservation. Equipment
    costs average $150-250 per operating hour for modern planters.

References:
    Nafziger, E. (2008). Illinois Agronomy Handbook. University of
    Illinois Extension. Crop Sciences Special Publication 27.

    Bochtis, D.D. & Sorensen, C.G. (2009). The vehicle routing
    problem in field logistics part I. Biosystems Engineering,
    104(4), 447-457.
    https://doi.org/10.1016/j.biosystemseng.2009.09.003

    Graham, R.L. (1969). Bounds on multiprocessing timing anomalies.
    SIAM Journal on Applied Mathematics, 17(2), 416-429.
    https://doi.org/10.1137/0117039
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

N_TRACTORS = 4  # identical planting units

# 15 planting tasks with realistic durations (in hours)
# Duration depends on: field acreage, crop row spacing, soil prep level
PLANTING_TASKS = [
    {"name": "North Ridge - Corn",          "field": "North Ridge",     "crop": "Corn",
     "acres": 160, "duration": 14, "priority": "high"},
    {"name": "North Ridge - Soybean",       "field": "North Ridge",     "crop": "Soybean",
     "acres": 160, "duration": 12, "priority": "high"},
    {"name": "River Bottom - Corn",         "field": "River Bottom",    "crop": "Corn",
     "acres": 280, "duration": 24, "priority": "high"},
    {"name": "West Prairie - Wheat",        "field": "West Prairie",    "crop": "Wheat",
     "acres": 250, "duration": 18, "priority": "medium"},
    {"name": "Hilltop East - Corn",         "field": "Hilltop East",    "crop": "Corn",
     "acres": 200, "duration": 17, "priority": "high"},
    {"name": "South Meadow - Soybean",      "field": "South Meadow",   "crop": "Soybean",
     "acres": 300, "duration": 22, "priority": "medium"},
    {"name": "Creek Bend - Canola",         "field": "Creek Bend",     "crop": "Canola",
     "acres": 180, "duration": 15, "priority": "medium"},
    {"name": "Sandy Flats - Sunflower",     "field": "Sandy Flats",    "crop": "Sunflower",
     "acres": 220, "duration": 16, "priority": "low"},
    {"name": "Central Valley - Corn",       "field": "Central Valley",  "crop": "Corn",
     "acres": 250, "duration": 21, "priority": "high"},
    {"name": "South Meadow - Wheat",        "field": "South Meadow",   "crop": "Wheat",
     "acres": 150, "duration": 11, "priority": "medium"},
    {"name": "West Prairie - Barley",       "field": "West Prairie",    "crop": "Barley",
     "acres": 130, "duration": 10, "priority": "low"},
    {"name": "Hilltop East - Alfalfa",      "field": "Hilltop East",   "crop": "Alfalfa",
     "acres": 100, "duration": 8,  "priority": "low"},
    {"name": "Creek Bend - Oats",           "field": "Creek Bend",     "crop": "Oats",
     "acres": 90,  "duration": 7,  "priority": "low"},
    {"name": "Central Valley - Soybean",    "field": "Central Valley",  "crop": "Soybean",
     "acres": 120, "duration": 9,  "priority": "medium"},
    {"name": "Sandy Flats - Barley",        "field": "Sandy Flats",    "crop": "Barley",
     "acres": 110, "duration": 8,  "priority": "low"},
]

# Cost parameters
TRACTOR_COST_PER_HOUR = 185.0   # fuel + operator + depreciation
YIELD_LOSS_PER_DAY = 0.015      # 1.5% yield loss per day of delay
AVG_REVENUE_PER_ACRE = 750.0    # average gross revenue per acre


def create_equipment_instance() -> dict:
    """Create a parallel machine instance for tractor scheduling.

    Returns:
        Dictionary with instance data.
    """
    n = len(PLANTING_TASKS)
    processing_times = np.array(
        [t["duration"] for t in PLANTING_TASKS], dtype=float
    )

    total_acres = sum(t["acres"] for t in PLANTING_TASKS)
    total_hours = processing_times.sum()

    return {
        "n_tasks": n,
        "n_tractors": N_TRACTORS,
        "processing_times": processing_times,
        "tasks": PLANTING_TASKS,
        "total_hours": total_hours,
        "total_acres": total_acres,
    }


def solve_equipment_scheduling(verbose: bool = True) -> dict:
    """Solve tractor scheduling using parallel machine algorithms.

    Args:
        verbose: Whether to print detailed results.

    Returns:
        Dictionary with scheduling results from multiple methods.
    """
    data = create_equipment_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pm_dir = os.path.join(
        base_dir, "problems", "scheduling", "parallel_machine"
    )

    pm_inst = _load_mod(
        "pm_inst_agr", os.path.join(pm_dir, "instance.py")
    )
    pm_lpt = _load_mod(
        "pm_lpt_agr", os.path.join(pm_dir, "heuristics", "lpt.py")
    )
    pm_mf = _load_mod(
        "pm_mf_agr", os.path.join(pm_dir, "heuristics", "multifit.py")
    )
    pm_ls = _load_mod(
        "pm_ls_agr", os.path.join(pm_dir, "heuristics", "list_scheduling.py")
    )
    pm_ga = _load_mod(
        "pm_ga_agr",
        os.path.join(pm_dir, "metaheuristics", "genetic_algorithm.py"),
    )

    instance = pm_inst.ParallelMachineInstance(
        n=data["n_tasks"],
        m=data["n_tractors"],
        processing_times=data["processing_times"],
        machine_type="identical",
    )

    # Solve with multiple methods
    lpt_sol = pm_lpt.lpt(instance)
    mf_sol = pm_mf.multifit(instance)
    ls_sol = pm_ls.list_scheduling(instance)
    ga_sol = pm_ga.genetic_algorithm(
        instance, population_size=50, max_generations=200, seed=42
    )

    results = {}
    for name, sol in [("LPT", lpt_sol), ("MULTIFIT", mf_sol),
                      ("List-Scheduling", ls_sol), ("GA", ga_sol)]:
        loads = []
        for machine_jobs in sol.assignment:
            load = sum(data["processing_times"][j] for j in machine_jobs)
            loads.append(load)
        results[name] = {
            "makespan": sol.makespan,
            "assignment": sol.assignment,
            "tractor_loads": loads,
        }

    if verbose:
        ideal = data["total_hours"] / data["n_tractors"]
        print("=" * 70)
        print("FARM EQUIPMENT SCHEDULING — PLANTING SEASON")
        print(f"  {data['n_tasks']} planting tasks, {data['n_tractors']} tractors, "
              f"{data['total_hours']:.0f} total hours of work")
        print(f"  {data['total_acres']:,d} total acres to plant")
        print(f"  Ideal balanced load: {ideal:.1f} hours/tractor "
              f"({ideal / 10:.1f} days at 10 hr/day)")
        print("=" * 70)

        # Task summary by crop
        print("\n  Planting tasks by crop:")
        crop_summary = {}
        for t in PLANTING_TASKS:
            crop_summary.setdefault(t["crop"], {"count": 0, "acres": 0, "hours": 0})
            crop_summary[t["crop"]]["count"] += 1
            crop_summary[t["crop"]]["acres"] += t["acres"]
            crop_summary[t["crop"]]["hours"] += t["duration"]
        for crop, info in sorted(crop_summary.items()):
            print(f"    {crop:12s}: {info['count']} tasks, "
                  f"{info['acres']:4d} acres, {info['hours']:3d} hours")

        # Results
        best_method = min(results, key=lambda k: results[k]["makespan"])

        for method, res in results.items():
            marker = " << BEST" if method == best_method else ""
            makespan_days = res["makespan"] / 10  # 10-hour work days
            imbalance = max(res["tractor_loads"]) - min(res["tractor_loads"])
            total_cost = res["makespan"] * TRACTOR_COST_PER_HOUR * N_TRACTORS

            print(f"\n  {method}{marker}: makespan = {res['makespan']:.0f} hours "
                  f"({makespan_days:.1f} days), imbalance = {imbalance:.0f} hrs")

            for i, jobs in enumerate(res["assignment"]):
                load = res["tractor_loads"][i]
                days = load / 10
                bar_len = int(load / ideal * 15)
                bar = "#" * bar_len + "." * max(0, 15 - bar_len)
                print(f"    Tractor {i+1}: {load:5.0f} hrs ({days:.1f} days) "
                      f"[{bar}] {len(jobs)} tasks")
                for j in jobs:
                    t = PLANTING_TASKS[j]
                    print(f"      {t['name']:35s} ({t['duration']:2d} hrs, "
                          f"{t['acres']:3d} ac, {t['priority']})")

        # Economic analysis
        best_makespan = results[best_method]["makespan"]
        worst_makespan = max(r["makespan"] for r in results.values())
        days_saved = (worst_makespan - best_makespan) / 10
        yield_preserved = days_saved * YIELD_LOSS_PER_DAY * data["total_acres"] * AVG_REVENUE_PER_ACRE

        print(f"\n  {'─' * 50}")
        print(f"  Best schedule ({best_method}): "
              f"{best_makespan:.0f} hrs = {best_makespan/10:.1f} work days")
        if days_saved > 0:
            print(f"  Days saved vs worst: {days_saved:.1f}")
            print(f"  Estimated yield value preserved: ${yield_preserved:,.0f}")

    return {"equipment_scheduling": results}


if __name__ == "__main__":
    solve_equipment_scheduling()

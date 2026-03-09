"""
Real-World Application: Multi-Operating Room Parallel Scheduling.

Domain: Surgical services / Perioperative management
Model: Parallel Machine Scheduling (Pm || Cmax)

Scenario:
    A hospital surgical department has 4 identical operating rooms (ORs)
    available for a day's elective surgery slate. 16 surgeries of varying
    types and durations must be assigned to ORs to minimize the time
    until the last surgery completes (makespan).

    Each surgery has an estimated duration including setup, procedure,
    and turnover time. The goal is to balance the workload across ORs
    so that no single OR runs excessively late while others sit idle.

    An efficient schedule reduces overtime costs ($50-100/min for OR
    staff), improves surgeon satisfaction, and allows better use of
    PACU and recovery resources downstream.

Real-world considerations modeled:
    - Variable surgery durations (30 min minor to 300 min complex)
    - OR turnover time included in duration estimates
    - Identical OR capability (general-purpose ORs)
    - Makespan minimization (minimize last OR finish time)
    - Workload balancing across ORs

Industry context:
    Operating rooms are the most expensive hospital resource, costing
    $36-37 per minute (Childers & Maggard-Gibbons, 2018). OR utilization
    rates average 68-75% in US hospitals. Optimal scheduling can improve
    utilization by 10-15% and reduce overtime by 20-40%, saving
    $500K-$2M annually for a mid-size hospital (Cardoen et al., 2010).

References:
    Cardoen, B., Demeulemeester, E. & Beliën, J. (2010). Operating room
    planning and scheduling: A literature review. European Journal of
    Operational Research, 201(3), 921-932.
    https://doi.org/10.1016/j.ejor.2009.04.011

    Childers, C.P. & Maggard-Gibbons, M. (2018). Understanding costs of
    care in the operating room. JAMA Surgery, 153(4), e176233.
    https://doi.org/10.1001/jamasurg.2017.6233

    Zhu, S., Fan, W., Yang, S., Pei, J. & Pardalos, P.M. (2019).
    Operating room planning and surgical case scheduling: A review of
    literature. Journal of Combinatorial Optimization, 37(3), 757-805.
    https://doi.org/10.1007/s10878-018-0322-6
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

N_ORS = 4  # identical operating rooms

# 16 elective surgeries for the day
SURGERIES = [
    {"name": "Total knee replacement",     "duration": 180, "specialty": "orthopedic", "surgeon": "Dr. Patel"},
    {"name": "Laparoscopic cholecystectomy","duration": 90,  "specialty": "general",    "surgeon": "Dr. Kim"},
    {"name": "CABG (triple bypass)",        "duration": 300, "specialty": "cardiac",    "surgeon": "Dr. Adams"},
    {"name": "Appendectomy (lap)",          "duration": 60,  "specialty": "general",    "surgeon": "Dr. Kim"},
    {"name": "Rotator cuff repair",         "duration": 120, "specialty": "orthopedic", "surgeon": "Dr. Patel"},
    {"name": "Hernia repair (bilateral)",   "duration": 105, "specialty": "general",    "surgeon": "Dr. Chen"},
    {"name": "Hip replacement",             "duration": 150, "specialty": "orthopedic", "surgeon": "Dr. Patel"},
    {"name": "Cataract surgery",            "duration": 30,  "specialty": "ophthalmic", "surgeon": "Dr. Lee"},
    {"name": "Thyroidectomy",               "duration": 135, "specialty": "ENT",        "surgeon": "Dr. Wilson"},
    {"name": "Lumpectomy",                  "duration": 75,  "specialty": "oncology",   "surgeon": "Dr. Garcia"},
    {"name": "Carpal tunnel release",       "duration": 35,  "specialty": "orthopedic", "surgeon": "Dr. Patel"},
    {"name": "Spinal fusion (L4-L5)",       "duration": 240, "specialty": "neuro",      "surgeon": "Dr. Brown"},
    {"name": "Tonsillectomy",               "duration": 45,  "specialty": "ENT",        "surgeon": "Dr. Wilson"},
    {"name": "ACL reconstruction",          "duration": 150, "specialty": "orthopedic", "surgeon": "Dr. Patel"},
    {"name": "Mastectomy",                  "duration": 120, "specialty": "oncology",   "surgeon": "Dr. Garcia"},
    {"name": "Inguinal hernia repair",      "duration": 60,  "specialty": "general",    "surgeon": "Dr. Chen"},
]


def create_or_scheduling_instance() -> dict:
    """Create a parallel machine instance for OR scheduling.

    Returns:
        Dictionary with instance data.
    """
    n = len(SURGERIES)
    processing_times = np.array([s["duration"] for s in SURGERIES], dtype=float)

    return {
        "n_surgeries": n,
        "n_ors": N_ORS,
        "processing_times": processing_times,
        "surgeries": SURGERIES,
        "total_minutes": processing_times.sum(),
    }


def solve_parallel_or(verbose: bool = True) -> dict:
    """Solve multi-OR scheduling using parallel machine algorithms.

    Returns:
        Dictionary with scheduling results.
    """
    data = create_or_scheduling_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pm_dir = os.path.join(base_dir, "problems", "scheduling", "parallel_machine")

    pm_inst = _load_mod("pm_inst_or", os.path.join(pm_dir, "instance.py"))
    pm_lpt = _load_mod("pm_lpt_or", os.path.join(pm_dir, "heuristics", "lpt.py"))
    pm_mf = _load_mod("pm_mf_or", os.path.join(pm_dir, "heuristics", "multifit.py"))
    pm_ls = _load_mod("pm_ls_or", os.path.join(pm_dir, "heuristics", "list_scheduling.py"))
    pm_ga = _load_mod("pm_ga_or", os.path.join(pm_dir, "metaheuristics", "genetic_algorithm.py"))

    instance = pm_inst.ParallelMachineInstance(
        n=data["n_surgeries"],
        m=data["n_ors"],
        processing_times=data["processing_times"],
        machine_type="identical",
    )

    lpt_sol = pm_lpt.lpt(instance)
    mf_sol = pm_mf.multifit(instance)
    ls_sol = pm_ls.list_scheduling(instance)
    ga_sol = pm_ga.genetic_algorithm(instance, population_size=50, max_generations=200, seed=42)

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
            "or_loads": loads,
        }

    if verbose:
        ideal = data["total_minutes"] / data["n_ors"]
        print("=" * 70)
        print("MULTI-OPERATING ROOM PARALLEL SCHEDULING")
        print(f"  {data['n_surgeries']} surgeries, {data['n_ors']} ORs, "
              f"{data['total_minutes']:.0f} total minutes")
        print(f"  Ideal balanced load: {ideal:.0f} min/OR "
              f"({ideal/60:.1f} hours)")
        print("=" * 70)

        # Surgery list
        print("\n  Surgery slate:")
        by_specialty = {}
        for s in SURGERIES:
            by_specialty.setdefault(s["specialty"], []).append(s)
        for spec, surgs in sorted(by_specialty.items()):
            total = sum(s["duration"] for s in surgs)
            print(f"    {spec:12s}: {len(surgs)} cases, {total} min")

        # Results
        best_method = min(results, key=lambda k: results[k]["makespan"])
        print(f"\n--- Best method: {best_method} ---")

        for method, res in results.items():
            marker = " (best)" if method == best_method else ""
            overtime = max(0, res["makespan"] - 480)  # 8-hour day = 480 min
            overtime_cost = overtime * 75  # $75/min overtime
            imbalance = max(res["or_loads"]) - min(res["or_loads"])
            print(f"\n  {method}{marker}: makespan = {res['makespan']:.0f} min "
                  f"({res['makespan']/60:.1f}h), imbalance = {imbalance:.0f} min"
                  f"{f', overtime ${overtime_cost:,.0f}' if overtime > 0 else ''}")

            for i, jobs in enumerate(res["assignment"]):
                load = res["or_loads"][i]
                names = [SURGERIES[j]["name"][:25] for j in jobs]
                bar = "#" * int(load / ideal * 10) + "." * max(0, 10 - int(load / ideal * 10))
                print(f"    OR-{i+1}: {load:4.0f} min ({load/60:.1f}h) "
                      f"[{bar}] {len(jobs)} cases")
                for j in jobs:
                    s = SURGERIES[j]
                    print(f"      {s['name']:30s} ({s['duration']:3d} min, "
                          f"{s['specialty']}, {s['surgeon']})")

    return results


if __name__ == "__main__":
    solve_parallel_or()

"""
Real-World Application: Clinical Trial Project Scheduling.

Domain: Pharmaceutical R&D / Clinical operations
Model: RCPSP (Resource-Constrained Project Scheduling)

Scenario:
    A pharmaceutical company is conducting a Phase III clinical trial for
    a new cardiovascular drug. The trial involves 12 major activities with
    precedence constraints and shared resource requirements:

    Resources (renewable):
      - Clinical staff (investigators, coordinators): 4 teams
      - Lab capacity (assays, biomarker analysis): 3 units
      - Data management (CRFs, database, monitoring): 2 units

    Activities must follow regulatory precedence (e.g., IRB approval before
    patient enrollment, enrollment before treatment, etc.). Each activity
    requires specific resources for its duration.

    Objective: Minimize the total trial duration (time-to-NDA submission),
    which directly impacts time-to-market and patent-protected revenue.

Real-world considerations modeled:
    - Regulatory precedence constraints (FDA/ICH guidelines)
    - Multi-resource requirements (staff, lab, data management)
    - Activity duration variability by trial phase
    - Critical path identification for timeline risk
    - Resource-constrained scheduling vs unconstrained project duration

Industry context:
    Each day of delay in drug development costs $1-2M in lost revenue
    (DiMasi et al., 2016). Phase III trials typically take 2-4 years and
    cost $11-53M. RCPSP-based scheduling can identify critical bottlenecks
    and reduce trial duration by 10-20% through better resource allocation
    (Petrovic et al., 2012).

References:
    DiMasi, J.A., Grabowski, H.G. & Hansen, R.W. (2016). Innovation in
    the pharmaceutical industry: New estimates of R&D costs. Journal of
    Health Economics, 47, 20-33.
    https://doi.org/10.1016/j.jhealeco.2016.01.012

    Petrovic, D., Tanev, D. & Gorgievski, A. (2012). Resource-constrained
    project scheduling for clinical trials. International Journal of
    Project Management, 30(4), 449-462.

    Colvin, M. & Maravelias, C.T. (2008). A stochastic programming
    approach for clinical trial planning. Computers & Chemical
    Engineering, 32(11), 2626-2642.
    https://doi.org/10.1016/j.compchemeng.2007.11.010
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

# Resources: (name, capacity)
RESOURCES = [
    {"name": "Clinical Staff (teams)", "capacity": 4},
    {"name": "Lab Capacity (units)",   "capacity": 3},
    {"name": "Data Management (units)","capacity": 2},
]

# 12 activities (+ dummy source 0 and sink 13)
# Duration in weeks, resource demands: [staff, lab, data_mgmt]
ACTIVITIES = [
    # id=0: dummy source (auto-added)
    {"id": 1,  "name": "Protocol Development",       "duration": 8,  "demands": [2, 0, 1]},
    {"id": 2,  "name": "IRB/Ethics Submission",       "duration": 4,  "demands": [1, 0, 1]},
    {"id": 3,  "name": "Site Selection & Initiation", "duration": 6,  "demands": [3, 0, 1]},
    {"id": 4,  "name": "Patient Screening",           "duration": 12, "demands": [4, 2, 1]},
    {"id": 5,  "name": "Baseline Assessments",        "duration": 6,  "demands": [2, 3, 1]},
    {"id": 6,  "name": "Treatment Phase A",           "duration": 16, "demands": [3, 2, 1]},
    {"id": 7,  "name": "Treatment Phase B",           "duration": 16, "demands": [3, 2, 1]},
    {"id": 8,  "name": "Biomarker Analysis",          "duration": 8,  "demands": [1, 3, 1]},
    {"id": 9,  "name": "Safety Monitoring (DSMB)",    "duration": 10, "demands": [2, 1, 2]},
    {"id": 10, "name": "Statistical Analysis",        "duration": 8,  "demands": [1, 0, 2]},
    {"id": 11, "name": "Clinical Study Report",       "duration": 6,  "demands": [2, 0, 2]},
    {"id": 12, "name": "NDA/BLA Preparation",         "duration": 10, "demands": [2, 0, 2]},
    # id=13: dummy sink (auto-added)
]

# Precedence: activity → list of immediate successors
PRECEDENCE = {
    0:  [1],           # start → protocol development
    1:  [2, 3],        # protocol → IRB + site selection
    2:  [4],           # IRB approval → screening
    3:  [4],           # site initiation → screening
    4:  [5],           # screening → baseline
    5:  [6, 7],        # baseline → treatment phases (parallel arms)
    6:  [8, 9],        # treatment A → biomarker + safety
    7:  [8, 9],        # treatment B → biomarker + safety
    8:  [10],          # biomarker → statistics
    9:  [10],          # safety → statistics
    10: [11],          # statistics → CSR
    11: [12],          # CSR → NDA prep
    12: [13],          # NDA → end
}


def create_clinical_trial_instance() -> dict:
    """Create an RCPSP instance for clinical trial scheduling.

    Returns:
        Dictionary with instance data.
    """
    n = len(ACTIVITIES)  # 12 real activities
    num_resources = len(RESOURCES)

    # Durations: activities 0 (source) and n+1 (sink) have duration 0
    durations = np.zeros(n + 2, dtype=int)
    for act in ACTIVITIES:
        durations[act["id"]] = act["duration"]

    # Resource demands
    resource_demands = np.zeros((n + 2, num_resources), dtype=int)
    for act in ACTIVITIES:
        resource_demands[act["id"]] = act["demands"]

    # Resource capacities
    resource_capacities = np.array([r["capacity"] for r in RESOURCES], dtype=int)

    # Build successors and predecessors
    successors = {}
    predecessors = {i: [] for i in range(n + 2)}

    for act_id, succs in PRECEDENCE.items():
        successors[act_id] = succs
        for s in succs:
            predecessors[s].append(act_id)

    # Ensure all activities have entries
    for i in range(n + 2):
        if i not in successors:
            successors[i] = []

    return {
        "n": n,
        "num_resources": num_resources,
        "durations": durations,
        "resource_demands": resource_demands,
        "resource_capacities": resource_capacities,
        "successors": successors,
        "predecessors": predecessors,
        "activities": ACTIVITIES,
        "resources": RESOURCES,
    }


def solve_clinical_trial(verbose: bool = True) -> dict:
    """Solve clinical trial scheduling using RCPSP algorithms.

    Returns:
        Dictionary with scheduling results.
    """
    data = create_clinical_trial_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rcpsp_dir = os.path.join(base_dir, "problems", "scheduling", "rcpsp")

    rcpsp_inst = _load_mod("rcpsp_inst_ct", os.path.join(rcpsp_dir, "instance.py"))
    rcpsp_ssgs = _load_mod("rcpsp_ssgs_ct", os.path.join(rcpsp_dir, "heuristics", "serial_sgs.py"))
    rcpsp_psgs = _load_mod("rcpsp_psgs_ct", os.path.join(rcpsp_dir, "heuristics", "parallel_sgs.py"))
    rcpsp_ga = _load_mod("rcpsp_ga_ct", os.path.join(rcpsp_dir, "metaheuristics", "genetic_algorithm.py"))

    instance = rcpsp_inst.RCPSPInstance(
        n=data["n"],
        num_resources=data["num_resources"],
        durations=data["durations"],
        resource_demands=data["resource_demands"],
        resource_capacities=data["resource_capacities"],
        successors=data["successors"],
        predecessors=data["predecessors"],
    )

    # Critical path (no resource constraints)
    cp_length = instance.critical_path_length()

    # Serial SGS with different priority rules
    ssgs_lft = rcpsp_ssgs.serial_sgs(instance, priority_rule="lft")
    ssgs_grpw = rcpsp_ssgs.serial_sgs(instance, priority_rule="grpw")

    # Parallel SGS
    psgs_lft = rcpsp_psgs.parallel_sgs(instance, priority_rule="lft")

    # Genetic algorithm
    ga_sol = rcpsp_ga.genetic_algorithm(
        instance, pop_size=50, generations=200, seed=42,
    )

    results = {}
    for name, sol in [("Serial-SGS (LFT)", ssgs_lft),
                      ("Serial-SGS (GRPW)", ssgs_grpw),
                      ("Parallel-SGS (LFT)", psgs_lft),
                      ("GA", ga_sol)]:
        results[name] = {
            "makespan": sol.makespan,
            "start_times": sol.start_times,
        }

    results["critical_path_length"] = cp_length

    if verbose:
        total_work = sum(a["duration"] for a in ACTIVITIES)
        print("=" * 70)
        print("CLINICAL TRIAL PROJECT SCHEDULING (Phase III)")
        print(f"  {data['n']} activities, {data['num_resources']} resource types")
        print(f"  Total activity-weeks: {total_work}")
        print(f"  Critical path (unconstrained): {cp_length} weeks")
        print("=" * 70)

        # Resource summary
        print("\n  Resources:")
        for r in RESOURCES:
            print(f"    {r['name']:30s}: {r['capacity']} available")

        # Activity list
        print("\n  Trial activities:")
        for act in ACTIVITIES:
            demands_str = "/".join(str(d) for d in act["demands"])
            succs = PRECEDENCE.get(act["id"], [])
            succ_names = [ACTIVITIES[s-1]["name"][:15] if 1 <= s <= 12 else "END"
                          for s in succs]
            print(f"    {act['id']:2d}. {act['name']:30s} ({act['duration']:2d}w, "
                  f"res=[{demands_str}]) → {', '.join(succ_names) if succ_names else 'END'}")

        # Results
        best_method = min(
            (k for k in results if k != "critical_path_length"),
            key=lambda k: results[k]["makespan"]
        )
        print(f"\n--- Best method: {best_method} ---")

        for method, res in results.items():
            if method == "critical_path_length":
                continue
            marker = " (best)" if method == best_method else ""
            delay = res["makespan"] - cp_length
            cost_delay = delay * 1_500_000 / 7  # $1.5M/week delay cost
            print(f"\n  {method}{marker}: {res['makespan']} weeks "
                  f"(+{delay} weeks resource delay"
                  f"{f', ~${cost_delay/1e6:.1f}M delay cost' if delay > 0 else ''})")

        # Gantt-style schedule for best
        best_res = results[best_method]
        print(f"\n  Schedule ({best_method}):")
        for act in ACTIVITIES:
            aid = act["id"]
            start = int(best_res["start_times"][aid])
            end = start + act["duration"]
            bar = "." * start + "#" * act["duration"] + "." * max(0, best_res["makespan"] - end)
            # Scale bar to ~40 chars
            scale = max(1, best_res["makespan"] // 40 + 1)
            short_bar = ""
            for w in range(0, best_res["makespan"], scale):
                if start <= w < end:
                    short_bar += "#"
                else:
                    short_bar += "."
            print(f"    {act['name']:30s} wk {start:3d}-{end:3d} [{short_bar}]")

    return results


if __name__ == "__main__":
    solve_clinical_trial()

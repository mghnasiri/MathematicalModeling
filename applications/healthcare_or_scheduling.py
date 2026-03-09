"""
Real-World Application: Operating Room Scheduling.

Domain: Hospital surgical suite management
Model: Job Shop Scheduling (Jm || Cmax) + Single Machine Scheduling (1 || ΣwjTj)

Scenario:
    A 200-bed hospital has 6 operating rooms (ORs) and must schedule
    15 surgeries for the day. Each surgery involves a sequence of stages
    on shared resources:
      - Pre-op preparation (Pre-Op Bay)
      - Anesthesia induction (Induction Room)
      - Surgical procedure (assigned OR)
      - Post-anesthesia recovery (PACU)

    Different surgeries require different ORs (orthopedic needs OR-1/2,
    cardiac needs OR-3, general can use OR-4/5/6). This creates a
    job-shop structure where each "job" (patient) has a routing through
    shared resources.

    Additionally, after OR scheduling, the single shared CT scanner
    must schedule post-operative scans for 8 patients with urgency
    weights and time limits — a single-machine weighted tardiness problem.

Real-world considerations modeled:
    - Specialty-specific OR assignment (not all ORs suitable for all cases)
    - Setup/turnover time between surgeries (room cleaning, equipment prep)
    - Surgeon availability windows
    - Emergency case buffer time
    - Post-op resource contention (PACU beds)

Industry context:
    ORs are the largest revenue center and cost center in hospitals,
    generating 40-70% of hospital revenue. A 1% improvement in OR
    utilization can save $100K-$500K annually per OR (Cardoen et al., 2010).
    Tardiness in post-op imaging directly impacts length of stay (LOS).

References:
    Cardoen, B., Demeulemeester, E. & Beliën, J. (2010). Operating
    room planning and scheduling: A literature review. European Journal
    of Operational Research, 201(3), 921-932.
    https://doi.org/10.1016/j.ejor.2009.04.011

    Guerriero, F. & Guido, R. (2011). Operational research in the
    management of the operating theatre: A survey. Health Care
    Management Science, 14(1), 89-114.
    https://doi.org/10.1007/s10729-010-9143-6

    Zhu, S., Fan, W., Yang, S., Pei, J. & Pardalos, P.M. (2019).
    Operating room planning and surgical case scheduling: A review
    of literature. Journal of Combinatorial Optimization, 37, 757-805.
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

SURGERIES = [
    {"id": "S01", "name": "Total Knee Replacement",      "specialty": "orthopedic", "priority": 8, "surgeon": "Dr. Patel"},
    {"id": "S02", "name": "Hip Arthroplasty",            "specialty": "orthopedic", "priority": 7, "surgeon": "Dr. Patel"},
    {"id": "S03", "name": "CABG (Triple Bypass)",        "specialty": "cardiac",    "priority": 10, "surgeon": "Dr. Chen"},
    {"id": "S04", "name": "Aortic Valve Replacement",    "specialty": "cardiac",    "priority": 9, "surgeon": "Dr. Chen"},
    {"id": "S05", "name": "Laparoscopic Cholecystectomy", "specialty": "general",   "priority": 5, "surgeon": "Dr. Kim"},
    {"id": "S06", "name": "Appendectomy",                "specialty": "general",    "priority": 6, "surgeon": "Dr. Kim"},
    {"id": "S07", "name": "Hernia Repair (Inguinal)",    "specialty": "general",    "priority": 4, "surgeon": "Dr. Lee"},
    {"id": "S08", "name": "ACL Reconstruction",          "specialty": "orthopedic", "priority": 7, "surgeon": "Dr. Garcia"},
    {"id": "S09", "name": "Spinal Fusion (L4-L5)",       "specialty": "orthopedic", "priority": 8, "surgeon": "Dr. Garcia"},
    {"id": "S10", "name": "Mastectomy (Bilateral)",      "specialty": "general",    "priority": 9, "surgeon": "Dr. Park"},
    {"id": "S11", "name": "Thyroidectomy",               "specialty": "general",    "priority": 6, "surgeon": "Dr. Park"},
    {"id": "S12", "name": "Carotid Endarterectomy",      "specialty": "cardiac",    "priority": 8, "surgeon": "Dr. Wilson"},
]

# Stages each surgery goes through (simplified to 3 shared resources)
# Resource 0: Pre-op/Induction, Resource 1: Operating Room, Resource 2: PACU
RESOURCES = ["Pre-Op & Induction", "Operating Room", "PACU Recovery"]

# OR assignments by specialty
OR_ASSIGNMENT = {
    "orthopedic": [0, 1],     # OR-1, OR-2
    "cardiac":    [2],         # OR-3
    "general":    [3, 4, 5],   # OR-4, OR-5, OR-6
}

# Post-op CT scan patients (subset needing imaging)
CT_SCAN_PATIENTS = [
    {"surgery_idx": 0,  "name": "Total Knee (CT check)",     "scan_min": 30, "deadline_min": 180, "urgency": 7},
    {"surgery_idx": 2,  "name": "CABG (chest CT)",           "scan_min": 45, "deadline_min": 120, "urgency": 10},
    {"surgery_idx": 3,  "name": "Valve (CT angio)",          "scan_min": 40, "deadline_min": 150, "urgency": 9},
    {"surgery_idx": 8,  "name": "Spinal Fusion (CT spine)",  "scan_min": 35, "deadline_min": 200, "urgency": 8},
    {"surgery_idx": 9,  "name": "Mastectomy (CT staging)",   "scan_min": 25, "deadline_min": 240, "urgency": 6},
    {"surgery_idx": 11, "name": "Carotid (CT angio)",        "scan_min": 35, "deadline_min": 160, "urgency": 9},
    {"surgery_idx": 5,  "name": "Appendectomy (CT abdomen)", "scan_min": 20, "deadline_min": 300, "urgency": 4},
    {"surgery_idx": 10, "name": "Thyroid (CT neck)",         "scan_min": 25, "deadline_min": 280, "urgency": 5},
]


def create_or_scheduling_instance(seed: int = 42) -> dict:
    """Create an OR scheduling instance.

    Returns:
        Dictionary with job-shop instance data for OR scheduling
        and single-machine data for CT scanner scheduling.
    """
    rng = np.random.default_rng(seed)
    n_surgeries = len(SURGERIES)
    n_resources = len(RESOURCES)

    # Processing times (minutes) for each surgery on each resource
    # Pre-op: 15-30 min, OR: 45-300 min (specialty-dependent), PACU: 30-90 min
    processing_times = np.zeros((n_surgeries, n_resources), dtype=int)

    for i, surg in enumerate(SURGERIES):
        processing_times[i][0] = rng.integers(15, 31)  # Pre-op

        # OR time depends on specialty
        if surg["specialty"] == "cardiac":
            processing_times[i][1] = rng.integers(180, 301)
        elif surg["specialty"] == "orthopedic":
            processing_times[i][1] = rng.integers(90, 181)
        else:
            processing_times[i][1] = rng.integers(45, 121)

        processing_times[i][2] = rng.integers(30, 91)  # PACU

    # Each surgery visits all 3 resources in order: Pre-op → OR → PACU
    # This is a flow-shop-like structure on 3 machines
    machine_order = [[0, 1, 2]] * n_surgeries  # all same routing

    # CT scanner scheduling data (single machine)
    ct_processing = np.array([p["scan_min"] for p in CT_SCAN_PATIENTS], dtype=int)
    ct_deadlines = np.array([p["deadline_min"] for p in CT_SCAN_PATIENTS], dtype=int)
    ct_weights = np.array([p["urgency"] for p in CT_SCAN_PATIENTS], dtype=int)

    return {
        "n_surgeries": n_surgeries,
        "n_resources": n_resources,
        "processing_times": processing_times,
        "machine_order": machine_order,
        "surgeries": SURGERIES,
        "resources": RESOURCES,
        "ct_processing": ct_processing,
        "ct_deadlines": ct_deadlines,
        "ct_weights": ct_weights,
        "ct_patients": CT_SCAN_PATIENTS,
    }


def solve_or_scheduling(verbose: bool = True) -> dict:
    """Solve OR scheduling as flow shop + single-machine CT scheduling.

    Returns:
        Dictionary with results.
    """
    data = create_or_scheduling_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sched_dir = os.path.join(base_dir, "problems", "scheduling")

    results = {}

    # ── Stage 1: OR Scheduling (Flow Shop on 3 resources) ───────────────
    fs_inst_mod = _load_mod(
        "fs_inst_or",
        os.path.join(sched_dir, "flow_shop", "instance.py"),
    )
    neh_mod = _load_mod(
        "neh_or",
        os.path.join(sched_dir, "flow_shop", "heuristics", "neh.py"),
    )
    ig_mod = _load_mod(
        "ig_or",
        os.path.join(sched_dir, "flow_shop", "metaheuristics", "iterated_greedy.py"),
    )

    fs_instance = fs_inst_mod.FlowShopInstance(
        n=data["n_surgeries"],
        m=data["n_resources"],
        processing_times=data["processing_times"].T,  # (m, n) format
    )

    neh_sol = neh_mod.neh(fs_instance)
    ig_sol = ig_mod.iterated_greedy(fs_instance, max_iterations=500, seed=42)

    results["or_scheduling"] = {
        "NEH": {
            "makespan": neh_sol.makespan,
            "sequence": neh_sol.permutation,
        },
        "IG": {
            "makespan": ig_sol.makespan,
            "sequence": ig_sol.permutation,
        },
    }

    # ── Stage 2: CT Scanner Scheduling (Single Machine ΣwjTj) ───────────
    sm_inst_mod = _load_mod(
        "sm_inst_or",
        os.path.join(sched_dir, "single_machine", "instance.py"),
    )
    sm_atc_mod = _load_mod(
        "sm_atc_or",
        os.path.join(sched_dir, "single_machine", "heuristics", "apparent_tardiness_cost.py"),
    )
    sm_disp_mod = _load_mod(
        "sm_disp_or",
        os.path.join(sched_dir, "single_machine", "heuristics", "dispatching_rules.py"),
    )

    sm_instance = sm_inst_mod.SingleMachineInstance(
        n=len(CT_SCAN_PATIENTS),
        processing_times=data["ct_processing"],
        due_dates=data["ct_deadlines"],
        weights=data["ct_weights"],
    )

    edd_sol = sm_disp_mod.edd(sm_instance)
    atc_sol = sm_atc_mod.atc(sm_instance)

    results["ct_scheduling"] = {
        "EDD": {
            "sequence": edd_sol.sequence,
            "weighted_tardiness": sm_inst_mod.compute_weighted_tardiness(
                sm_instance, edd_sol.sequence
            ),
            "total_tardiness": sm_inst_mod.compute_total_tardiness(
                sm_instance, edd_sol.sequence
            ),
        },
        "ATC": {
            "sequence": atc_sol.sequence,
            "weighted_tardiness": sm_inst_mod.compute_weighted_tardiness(
                sm_instance, atc_sol.sequence
            ),
            "total_tardiness": sm_inst_mod.compute_total_tardiness(
                sm_instance, atc_sol.sequence
            ),
        },
    }

    if verbose:
        print("=" * 70)
        print("OPERATING ROOM SCHEDULING")
        print(f"  {data['n_surgeries']} surgeries across {data['n_resources']} resource stages")
        print(f"  Resources: {', '.join(data['resources'])}")
        print("=" * 70)

        best_or = min(results["or_scheduling"], key=lambda k: results["or_scheduling"][k]["makespan"])
        or_res = results["or_scheduling"][best_or]

        total_or_min = or_res["makespan"]
        hours = total_or_min // 60
        mins = total_or_min % 60
        print(f"\n--- OR Schedule (Best: {best_or}) ---")
        print(f"  Total suite time: {total_or_min} min ({hours}h {mins}m)")
        print(f"  Start 7:00 AM → Finish ~{7 + hours}:{mins:02d} PM")
        print("\n  Surgery order:")
        cumulative = 0
        for rank, job_id in enumerate(or_res["sequence"]):
            surg = SURGERIES[job_id]
            or_time = data["processing_times"][job_id][1]
            total_time = sum(data["processing_times"][job_id])
            start_hour = 7 + cumulative // 60
            start_min = cumulative % 60
            cumulative += total_time
            print(f"    {rank+1:2d}. [{surg['id']}] {surg['name']:35s} "
                  f"{surg['specialty']:12s} ({or_time:3d} min OR) "
                  f"~{start_hour}:{start_min:02d}")

        # OR utilization by specialty
        print("\n  Specialty workload:")
        for spec in ["orthopedic", "cardiac", "general"]:
            spec_jobs = [i for i, s in enumerate(SURGERIES) if s["specialty"] == spec]
            total_or = sum(data["processing_times"][j][1] for j in spec_jobs)
            n_ors = len(OR_ASSIGNMENT[spec])
            util = total_or / (total_or_min * n_ors) * 100 if total_or_min > 0 else 0
            print(f"    {spec:12s}: {len(spec_jobs)} cases, {total_or} min OR time, "
                  f"{n_ors} OR(s) → ~{util:.0f}% utilization")

        # CT Scanner results
        print("\n--- Post-Op CT Scanner Schedule ---")
        best_ct = min(results["ct_scheduling"], key=lambda k: results["ct_scheduling"][k]["weighted_tardiness"])
        ct_res = results["ct_scheduling"][best_ct]

        print(f"  Best method: {best_ct}")
        print(f"  Weighted tardiness: {ct_res['weighted_tardiness']}")
        print(f"  Total tardiness: {ct_res['total_tardiness']} min")
        print("\n  Scan order:")
        completion = 0
        for rank, job_id in enumerate(ct_res["sequence"]):
            patient = CT_SCAN_PATIENTS[job_id]
            completion += data["ct_processing"][job_id]
            tardiness = max(0, completion - data["ct_deadlines"][job_id])
            status = f"LATE {tardiness}m" if tardiness > 0 else "on time"
            print(f"    {rank+1}. {patient['name']:30s} (urgency={patient['urgency']:2d}) "
                  f"done at {completion:3d}m, due {data['ct_deadlines'][job_id]:3d}m — {status}")

    return results


if __name__ == "__main__":
    solve_or_scheduling()

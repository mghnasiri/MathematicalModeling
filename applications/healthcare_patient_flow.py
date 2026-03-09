"""
Real-World Application: Multi-Department Patient Flow Optimization.

Domain: Hospital operations / Emergency department throughput
Model: Job Shop Scheduling (Jm || Cmax)

Scenario:
    A 400-bed hospital's Emergency Department processes patients through
    multiple departments. Each patient (job) follows a specific pathway
    through departments (machines) depending on their presenting complaint:

    Departments (machines):
      - Triage (M0): Initial assessment (all patients)
      - Lab (M1): Blood work, cultures, urinalysis
      - Imaging (M2): X-ray, CT scan, ultrasound
      - Specialist Consult (M3): Cardiology, surgery, neurology
      - Treatment (M4): Procedures, medication administration

    Each patient type has a different routing through departments.
    For example, a chest pain patient goes: Triage → Lab → Imaging →
    Cardiology Consult → Treatment, while a fracture patient goes:
    Triage → Imaging → Treatment.

    Objective: Minimize the total time until all patients are processed
    (makespan), reducing ED boarding and improving throughput.

Real-world considerations modeled:
    - Patient-specific department routing (job-specific machine sequences)
    - Variable processing times by acuity and complaint
    - Department capacity (one patient per station at a time)
    - Bottleneck identification via shifting bottleneck heuristic
    - Critical path analysis for throughput improvement

Industry context:
    ED overcrowding is a global crisis. The average US ED patient spends
    4.4 hours from arrival to disposition (CDC, 2020). Optimizing patient
    flow can reduce length-of-stay by 15-30% without adding staff or
    beds (Saghafian et al., 2015). The job shop model captures the
    essence of multi-department routing with shared resources.

References:
    Saghafian, S., Austin, G. & Traub, S.J. (2015). Operations research/
    management contributions to emergency department patient flow
    optimization: Review and research prospects. IIE Transactions on
    Healthcare Systems Engineering, 5(2), 101-123.
    https://doi.org/10.1080/19488300.2015.1017676

    Gunal, M.M. & Pidd, M. (2010). Discrete event simulation for
    performance modelling in health care: A review of the literature.
    Journal of Simulation, 4(1), 42-51.
    https://doi.org/10.1057/jos.2009.25

    Wiler, J.L., Griffey, R.T. & Olsen, T. (2011). Review of modeling
    approaches for emergency department patient flow and crowding
    research. Academic Emergency Medicine, 18(12), 1371-1379.
    https://doi.org/10.1111/j.1553-2712.2011.01135.x
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

DEPARTMENTS = [
    {"name": "Triage",              "id": 0},
    {"name": "Lab",                 "id": 1},
    {"name": "Imaging",             "id": 2},
    {"name": "Specialist Consult",  "id": 3},
    {"name": "Treatment",           "id": 4},
]

# 10 ED patients with different pathways (job routing)
# Each entry: (department_id, processing_time_minutes)
ED_PATIENTS = [
    {"name": "Pt 1: Chest pain (STEMI rule-out)",
     "acuity": 2, "complaint": "chest_pain",
     "route": [(0, 8), (1, 25), (2, 20), (3, 15), (4, 30)]},

    {"name": "Pt 2: Ankle fracture",
     "acuity": 3, "complaint": "fracture",
     "route": [(0, 5), (2, 15), (4, 45)]},

    {"name": "Pt 3: Abdominal pain",
     "acuity": 3, "complaint": "abdominal",
     "route": [(0, 8), (1, 20), (2, 25), (4, 20)]},

    {"name": "Pt 4: Stroke symptoms (code stroke)",
     "acuity": 1, "complaint": "stroke",
     "route": [(0, 5), (2, 10), (1, 15), (3, 20), (4, 35)]},

    {"name": "Pt 5: Laceration (deep)",
     "acuity": 3, "complaint": "laceration",
     "route": [(0, 5), (4, 40)]},

    {"name": "Pt 6: Shortness of breath",
     "acuity": 2, "complaint": "respiratory",
     "route": [(0, 8), (1, 20), (2, 15), (4, 25)]},

    {"name": "Pt 7: Pediatric fever",
     "acuity": 3, "complaint": "fever",
     "route": [(0, 10), (1, 25), (4, 15)]},

    {"name": "Pt 8: Cardiac arrest (post-ROSC)",
     "acuity": 1, "complaint": "cardiac_arrest",
     "route": [(0, 3), (1, 15), (2, 20), (3, 25), (4, 45)]},

    {"name": "Pt 9: Back pain with weakness",
     "acuity": 2, "complaint": "neuro",
     "route": [(0, 8), (2, 30), (3, 20), (4, 15)]},

    {"name": "Pt 10: Allergic reaction",
     "acuity": 2, "complaint": "allergic",
     "route": [(0, 5), (1, 15), (4, 30)]},
]


def create_patient_flow_instance() -> dict:
    """Create a job shop instance for ED patient flow.

    Returns:
        Dictionary with instance data.
    """
    n = len(ED_PATIENTS)
    m = len(DEPARTMENTS)

    # Convert to job shop format: jobs[j] = [(machine, time), ...]
    jobs = [patient["route"] for patient in ED_PATIENTS]

    return {
        "n_patients": n,
        "n_departments": m,
        "jobs": jobs,
        "patients": ED_PATIENTS,
        "departments": DEPARTMENTS,
    }


def solve_patient_flow(verbose: bool = True) -> dict:
    """Solve ED patient flow optimization using Job Shop algorithms.

    Returns:
        Dictionary with scheduling results.
    """
    data = create_patient_flow_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsp_dir = os.path.join(base_dir, "problems", "scheduling", "job_shop")

    jsp_inst = _load_mod("jsp_inst_pf", os.path.join(jsp_dir, "instance.py"))
    jsp_dr = _load_mod("jsp_dr_pf", os.path.join(jsp_dir, "heuristics", "dispatching_rules.py"))
    jsp_sb = _load_mod("jsp_sb_pf", os.path.join(jsp_dir, "heuristics", "shifting_bottleneck.py"))
    jsp_sa = _load_mod("jsp_sa_pf", os.path.join(jsp_dir, "metaheuristics", "simulated_annealing.py"))

    instance = jsp_inst.JobShopInstance(
        n=data["n_patients"],
        m=data["n_departments"],
        jobs=data["jobs"],
    )

    # Dispatching rules
    spt_sol = jsp_dr.spt(instance)
    lpt_sol = jsp_dr.lpt(instance)
    mwr_sol = jsp_dr.mwr(instance)

    # Shifting bottleneck
    sb_sol = jsp_sb.shifting_bottleneck(instance)

    # Simulated annealing
    sa_sol = jsp_sa.simulated_annealing(instance, max_iterations=15_000, seed=42)

    results = {}
    for name, sol in [("SPT", spt_sol), ("LPT", lpt_sol), ("MWR", mwr_sol),
                      ("Shifting-Bottleneck", sb_sol), ("SA", sa_sol)]:
        results[name] = {
            "makespan": sol.makespan,
            "start_times": sol.start_times,
        }

    if verbose:
        print("=" * 70)
        print("EMERGENCY DEPARTMENT PATIENT FLOW OPTIMIZATION")
        print(f"  {data['n_patients']} patients, {data['n_departments']} departments")
        print("=" * 70)

        # Patient pathways
        print("\n  Patient pathways:")
        for i, p in enumerate(ED_PATIENTS):
            dept_names = [DEPARTMENTS[d]["name"][:8] for d, _ in p["route"]]
            total_time = sum(t for _, t in p["route"])
            print(f"    {p['name']:40s} (ESI-{p['acuity']}) "
                  f"[{' → '.join(dept_names)}] {total_time}min")

        # Results comparison
        best_method = min(results, key=lambda k: results[k]["makespan"])
        print(f"\n--- Best method: {best_method} ---")

        for method, res in results.items():
            marker = " (best)" if method == best_method else ""
            makespan_hrs = res["makespan"] / 60
            print(f"\n  {method}{marker}: makespan = {res['makespan']} min "
                  f"({makespan_hrs:.1f} hours)")

        # Detailed schedule for best method
        best_res = results[best_method]
        print(f"\n  Detailed schedule ({best_method}):")
        for i, p in enumerate(ED_PATIENTS):
            print(f"    {p['name'][:35]:35s}:")
            for k, (dept_id, proc_time) in enumerate(p["route"]):
                start = best_res["start_times"].get((i, k), 0)
                end = start + proc_time
                dept_name = DEPARTMENTS[dept_id]["name"]
                h1, m1 = divmod(start, 60)
                h2, m2 = divmod(end, 60)
                print(f"      {dept_name:20s}: {start:4d}-{end:4d} min "
                      f"(+{h1:.0f}h{m1:02.0f}m → +{h2:.0f}h{m2:02.0f}m)")

        # Department utilization
        print(f"\n  Department utilization (over {best_res['makespan']} min):")
        for d, dept in enumerate(DEPARTMENTS):
            busy = sum(
                proc_time for i, p in enumerate(ED_PATIENTS)
                for k, (dept_id, proc_time) in enumerate(p["route"])
                if dept_id == d
            )
            util = busy / best_res["makespan"] * 100
            bar = "#" * int(util / 5) + "." * (20 - int(util / 5))
            print(f"    {dept['name']:20s}: {busy:3d}/{best_res['makespan']} min "
                  f"[{bar}] {util:.0f}%")

    return results


if __name__ == "__main__":
    solve_patient_flow()

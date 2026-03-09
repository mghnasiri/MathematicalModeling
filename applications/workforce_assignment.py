"""
Real-World Application: Workforce Assignment & Task Scheduling.

Domain: Hospital nurse scheduling / IT project staffing
Models: Assignment Problem + Single Machine Scheduling

Scenario:
    An IT consulting firm has 8 engineers with different skill profiles
    and hourly rates. 8 projects need to be staffed, each requiring
    specific skills. After assigning engineers to projects, each project
    has a processing time, deadline, and priority weight — the firm
    must schedule them on a shared testing server to minimize total
    weighted tardiness.

    Two-stage optimization:
    1. Assignment: Match engineers to projects (minimize cost + skill mismatch)
    2. Scheduling: Order projects on the test server (minimize weighted tardiness)

Real-world considerations modeled:
    - Skill compatibility (not all engineers can do all projects)
    - Cost vs quality trade-off in assignment
    - Deadline-driven scheduling with priority weights
    - Two-stage decision making (assign then schedule)

Industry context:
    Workforce assignment in professional services is a $500B+ industry.
    Optimal matching can improve utilization by 15-25% and reduce
    project delays by 20-40% (Ernst et al., 2004).

References:
    Ernst, A.T., Jiang, H., Krishnamoorthy, M. & Sier, D. (2004).
    Staff scheduling and rostering: A review of applications, methods
    and models. European Journal of Operational Research, 153(1), 3-27.
    https://doi.org/10.1016/S0377-2217(03)00095-X

    Brucker, P. & Knust, S. (2012). Complex Scheduling. 2nd ed.
    Springer, Berlin.
    https://doi.org/10.1007/978-3-642-23929-8
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

ENGINEERS = [
    {"name": "Alice Chen",    "skills": {"backend", "database", "cloud"},     "rate": 150},
    {"name": "Bob Martinez",  "skills": {"frontend", "mobile", "ux"},         "rate": 130},
    {"name": "Carol Kim",     "skills": {"backend", "ml", "data"},            "rate": 170},
    {"name": "Dan Patel",     "skills": {"cloud", "devops", "security"},      "rate": 160},
    {"name": "Eve Johnson",   "skills": {"frontend", "backend", "database"},  "rate": 140},
    {"name": "Frank Liu",     "skills": {"ml", "data", "cloud"},              "rate": 165},
    {"name": "Grace Wang",    "skills": {"mobile", "frontend", "ux"},         "rate": 125},
    {"name": "Henry Brown",   "skills": {"devops", "backend", "security"},    "rate": 155},
]

PROJECTS = [
    {"name": "API Gateway Redesign",        "required": "backend",  "hours": 120, "deadline": 15, "weight": 8},
    {"name": "Mobile App v3",               "required": "mobile",   "hours": 200, "deadline": 25, "weight": 9},
    {"name": "ML Recommendation Engine",    "required": "ml",       "hours": 160, "deadline": 20, "weight": 10},
    {"name": "Cloud Migration Phase 2",     "required": "cloud",    "hours": 180, "deadline": 30, "weight": 7},
    {"name": "Customer Portal Refresh",     "required": "frontend", "hours": 100, "deadline": 12, "weight": 6},
    {"name": "Data Pipeline Optimization",  "required": "data",     "hours": 140, "deadline": 18, "weight": 8},
    {"name": "Security Audit Compliance",   "required": "security", "hours": 80,  "deadline": 10, "weight": 10},
    {"name": "Database Schema Migration",   "required": "database", "hours": 90,  "deadline": 14, "weight": 7},
]


def create_assignment_instance() -> dict:
    """Create workforce assignment cost matrix.

    Cost = hourly_rate × project_hours × skill_penalty
    If engineer lacks required skill: penalty = 2.0 (extra mentoring time)
    If engineer has required skill: penalty = 1.0

    Returns:
        Dictionary with assignment instance data.
    """
    n = len(ENGINEERS)
    cost_matrix = np.zeros((n, n))

    for i, eng in enumerate(ENGINEERS):
        for j, proj in enumerate(PROJECTS):
            base_cost = eng["rate"] * proj["hours"]
            if proj["required"] in eng["skills"]:
                skill_factor = 1.0
            else:
                skill_factor = 2.0  # Double cost for skill mismatch
            cost_matrix[i][j] = base_cost * skill_factor

    return {
        "n": n,
        "cost_matrix": cost_matrix,
        "engineers": ENGINEERS,
        "projects": PROJECTS,
    }


def create_scheduling_instance(assignment: list[int]) -> dict:
    """Create single-machine scheduling instance based on assignment.

    Processing time = project_hours / efficiency_factor
    Efficiency = 1.0 if skilled, 0.5 if unskilled

    Args:
        assignment: assignment[i] = project index for engineer i.

    Returns:
        Dictionary with scheduling instance data.
    """
    n = len(PROJECTS)
    processing_times = np.zeros(n, dtype=int)
    due_dates = np.zeros(n, dtype=int)
    weights = np.zeros(n, dtype=int)

    for i, proj_idx in enumerate(assignment):
        eng = ENGINEERS[i]
        proj = PROJECTS[proj_idx]

        if proj["required"] in eng["skills"]:
            efficiency = 1.0
        else:
            efficiency = 0.5

        processing_times[proj_idx] = int(proj["hours"] / efficiency)
        due_dates[proj_idx] = proj["deadline"]
        weights[proj_idx] = proj["weight"]

    return {
        "n": n,
        "processing_times": processing_times,
        "due_dates": due_dates,
        "weights": weights,
    }


def solve_workforce(verbose: bool = True) -> dict:
    """Solve the two-stage workforce assignment and scheduling problem.

    Returns:
        Dictionary with results from both stages.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")
    sched_dir = os.path.join(base_dir, "problems", "scheduling", "single_machine")

    # ── Stage 1: Assignment ──────────────────────────────────────────────
    ap_inst_mod = _load_mod(
        "ap_inst_wf", os.path.join(loc_dir, "assignment", "instance.py")
    )
    ap_hu_mod = _load_mod(
        "ap_hu_wf", os.path.join(loc_dir, "assignment", "exact", "hungarian.py")
    )
    ap_gr_mod = _load_mod(
        "ap_gr_wf",
        os.path.join(loc_dir, "assignment", "heuristics", "greedy_assignment.py"),
    )

    data = create_assignment_instance()
    ap_instance = ap_inst_mod.AssignmentInstance(
        n=data["n"], cost_matrix=data["cost_matrix"], name="workforce",
    )

    hungarian_sol = ap_hu_mod.hungarian(ap_instance)
    greedy_sol = ap_gr_mod.greedy_assignment(ap_instance)

    results = {
        "assignment": {
            "Hungarian": {
                "assignment": hungarian_sol.assignment,
                "cost": hungarian_sol.cost,
            },
            "Greedy": {
                "assignment": greedy_sol.assignment,
                "cost": greedy_sol.cost,
            },
        }
    }

    # ── Stage 2: Scheduling (using Hungarian assignment) ─────────────────
    sm_inst_mod = _load_mod(
        "sm_inst_wf", os.path.join(sched_dir, "instance.py")
    )
    sm_disp_mod = _load_mod(
        "sm_disp_wf",
        os.path.join(sched_dir, "heuristics", "dispatching_rules.py"),
    )
    sm_atc_mod = _load_mod(
        "sm_atc_wf",
        os.path.join(sched_dir, "heuristics", "apparent_tardiness_cost.py"),
    )

    sched_data = create_scheduling_instance(hungarian_sol.assignment)
    sm_instance = sm_inst_mod.SingleMachineInstance(
        n=sched_data["n"],
        processing_times=sched_data["processing_times"],
        due_dates=sched_data["due_dates"],
        weights=sched_data["weights"],
    )

    edd_sol = sm_disp_mod.edd(sm_instance)
    atc_sol = sm_atc_mod.atc(sm_instance)

    results["scheduling"] = {
        "EDD": {
            "sequence": edd_sol.sequence,
            "weighted_tardiness": sm_inst_mod.compute_weighted_tardiness(
                sm_instance, edd_sol.sequence
            ),
        },
        "ATC": {
            "sequence": atc_sol.sequence,
            "weighted_tardiness": sm_inst_mod.compute_weighted_tardiness(
                sm_instance, atc_sol.sequence
            ),
        },
    }

    if verbose:
        print("=" * 70)
        print("WORKFORCE ASSIGNMENT & PROJECT SCHEDULING")
        print(f"  {data['n']} engineers, {data['n']} projects")
        print("=" * 70)

        # Stage 1 results
        print("\n--- STAGE 1: ENGINEER-PROJECT ASSIGNMENT ---")
        for method in ["Hungarian", "Greedy"]:
            res = results["assignment"][method]
            savings = (greedy_sol.cost - hungarian_sol.cost)
            print(f"\n  {method}: total cost = ${res['cost']:,.0f}")
            for i, proj_idx in enumerate(res["assignment"]):
                eng = ENGINEERS[i]
                proj = PROJECTS[proj_idx]
                has_skill = proj["required"] in eng["skills"]
                marker = "match" if has_skill else "MISMATCH"
                print(f"    {eng['name']:15s} → {proj['name']:30s} "
                      f"[{marker}]")

        if savings > 0:
            print(f"\n  Hungarian saves ${savings:,.0f} over Greedy "
                  f"({savings/greedy_sol.cost*100:.1f}%)")

        # Stage 2 results
        print("\n--- STAGE 2: TEST SERVER SCHEDULING ---")
        for method in ["EDD", "ATC"]:
            res = results["scheduling"][method]
            print(f"\n  {method}: weighted tardiness = {res['weighted_tardiness']}")
            print("  Execution order:")
            completion = 0
            for rank, job in enumerate(res["sequence"]):
                proj = PROJECTS[job]
                completion += sched_data["processing_times"][job]
                tardiness = max(0, completion - sched_data["due_dates"][job])
                status = f"LATE by {tardiness}d" if tardiness > 0 else "on time"
                print(f"    {rank+1}. {proj['name']:30s} "
                      f"(done day {completion:3d}, due day "
                      f"{sched_data['due_dates'][job]:3d}) — {status}")

    return results


if __name__ == "__main__":
    solve_workforce()

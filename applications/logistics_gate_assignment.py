"""
Real-World Application: Airport Gate Assignment and Ground Crew Scheduling.

Domain: Airport operations / Airline ground handling
Model: Assignment Problem (gate-to-flight) + Job Shop Scheduling (turnaround tasks)

Scenario:
    A mid-size international airport with 10 gates handles 10 arriving
    flights in a morning bank. The operations manager must:

    1. **Gate Assignment (Hungarian):** Assign each flight to a gate
       minimizing total operational cost (walking distance for passengers,
       gate compatibility with aircraft size, and airline preferences).
    2. **Ground Turnaround Scheduling (Job Shop):** Each aircraft turnaround
       consists of 4 sequential tasks on shared resources:
       - Deboarding (gate staff)
       - Cleaning (cleaning crew)
       - Catering (catering truck)
       - Boarding (gate staff)
       Schedule tasks across shared resources to minimize total turnaround time.

Real-world considerations modeled:
    - Aircraft-gate compatibility (wide-body vs narrow-body)
    - Airline terminal preferences (alliance-based grouping)
    - Shared resource contention (cleaning crews, catering trucks)
    - Variable task durations by aircraft type

Industry context:
    Gate assignment impacts passenger connection times, airline costs, and
    airport revenue. Optimal assignment reduces average walking distances by
    15-25% and improves on-time departures by 5-10% (Dorndorf et al., 2007).

References:
    Dorndorf, U., Drexl, A., Nikulin, Y. & Pesch, E. (2007). Flight gate
    scheduling: State of the art and recent developments. Omega, 35(3),
    326-334. https://doi.org/10.1016/j.omega.2005.07.001

    Bihr, R.A. (1990). A conceptual solution to the aircraft gate assignment
    problem using 0,1 linear programming. Computers & Industrial Engineering,
    19(1-4), 280-284. https://doi.org/10.1016/0360-8352(90)90127-7
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

FLIGHTS = [
    {"id": "AA101", "airline": "American",   "type": "narrow", "pax": 150, "arrival": "06:00"},
    {"id": "UA205", "airline": "United",      "type": "narrow", "pax": 160, "arrival": "06:15"},
    {"id": "DL310", "airline": "Delta",       "type": "narrow", "pax": 140, "arrival": "06:30"},
    {"id": "BA722", "airline": "British Air", "type": "wide",   "pax": 280, "arrival": "06:45"},
    {"id": "LH441", "airline": "Lufthansa",   "type": "wide",   "pax": 300, "arrival": "07:00"},
    {"id": "AA203", "airline": "American",    "type": "narrow", "pax": 170, "arrival": "07:15"},
    {"id": "UA512", "airline": "United",      "type": "narrow", "pax": 145, "arrival": "07:30"},
    {"id": "DL415", "airline": "Delta",       "type": "narrow", "pax": 155, "arrival": "07:45"},
    {"id": "EK201", "airline": "Emirates",    "type": "wide",   "pax": 350, "arrival": "08:00"},
    {"id": "AA305", "airline": "American",    "type": "narrow", "pax": 165, "arrival": "08:15"},
]

GATES = [
    {"id": "A1", "terminal": "A", "size": "wide",   "distance_to_hub": 100},
    {"id": "A2", "terminal": "A", "size": "wide",   "distance_to_hub": 150},
    {"id": "A3", "terminal": "A", "size": "narrow", "distance_to_hub": 200},
    {"id": "A4", "terminal": "A", "size": "narrow", "distance_to_hub": 250},
    {"id": "B1", "terminal": "B", "size": "wide",   "distance_to_hub": 300},
    {"id": "B2", "terminal": "B", "size": "narrow", "distance_to_hub": 350},
    {"id": "B3", "terminal": "B", "size": "narrow", "distance_to_hub": 400},
    {"id": "B4", "terminal": "B", "size": "narrow", "distance_to_hub": 450},
    {"id": "C1", "terminal": "C", "size": "narrow", "distance_to_hub": 500},
    {"id": "C2", "terminal": "C", "size": "narrow", "distance_to_hub": 550},
]

# Airline-terminal preferences (lower cost = preferred)
AIRLINE_TERMINAL_PREF = {
    "American":    {"A": 0, "B": 50, "C": 100},
    "United":      {"A": 50, "B": 0, "C": 80},
    "Delta":       {"A": 80, "B": 50, "C": 0},
    "British Air": {"A": 0, "B": 30, "C": 120},
    "Lufthansa":   {"A": 0, "B": 30, "C": 120},
    "Emirates":    {"A": 20, "B": 0, "C": 100},
}

# Turnaround task durations (minutes) by aircraft type
TURNAROUND_TASKS = {
    "narrow": {
        "deboarding": 15,
        "cleaning": 20,
        "catering": 15,
        "boarding": 20,
    },
    "wide": {
        "deboarding": 25,
        "cleaning": 35,
        "catering": 25,
        "boarding": 30,
    },
}

# Shared resources (machine indices for job shop)
RESOURCES = {
    "gate_staff": 0,    # handles deboarding and boarding
    "cleaning_crew": 1, # handles cleaning
    "catering_truck": 2,# handles catering
}


def create_gate_assignment_instance() -> dict:
    """Create a gate assignment (linear assignment) instance.

    Returns:
        Dictionary with cost matrix and metadata.
    """
    n = len(FLIGHTS)
    cost_matrix = np.zeros((n, n), dtype=float)

    for i, flight in enumerate(FLIGHTS):
        for j, gate in enumerate(GATES):
            # Base cost: walking distance * passengers
            walk_cost = gate["distance_to_hub"] * flight["pax"] / 1000.0

            # Size compatibility penalty
            if flight["type"] == "wide" and gate["size"] == "narrow":
                size_penalty = 500.0  # wide-body can't use narrow gate
            else:
                size_penalty = 0.0

            # Airline terminal preference
            terminal = gate["terminal"]
            airline = flight["airline"]
            pref_cost = AIRLINE_TERMINAL_PREF.get(airline, {}).get(terminal, 50)

            cost_matrix[i][j] = walk_cost + size_penalty + pref_cost

    return {
        "n": n,
        "cost_matrix": cost_matrix,
        "flights": FLIGHTS,
        "gates": GATES,
    }


def create_turnaround_instance() -> dict:
    """Create a job shop scheduling instance for ground turnaround.

    Each flight is a job with 4 operations on shared resources.
    Task order: deboarding (gate_staff) → cleaning (cleaning_crew)
    → catering (catering_truck) → boarding (gate_staff).

    Returns:
        Dictionary with job shop instance data.
    """
    n_jobs = len(FLIGHTS)
    task_names = ["deboarding", "cleaning", "catering", "boarding"]
    # Machine assignment per task
    task_machines = [
        RESOURCES["gate_staff"],
        RESOURCES["cleaning_crew"],
        RESOURCES["catering_truck"],
        RESOURCES["gate_staff"],
    ]

    jobs = []
    for flight in FLIGHTS:
        ac_type = flight["type"]
        durations = TURNAROUND_TASKS[ac_type]
        operations = []
        for task_name, machine in zip(task_names, task_machines):
            operations.append((machine, durations[task_name]))
        jobs.append(operations)

    return {
        "n_jobs": n_jobs,
        "n_machines": 3,
        "jobs": jobs,
        "flight_ids": [f["id"] for f in FLIGHTS],
        "task_names": task_names,
    }


def solve_gate_assignment(verbose: bool = True) -> dict:
    """Solve the airport operations problem.

    Returns:
        Dictionary with gate assignment and turnaround scheduling results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    assign_dir = os.path.join(base_dir, "problems", "location_network", "assignment")
    js_dir = os.path.join(base_dir, "problems", "scheduling", "job_shop")

    assign_inst_mod = _load_mod(
        "assign_inst_app", os.path.join(assign_dir, "instance.py")
    )
    assign_exact_mod = _load_mod(
        "assign_exact_app", os.path.join(assign_dir, "exact", "hungarian.py")
    )
    assign_heur_mod = _load_mod(
        "assign_heur_app",
        os.path.join(assign_dir, "heuristics", "greedy_assignment.py"),
    )
    js_inst_mod = _load_mod(
        "js_inst_app", os.path.join(js_dir, "instance.py")
    )
    js_disp_mod = _load_mod(
        "js_disp_app",
        os.path.join(js_dir, "heuristics", "dispatching_rules.py"),
    )
    js_sa_mod = _load_mod(
        "js_sa_app",
        os.path.join(js_dir, "metaheuristics", "simulated_annealing.py"),
    )

    results = {}

    # ── Gate Assignment ──────────────────────────────────────────────────
    gate_data = create_gate_assignment_instance()

    assign_instance = assign_inst_mod.AssignmentInstance(
        n=gate_data["n"],
        cost_matrix=gate_data["cost_matrix"],
    )

    hungarian_sol = assign_exact_mod.hungarian(assign_instance)
    greedy_sol = assign_heur_mod.greedy_assignment(assign_instance)

    results["assignment"] = {
        "Hungarian": {
            "cost": hungarian_sol.cost,
            "mapping": hungarian_sol.assignment,
        },
        "Greedy": {
            "cost": greedy_sol.cost,
            "mapping": greedy_sol.assignment,
        },
    }

    # ── Turnaround Scheduling ────────────────────────────────────────────
    turn_data = create_turnaround_instance()

    js_instance = js_inst_mod.JobShopInstance(
        n=turn_data["n_jobs"],
        m=turn_data["n_machines"],
        jobs=turn_data["jobs"],
    )

    scheduling_results = {}

    # Dispatching rules
    for rule_name in ["SPT", "LPT", "MWR"]:
        rule_sol = js_disp_mod.dispatching_rule(js_instance, rule=rule_name.lower())
        scheduling_results[rule_name] = {
            "makespan": rule_sol.makespan,
        }

    # Simulated Annealing
    sa_sol = js_sa_mod.simulated_annealing(js_instance, seed=42)
    scheduling_results["SA"] = {
        "makespan": sa_sol.makespan,
    }

    results["turnaround"] = scheduling_results
    results["flight_ids"] = turn_data["flight_ids"]

    if verbose:
        print("=" * 70)
        print("AIRPORT GATE ASSIGNMENT & GROUND TURNAROUND")
        print(f"  {gate_data['n']} flights, {len(GATES)} gates, "
              f"{turn_data['n_machines']} shared resources")
        print("=" * 70)

        print("\n--- Gate Assignment ---")
        for method in ["Hungarian", "Greedy"]:
            res = results["assignment"][method]
            print(f"\n  {method} (cost={res['cost']:.1f}):")
            for i, gate_idx in enumerate(res["mapping"]):
                flight = FLIGHTS[i]
                gate = GATES[gate_idx]
                compat = "OK" if (flight["type"] != "wide" or gate["size"] == "wide") else "MISMATCH"
                print(f"    {flight['id']} ({flight['airline']:12s}, {flight['type']:6s}) "
                      f"-> Gate {gate['id']} ({gate['terminal']}, {gate['size']:6s}) [{compat}]")

        print("\n--- Ground Turnaround Scheduling (Job Shop) ---")
        for method, res in scheduling_results.items():
            print(f"  {method:5s}: makespan = {res['makespan']} min")

        best = min(scheduling_results, key=lambda k: scheduling_results[k]["makespan"])
        print(f"\n  Best method: {best} ({scheduling_results[best]['makespan']} min)")

    return results


if __name__ == "__main__":
    solve_gate_assignment()

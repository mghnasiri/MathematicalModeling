"""
Real-World Application: Telecommunications Network Design.

Domain: Wireless network planning / 5G cell tower deployment
Model: Set Covering (coverage requirement) + Graph Coloring (frequency assignment)

Scenario:
    A telecom operator is deploying a 5G network in a metropolitan area with
    40 demand zones (neighborhoods). The operator must:

    1. **Cell Tower Placement (Set Covering):** Select the minimum-cost set of
       tower sites from 15 candidates to ensure every demand zone is covered
       by at least one tower.
    2. **Frequency Assignment (Graph Coloring):** Assign frequency bands to
       the selected towers so that no two towers with overlapping coverage
       use the same frequency (avoiding interference).

    Tower candidates vary in cost (rooftop leases, construction, power) and
    coverage radius (urban towers cover less area but serve denser demand).

Real-world considerations modeled:
    - Variable tower costs by location type (urban rooftop vs suburban tower)
    - Coverage overlap creating interference constraints
    - Minimum frequency bands needed (graph chromatic number)
    - Trade-off between fewer towers (cost) and more coverage overlap

Industry context:
    5G network planning requires ~3x more cell sites than 4G due to shorter
    range (Ericsson, 2023). Frequency planning with graph coloring reduces
    co-channel interference by 40-60% compared to naive assignment
    (Aardal et al., 2007).

References:
    Aardal, K., van Hoesel, S., Koster, A., Mannino, C. & Sassano, A.
    (2007). Models and solution techniques for frequency assignment problems.
    Annals of Operations Research, 153(1), 79-129.
    https://doi.org/10.1007/s10479-007-0178-0

    Mathar, R. & Niessen, T. (2000). Optimum positioning of base stations
    for cellular radio networks. Wireless Networks, 6(6), 421-428.
    https://doi.org/10.1023/A:1019263308849
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

# 15 candidate tower sites
TOWER_SITES = [
    {"name": "Downtown Tower A",    "coords": (50, 50), "radius": 8,  "cost": 120_000, "type": "urban"},
    {"name": "Downtown Tower B",    "coords": (55, 45), "radius": 8,  "cost": 115_000, "type": "urban"},
    {"name": "Financial District",  "coords": (60, 55), "radius": 7,  "cost": 130_000, "type": "urban"},
    {"name": "University Campus",   "coords": (35, 60), "radius": 10, "cost": 90_000,  "type": "suburban"},
    {"name": "Hospital Complex",    "coords": (45, 70), "radius": 9,  "cost": 95_000,  "type": "suburban"},
    {"name": "Shopping Mall",       "coords": (65, 40), "radius": 10, "cost": 85_000,  "type": "suburban"},
    {"name": "Industrial Park",     "coords": (75, 30), "radius": 12, "cost": 70_000,  "type": "industrial"},
    {"name": "Airport Area",        "coords": (80, 60), "radius": 15, "cost": 100_000, "type": "suburban"},
    {"name": "North Residential",   "coords": (40, 85), "radius": 12, "cost": 65_000,  "type": "suburban"},
    {"name": "South Gateway",       "coords": (50, 15), "radius": 12, "cost": 60_000,  "type": "suburban"},
    {"name": "West Hills",          "coords": (15, 50), "radius": 14, "cost": 55_000,  "type": "rural"},
    {"name": "East Valley",         "coords": (85, 50), "radius": 14, "cost": 55_000,  "type": "rural"},
    {"name": "Tech Park",           "coords": (70, 65), "radius": 10, "cost": 80_000,  "type": "suburban"},
    {"name": "Stadium Area",        "coords": (30, 35), "radius": 11, "cost": 75_000,  "type": "suburban"},
    {"name": "Port District",       "coords": (90, 25), "radius": 13, "cost": 50_000,  "type": "industrial"},
]


def _generate_demand_zones(n_zones: int = 40, seed: int = 42) -> list[dict]:
    """Generate demand zones spread across the metro area."""
    rng = np.random.default_rng(seed)
    zones = []
    for i in range(n_zones):
        x = rng.uniform(5, 95)
        y = rng.uniform(5, 95)
        zones.append({"id": i, "coords": (x, y)})
    return zones


def create_coverage_instance(seed: int = 42) -> dict:
    """Create a cell tower coverage (set covering) instance.

    Returns:
        Dictionary with coverage data, tower info, and the set covering
        problem definition.
    """
    zones = _generate_demand_zones(n_zones=40, seed=seed)
    n_towers = len(TOWER_SITES)
    n_zones = len(zones)

    # Determine which zones each tower covers
    subsets = []
    for tower in TOWER_SITES:
        tx, ty = tower["coords"]
        covered = set()
        for zone in zones:
            zx, zy = zone["coords"]
            dist = np.sqrt((tx - zx) ** 2 + (ty - zy) ** 2)
            if dist <= tower["radius"]:
                covered.add(zone["id"])
        subsets.append(covered)

    costs = np.array([t["cost"] for t in TOWER_SITES], dtype=float)

    return {
        "m": n_zones,
        "n": n_towers,
        "subsets": subsets,
        "costs": costs,
        "towers": TOWER_SITES,
        "zones": zones,
    }


def create_interference_graph(selected_towers: list[int]) -> dict:
    """Create an interference graph for frequency assignment.

    Two selected towers interfere if their coverage areas overlap
    (i.e., distance < sum of radii * overlap_factor).

    Args:
        selected_towers: Indices of towers selected from set covering.

    Returns:
        Dictionary with graph coloring instance data.
    """
    overlap_factor = 0.8  # towers interfere if closer than 80% of combined radii
    edges = []

    for i, ti in enumerate(selected_towers):
        for j, tj in enumerate(selected_towers):
            if i >= j:
                continue
            t1 = TOWER_SITES[ti]
            t2 = TOWER_SITES[tj]
            dist = np.sqrt(
                (t1["coords"][0] - t2["coords"][0]) ** 2
                + (t1["coords"][1] - t2["coords"][1]) ** 2
            )
            if dist < (t1["radius"] + t2["radius"]) * overlap_factor:
                edges.append((i, j))

    return {
        "n_vertices": len(selected_towers),
        "edges": edges,
        "tower_indices": selected_towers,
    }


def solve_telecom_network(seed: int = 42, verbose: bool = True) -> dict:
    """Solve the telecom network design problem.

    Phase 1: Set Covering — select tower sites.
    Phase 2: Graph Coloring — assign frequencies to selected towers.

    Returns:
        Dictionary with coverage and frequency assignment results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sc_dir = os.path.join(base_dir, "problems", "combinatorial", "set_covering")
    gc_dir = os.path.join(base_dir, "problems", "combinatorial", "graph_coloring")

    sc_inst_mod = _load_mod(
        "sc_inst_app", os.path.join(sc_dir, "instance.py")
    )
    sc_heur_mod = _load_mod(
        "sc_heur_app", os.path.join(sc_dir, "heuristics", "greedy_scp.py")
    )
    gc_inst_mod = _load_mod(
        "gc_inst_app", os.path.join(gc_dir, "instance.py")
    )
    gc_heur_mod = _load_mod(
        "gc_heur_app", os.path.join(gc_dir, "heuristics", "greedy_coloring.py")
    )

    results = {}

    # ── Phase 1: Tower Placement (Set Covering) ─────────────────────────
    data = create_coverage_instance(seed=seed)

    sc_instance = sc_inst_mod.SetCoveringInstance(
        m=data["m"],
        n=data["n"],
        subsets=data["subsets"],
        costs=data["costs"],
    )

    coverage_results = {}

    # Greedy cost-effectiveness
    ce_sol = sc_heur_mod.greedy_cost_effectiveness(sc_instance)
    coverage_results["Greedy-CE"] = {
        "selected": ce_sol.selected,
        "total_cost": ce_sol.total_cost,
        "n_towers": ce_sol.n_selected,
    }

    # Greedy largest-first
    lf_sol = sc_heur_mod.greedy_largest_first(sc_instance)
    coverage_results["Greedy-LF"] = {
        "selected": lf_sol.selected,
        "total_cost": lf_sol.total_cost,
        "n_towers": lf_sol.n_selected,
    }

    results["coverage"] = coverage_results

    # ── Phase 2: Frequency Assignment (Graph Coloring) ───────────────────
    # Use the best coverage solution
    best_method = min(coverage_results, key=lambda k: coverage_results[k]["total_cost"])
    selected = coverage_results[best_method]["selected"]

    ig_data = create_interference_graph(selected)

    gc_instance = gc_inst_mod.GraphColoringInstance(
        n_vertices=ig_data["n_vertices"],
        edges=ig_data["edges"],
    )

    freq_results = {}

    # DSatur (best greedy coloring)
    dsatur_sol = gc_heur_mod.dsatur(gc_instance)
    freq_results["DSatur"] = {
        "n_frequencies": dsatur_sol.n_colors,
        "assignment": dsatur_sol.colors,
        "valid": dsatur_sol.is_valid,
    }

    # Greedy largest-first
    glf_sol = gc_heur_mod.greedy_largest_first(gc_instance)
    freq_results["Greedy-LF"] = {
        "n_frequencies": glf_sol.n_colors,
        "assignment": glf_sol.colors,
        "valid": glf_sol.is_valid,
    }

    results["frequency"] = freq_results
    results["selected_towers"] = selected
    results["interference_edges"] = ig_data["edges"]

    if verbose:
        print("=" * 70)
        print("5G CELL TOWER NETWORK DESIGN")
        print(f"  {data['n']} candidate sites, {data['m']} demand zones")
        print("=" * 70)

        print("\n--- Phase 1: Tower Placement (Set Covering) ---")
        for method, res in coverage_results.items():
            tower_names = [TOWER_SITES[i]["name"] for i in res["selected"]]
            print(f"\n  {method}: {res['n_towers']} towers, "
                  f"cost=${res['total_cost']:,.0f}")
            for name in tower_names:
                print(f"    - {name}")

        print(f"\n--- Phase 2: Frequency Assignment (using {best_method}) ---")
        freq_bands = ["Band A (3.5 GHz)", "Band B (3.7 GHz)",
                      "Band C (3.9 GHz)", "Band D (4.1 GHz)",
                      "Band E (4.3 GHz)"]
        for method, res in freq_results.items():
            print(f"\n  {method}: {res['n_frequencies']} frequency bands needed")
            print(f"    Valid (no interference): {res['valid']}")
            for i, color in enumerate(res["assignment"]):
                tower_name = TOWER_SITES[selected[i]]["name"]
                band = freq_bands[color] if color < len(freq_bands) else f"Band {color+1}"
                print(f"    {tower_name:25s} -> {band}")

    return results


if __name__ == "__main__":
    solve_telecom_network()

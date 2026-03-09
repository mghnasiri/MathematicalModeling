"""
Real-World Application: Hospital Emergency Network Optimization.

Domain: Healthcare systems engineering / Hospital network design
Model: Max Flow + Shortest Path + Minimum Spanning Tree

Scenario:
    A regional health authority manages a network of 8 facilities
    (hospitals, trauma centers, clinics) connected by patient transfer
    corridors. The network must handle patient surge scenarios such
    as mass casualty incidents (MCI) or pandemic overflow.

    Problem 1 (Max Flow): During an MCI, patients arrive at the Level I
    Trauma Center (source). Determine the maximum patient throughput
    rate (patients/hour) to distribute across the network to receiving
    hospitals (sink = aggregate discharge capacity).

    Problem 2 (Shortest Path): Find the fastest patient transfer route
    from the trauma center to each specialty hospital, considering
    transport time and handoff delays.

    Problem 3 (MST): Design the minimum-cost telemedicine backbone
    connecting all facilities for real-time consultation during
    emergencies. The MST represents the cheapest network that keeps
    all facilities connected.

Real-world considerations modeled:
    - Facility capacity constraints (beds, staff, equipment)
    - Transfer corridor capacities (ambulance availability, road access)
    - Multi-hop patient transfers (overflow chains)
    - Min-cut analysis identifies system bottlenecks
    - Network resilience planning

Industry context:
    Hospital networks handle 400+ mass casualty incidents annually in
    the US (FEMA, 2020). Network flow models can increase surge capacity
    utilization by 25-40% compared to ad-hoc transfers (Paul et al.,
    2006). Telemedicine networks reduce specialist response time by
    50-70% in rural trauma care (Duchesne et al., 2008).

References:
    Paul, J.A., George, S.K., Yi, P. & Lin, L. (2006). Transient
    modeling in simulation of hospital operations for emergency response.
    Prehospital and Disaster Medicine, 21(4), 223-236.
    https://doi.org/10.1017/S1049023X00003769

    Duchesne, J.C., Kyle, A., Simmons, J., Islam, S., Schmieg, R.E.,
    Olivier, J. & McSwain, N.E. (2008). Impact of telemedicine upon
    rural trauma care. Journal of Trauma, 64(1), 92-98.
    https://doi.org/10.1097/TA.0b013e31815dd4c4

    Dean, M.D. & Nair, S.K. (2014). Mass-casualty triage: Distribution
    of victims to multiple hospitals using the SAVE model. European
    Journal of Operational Research, 238(1), 363-373.
    https://doi.org/10.1016/j.ejor.2014.03.028
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

FACILITIES = [
    {"id": 0, "name": "Level I Trauma Center",     "type": "trauma",    "beds": 80,  "coords": (50, 50)},
    {"id": 1, "name": "Regional General Hospital",  "type": "general",   "beds": 120, "coords": (30, 70)},
    {"id": 2, "name": "Cardiac Specialty Center",   "type": "cardiac",   "beds": 45,  "coords": (70, 75)},
    {"id": 3, "name": "Community Hospital North",   "type": "community", "beds": 60,  "coords": (45, 90)},
    {"id": 4, "name": "Pediatric Hospital",         "type": "pediatric", "beds": 50,  "coords": (25, 45)},
    {"id": 5, "name": "Rehabilitation Center",      "type": "rehab",     "beds": 40,  "coords": (75, 35)},
    {"id": 6, "name": "Community Hospital South",   "type": "community", "beds": 55,  "coords": (55, 20)},
    {"id": 7, "name": "Regional Discharge Hub",     "type": "hub",       "beds": 200, "coords": (80, 60)},
]

# Transfer corridors: (from, to, capacity_patients_per_hour)
TRANSFER_LINKS = [
    (0, 1, 15),  # Trauma → General
    (0, 2, 12),  # Trauma → Cardiac
    (0, 4, 8),   # Trauma → Pediatric
    (1, 3, 10),  # General → Community North
    (1, 7, 18),  # General → Hub
    (2, 5, 7),   # Cardiac → Rehab
    (2, 7, 14),  # Cardiac → Hub
    (3, 7, 9),   # Community North → Hub
    (4, 1, 6),   # Pediatric → General
    (4, 6, 5),   # Pediatric → Community South
    (5, 7, 10),  # Rehab → Hub
    (6, 7, 8),   # Community South → Hub
    (0, 6, 4),   # Trauma → Community South (backup route)
]

# Telemedicine connection costs ($K for fiber/equipment)
TELEHEALTH_COSTS = [
    (0, 1, 45),  (0, 2, 55),  (0, 3, 70),  (0, 4, 35),
    (1, 2, 50),  (1, 3, 30),  (1, 4, 40),  (1, 7, 60),
    (2, 3, 65),  (2, 5, 45),  (2, 7, 35),
    (3, 7, 50),
    (4, 5, 55),  (4, 6, 30),
    (5, 6, 25),  (5, 7, 40),
    (6, 7, 45),
    (0, 5, 60),  (0, 6, 50),  (0, 7, 75),
    (1, 5, 55),  (1, 6, 48),
    (3, 4, 58),  (3, 5, 62),  (3, 6, 52),
    (2, 4, 70),  (2, 6, 58),
    (4, 7, 65),
]


def create_emergency_network() -> dict:
    """Create hospital emergency network data.

    Returns:
        Dictionary with max flow, shortest path, and MST data.
    """
    n = len(FACILITIES)

    # Transfer time matrix (minutes) based on distance
    coords = np.array([f["coords"] for f in FACILITIES], dtype=float)
    transfer_time = np.full((n, n), np.inf)
    for i in range(n):
        transfer_time[i][i] = 0

    # Build from transfer links: time ≈ distance * 1.5 (road factor) + handoff
    for u, v, cap in TRANSFER_LINKS:
        dist = np.sqrt(np.sum((coords[u] - coords[v]) ** 2))
        time_min = dist * 1.5 + 5  # 5 min handoff
        transfer_time[u][v] = time_min

    return {
        "n": n,
        "facilities": FACILITIES,
        "transfer_links": TRANSFER_LINKS,
        "telehealth_costs": TELEHEALTH_COSTS,
        "transfer_time": transfer_time,
        "coords": coords,
    }


def solve_emergency_network(verbose: bool = True) -> dict:
    """Solve hospital emergency network optimization.

    Returns:
        Dictionary with max flow, shortest path, and MST results.
    """
    data = create_emergency_network()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    results = {}

    # ── Problem 1: Max Flow (patient throughput) ─────────────────────────
    mf_inst = _load_mod("mf_inst_en", os.path.join(loc_dir, "max_flow", "instance.py"))
    mf_ek = _load_mod("mf_ek_en", os.path.join(loc_dir, "max_flow", "exact", "edmonds_karp.py"))

    mf_instance = mf_inst.MaxFlowInstance.from_edges(
        n=data["n"],
        source=0,  # Trauma Center
        sink=7,    # Discharge Hub
        edges=[(u, v, float(cap)) for u, v, cap in data["transfer_links"]],
        name="emergency_throughput",
    )

    mf_sol = mf_ek.edmonds_karp(mf_instance)
    s_set, t_set = mf_sol.min_cut

    results["max_flow"] = {
        "max_throughput": mf_sol.max_flow,
        "flow_matrix": mf_sol.flow_matrix,
        "min_cut": (s_set, t_set),
    }

    # ── Problem 2: Shortest Path (fastest transfer routes) ───────────────
    sp_inst = _load_mod("sp_inst_en", os.path.join(loc_dir, "shortest_path", "instance.py"))
    sp_dij = _load_mod("sp_dij_en", os.path.join(loc_dir, "shortest_path", "exact", "dijkstra.py"))

    # Build edges from transfer time matrix
    sp_edges = []
    for u, v, _ in data["transfer_links"]:
        time_min = data["transfer_time"][u][v]
        sp_edges.append((u, v, time_min))

    sp_instance = sp_inst.ShortestPathInstance.from_edges(
        n=data["n"],
        edges=sp_edges,
        name="transfer_routes",
    )

    # Find shortest path from trauma center to all reachable facilities
    sp_results = {}
    for target in [1, 2, 3, 5, 6, 7]:  # skip source (0) and pediatric direct
        sol = sp_dij.dijkstra(sp_instance, source=0, target=target)
        sp_results[target] = {
            "path": sol.path,
            "distance": sol.distance,
        }

    results["shortest_path"] = sp_results

    # ── Problem 3: MST (telemedicine backbone) ───────────────────────────
    mst_inst = _load_mod("mst_inst_en", os.path.join(loc_dir, "min_spanning_tree", "instance.py"))
    mst_alg = _load_mod("mst_alg_en", os.path.join(loc_dir, "min_spanning_tree", "exact", "mst_algorithms.py"))

    mst_instance = mst_inst.MSTInstance.from_edges(
        n=data["n"],
        edges=[(u, v, float(cost)) for u, v, cost in data["telehealth_costs"]],
        name="telemedicine_backbone",
    )

    kruskal_sol = mst_alg.kruskal(mst_instance)
    prim_sol = mst_alg.prim(mst_instance)

    results["mst"] = {
        "Kruskal": {
            "total_cost": kruskal_sol.total_weight,
            "backbone_links": kruskal_sol.tree_edges,
        },
        "Prim": {
            "total_cost": prim_sol.total_weight,
            "backbone_links": prim_sol.tree_edges,
        },
    }

    if verbose:
        total_beds = sum(f["beds"] for f in FACILITIES)
        print("=" * 70)
        print("HOSPITAL EMERGENCY NETWORK OPTIMIZATION")
        print(f"  {data['n']} facilities, {total_beds} total beds")
        print("=" * 70)

        # Facility list
        print("\n  Facility network:")
        for f in FACILITIES:
            print(f"    [{f['id']}] {f['name']:30s} ({f['type']:10s}, {f['beds']:3d} beds)")

        # Max Flow
        print(f"\n--- Max Patient Throughput (Trauma → Hub) ---")
        print(f"  Maximum throughput: {results['max_flow']['max_throughput']:.0f} patients/hour")
        print(f"  Min-cut S-side: {[FACILITIES[i]['name'][:20] for i in s_set]}")
        print(f"  Min-cut T-side: {[FACILITIES[i]['name'][:20] for i in t_set]}")

        # Active flows
        print("  Active transfers:")
        fm = results["max_flow"]["flow_matrix"]
        for u, v, cap in data["transfer_links"]:
            flow = fm[u][v]
            if flow > 0:
                util = flow / cap * 100
                print(f"    {FACILITIES[u]['name'][:20]:20s} → "
                      f"{FACILITIES[v]['name'][:20]:20s}: "
                      f"{flow:.0f}/{cap} pts/hr ({util:.0f}%)")

        # Shortest paths
        print(f"\n--- Fastest Transfer Routes (from Trauma Center) ---")
        for target, sp_res in results["shortest_path"].items():
            if sp_res["distance"] < np.inf:
                path_names = [FACILITIES[i]["name"][:15] for i in sp_res["path"]]
                print(f"  → {FACILITIES[target]['name']:30s}: "
                      f"{sp_res['distance']:.1f} min via [{' → '.join(path_names)}]")

        # MST
        best_mst = "Kruskal"
        mst_res = results["mst"][best_mst]
        print(f"\n--- Telemedicine Backbone (MST) ---")
        print(f"  Total infrastructure cost: ${mst_res['total_cost']:.0f}K")
        print(f"  Backbone links ({len(mst_res['backbone_links'])} connections):")
        for u, v, cost in mst_res["backbone_links"]:
            print(f"    {FACILITIES[u]['name'][:25]:25s} ↔ "
                  f"{FACILITIES[v]['name'][:25]:25s}: ${cost:.0f}K")

    return results


if __name__ == "__main__":
    solve_emergency_network()

"""
Real-World Application: Agricultural Irrigation Network Design.

Domain: Precision agriculture / Water resource management
Models: MST + Max Flow + Shortest Path

Scenario:
    A large farm with 10 nodes designs an irrigation pipe network.
    The farm includes a water pump station (source), main distribution
    junctions, five crop field zones, and a reservoir/collection point
    (sink). Approximately 15 potential pipe segments connect these
    nodes with varying installation costs, water capacities, and
    distances.

    Questions answered:
    1. MST: What is the minimum-cost pipe network backbone to connect
       all nodes? Solved with both Kruskal and Prim algorithms.
    2. Max Flow: What is the maximum water throughput from the pump
       station through the network to the reservoir? Where are the
       bottleneck pipes (min-cut)?
    3. Shortest Path: What is the fastest water delivery path from
       the pump station to each field zone (Dijkstra)?

Real-world considerations modeled:
    - Pipe installation cost proportional to distance and terrain
      difficulty (trenching through rocky vs soft soil)
    - Pipe capacity in liters/hour based on diameter and pressure
    - Gravity-fed vs pumped segments affecting flow direction
    - Bottleneck identification via min-cut for capacity upgrades
    - Minimum-cost connectivity for infrastructure budgeting

Industry context:
    Irrigation accounts for approximately 70% of global freshwater
    withdrawals (FAO, 2020). Network optimization can reduce water
    losses by 20-40% compared to ad-hoc pipe layouts. The MST
    backbone minimizes trenching and pipe material costs, while
    max-flow analysis identifies hydraulic bottlenecks limiting
    crop water delivery during peak demand.

References:
    Valiantzas, J.D. (2006). Simplified versions for the Penman
    evapotranspiration equation using routine weather data. Journal
    of Hydrology, 331(3-4), 690-702.
    https://doi.org/10.1016/j.jhydrol.2006.06.012

    Kang, Y. & Nishiyama, S. (1996). Analysis and design of
    microirrigation laterals. Journal of Irrigation and Drainage
    Engineering, 122(2), 75-82.
    https://doi.org/10.1061/(ASCE)0733-9437(1996)122:2(75)

    Planells, P., Carrion, P., Ortega, J.F., Moreno, M.A. &
    Tarjuelo, J.M. (2005). Pumping selection and regulation for
    water-distribution networks. Journal of Irrigation and Drainage
    Engineering, 131(3), 273-281.
    https://doi.org/10.1061/(ASCE)0733-9437(2005)131:3(273)
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


# -- Domain Data --------------------------------------------------------------

NODES = [
    {"id": 0, "name": "Water Pump Station",       "type": "source",     "crop": None},
    {"id": 1, "name": "Main Junction Alpha",      "type": "junction",   "crop": None},
    {"id": 2, "name": "Main Junction Beta",       "type": "junction",   "crop": None},
    {"id": 3, "name": "Main Junction Gamma",      "type": "junction",   "crop": None},
    {"id": 4, "name": "Corn Field Zone",           "type": "field",      "crop": "corn"},
    {"id": 5, "name": "Wheat Field Zone",          "type": "field",      "crop": "wheat"},
    {"id": 6, "name": "Soybean Field Zone",        "type": "field",      "crop": "soybeans"},
    {"id": 7, "name": "Vegetable Field Zone",      "type": "field",      "crop": "vegetables"},
    {"id": 8, "name": "Orchard Field Zone",        "type": "field",      "crop": "orchard"},
    {"id": 9, "name": "Reservoir Collection Pt",   "type": "sink",       "crop": None},
]

# Pipe segments: (from, to, install_cost_usd, capacity_liters_per_hr, distance_m)
PIPE_SEGMENTS = [
    (0, 1, 4500,  8000, 150),   # Pump → Junction Alpha (main trunk)
    (0, 2, 5200,  7000, 200),   # Pump → Junction Beta (main trunk)
    (0, 3, 6000,  5000, 250),   # Pump → Junction Gamma (longer route)
    (1, 2, 1800,  3000,  80),   # Alpha ↔ Beta cross-link
    (1, 4, 3200,  4500, 120),   # Alpha → Corn Field
    (1, 5, 3500,  4000, 140),   # Alpha → Wheat Field
    (2, 5, 2800,  3500, 100),   # Beta → Wheat Field
    (2, 6, 3000,  4000, 110),   # Beta → Soybean Field
    (2, 7, 3400,  3500, 130),   # Beta → Vegetable Field
    (3, 7, 2600,  3000,  90),   # Gamma → Vegetable Field
    (3, 8, 3800,  4500, 160),   # Gamma → Orchard Field
    (4, 9, 2200,  3000,  70),   # Corn Field → Reservoir (drainage)
    (5, 9, 2500,  2500,  85),   # Wheat Field → Reservoir (drainage)
    (6, 9, 2000,  2800,  65),   # Soybean Field → Reservoir (drainage)
    (7, 9, 2700,  2000,  95),   # Vegetable Field → Reservoir (drainage)
    (8, 9, 3100,  3500, 120),   # Orchard Field → Reservoir (drainage)
]

# Water demand per field zone (liters/hour during peak irrigation)
FIELD_DEMAND = {
    4: 3500,   # Corn: high water demand
    5: 2800,   # Wheat: moderate demand
    6: 2200,   # Soybeans: moderate demand
    7: 1800,   # Vegetables: lower but frequent
    8: 3000,   # Orchard: high demand (deep roots)
}


def create_irrigation_network() -> dict:
    """Create irrigation network data for all three analyses.

    Returns:
        Dictionary with network data for MST, Max Flow, and Shortest Path.
    """
    n = len(NODES)

    # For max flow: directed edges with capacities (liters/hr)
    flow_edges = [(f, t, float(cap)) for f, t, _, cap, _ in PIPE_SEGMENTS]

    # For shortest path: directed edges with distances (meters)
    sp_edges = [(f, t, float(dist)) for f, t, _, _, dist in PIPE_SEGMENTS]

    # For MST: undirected edges with installation costs
    mst_edges = []
    seen = set()
    for f, t, cost, _, _ in PIPE_SEGMENTS:
        key = (min(f, t), max(f, t))
        if key not in seen:
            mst_edges.append((f, t, float(cost)))
            seen.add(key)

    return {
        "n": n,
        "nodes": NODES,
        "flow_edges": flow_edges,
        "sp_edges": sp_edges,
        "mst_edges": mst_edges,
        "pipe_segments": PIPE_SEGMENTS,
    }


def solve_irrigation_network(verbose: bool = True) -> dict:
    """Analyze irrigation network using MST, Max Flow, and Shortest Path.

    Args:
        verbose: If True, print detailed analysis to stdout.

    Returns:
        Dictionary with results from all three analyses.
    """
    data = create_irrigation_network()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    results = {}

    # -- MST Analysis (minimum-cost pipe backbone) -----------------------------
    mst_inst_mod = _load_mod(
        "mst_inst_ag", os.path.join(loc_dir, "min_spanning_tree", "instance.py")
    )
    mst_alg_mod = _load_mod(
        "mst_alg_ag",
        os.path.join(loc_dir, "min_spanning_tree", "exact", "mst_algorithms.py"),
    )

    mst_instance = mst_inst_mod.MSTInstance.from_edges(
        data["n"], data["mst_edges"], name="irrigation_backbone",
    )

    kruskal_sol = mst_alg_mod.kruskal(mst_instance)
    prim_sol = mst_alg_mod.prim(mst_instance)

    results["mst"] = {
        "Kruskal": {
            "total_cost": kruskal_sol.total_weight,
            "backbone_pipes": kruskal_sol.tree_edges,
        },
        "Prim": {
            "total_cost": prim_sol.total_weight,
            "backbone_pipes": prim_sol.tree_edges,
        },
    }

    # -- Max Flow Analysis (water throughput pump → reservoir) -----------------
    mf_inst_mod = _load_mod(
        "mf_inst_ag", os.path.join(loc_dir, "max_flow", "instance.py")
    )
    mf_ek_mod = _load_mod(
        "mf_ek_ag", os.path.join(loc_dir, "max_flow", "exact", "edmonds_karp.py")
    )

    mf_instance = mf_inst_mod.MaxFlowInstance.from_edges(
        n=data["n"], source=0, sink=9,
        edges=data["flow_edges"], name="irrigation_flow",
    )

    mf_sol = mf_ek_mod.edmonds_karp(mf_instance)
    s_set, t_set = mf_sol.min_cut

    results["max_flow"] = {
        "max_throughput": mf_sol.max_flow,
        "flow_matrix": mf_sol.flow_matrix,
        "min_cut": (s_set, t_set),
    }

    # -- Shortest Path Analysis (fastest delivery to each field) ---------------
    sp_inst_mod = _load_mod(
        "sp_inst_ag", os.path.join(loc_dir, "shortest_path", "instance.py")
    )
    sp_dj_mod = _load_mod(
        "sp_dj_ag",
        os.path.join(loc_dir, "shortest_path", "exact", "dijkstra.py"),
    )

    sp_instance = sp_inst_mod.ShortestPathInstance.from_edges(
        n=data["n"], edges=data["sp_edges"], name="irrigation_delivery",
    )

    sp_results = {}
    for target in [4, 5, 6, 7, 8]:  # field zones only
        sol = sp_dj_mod.dijkstra(sp_instance, source=0, target=target)
        sp_results[target] = {
            "path": sol.path,
            "distance": sol.distance,
        }

    results["shortest_path"] = sp_results

    if verbose:
        print("=" * 70)
        print("AGRICULTURAL IRRIGATION NETWORK DESIGN")
        print(f"  {data['n']} nodes, {len(data['pipe_segments'])} pipe segments")
        print("=" * 70)

        # Node inventory
        print("\n  Network nodes:")
        for nd in NODES:
            crop_str = f", crop={nd['crop']}" if nd["crop"] else ""
            print(f"    [{nd['id']}] {nd['name']:30s} ({nd['type']}{crop_str})")

        # MST
        mst_res = results["mst"]["Kruskal"]
        print(f"\n--- 1. MINIMUM-COST PIPE BACKBONE (MST) ---")
        print(f"  Algorithm: Kruskal (verified by Prim)")
        print(f"  Total installation cost: ${mst_res['total_cost']:,.0f}")
        print(f"  Backbone pipes ({len(mst_res['backbone_pipes'])} segments):")
        for u, v, cost in mst_res["backbone_pipes"]:
            print(f"    {NODES[u]['name']:30s} -- {NODES[v]['name']:30s}: "
                  f"${cost:,.0f}")
        prim_cost = results["mst"]["Prim"]["total_cost"]
        print(f"  Prim verification: ${prim_cost:,.0f} "
              f"({'match' if abs(prim_cost - mst_res['total_cost']) < 0.01 else 'MISMATCH'})")

        # Max Flow
        print(f"\n--- 2. MAXIMUM WATER THROUGHPUT (Max Flow) ---")
        print(f"  Pump Station → Reservoir: "
              f"{results['max_flow']['max_throughput']:,.0f} liters/hour")
        print(f"  Total field demand: "
              f"{sum(FIELD_DEMAND.values()):,} liters/hour")
        print("  Flow on each pipe:")
        for f, t, cost, cap, dist in PIPE_SEGMENTS:
            flow = mf_sol.flow_matrix[f][t]
            if flow > 0:
                utilization = flow / cap * 100
                bar = "#" * int(utilization / 5) + "." * (20 - int(utilization / 5))
                print(f"    {NODES[f]['name'][:25]:25s} -> "
                      f"{NODES[t]['name'][:25]:25s}: "
                      f"{flow:6,.0f}/{cap:,} L/hr [{bar}] {utilization:.0f}%")

        print(f"\n  BOTTLENECK (Min-Cut):")
        print(f"    Source side: {[NODES[i]['name'] for i in s_set]}")
        print(f"    Sink side:   {[NODES[i]['name'] for i in t_set]}")
        print("    Saturated pipes crossing the cut:")
        for u in s_set:
            for v in t_set:
                if mf_instance.capacity_matrix[u][v] > 0:
                    print(f"      {NODES[u]['name']} -> {NODES[v]['name']}: "
                          f"capacity = {mf_instance.capacity_matrix[u][v]:,.0f} L/hr")

        # Shortest Path
        print(f"\n--- 3. FASTEST WATER DELIVERY (Shortest Path) ---")
        print(f"  From: {NODES[0]['name']}")
        for target, sp_res in results["shortest_path"].items():
            if sp_res["distance"] < float("inf"):
                path_names = [NODES[i]["name"][:18] for i in sp_res["path"]]
                print(f"  -> {NODES[target]['name']:30s}: "
                      f"{sp_res['distance']:6.0f} m via "
                      f"[{' -> '.join(path_names)}]")
            else:
                print(f"  -> {NODES[target]['name']:30s}: UNREACHABLE")

    return results


if __name__ == "__main__":
    solve_irrigation_network()

"""
Real-World Application: Supply Chain Network Design.

Domain: Manufacturing supply chain / Telecommunications backbone
Models: Max Flow + MST + Shortest Path

Scenario:
    A manufacturer has a raw material supplier (source), 3 intermediate
    processing plants, 2 regional distribution centers, and a major
    retail hub (sink). Transport links between nodes have limited
    throughput capacity (trucks/day).

    Questions answered:
    1. Max Flow: What is the maximum daily throughput from supplier to
       retail hub? Where are the bottlenecks (min-cut)?
    2. MST: What is the minimum-cost backbone network to connect all
       facilities (e.g., for a private data network)?
    3. Shortest Path: What is the fastest route from supplier to hub
       for express shipments?

Real-world considerations modeled:
    - Heterogeneous link capacities (highway vs local roads)
    - Bottleneck identification via min-cut analysis
    - Infrastructure cost minimization (MST for network backbone)
    - Express routing for time-sensitive shipments

Industry context:
    Supply chain network design is typically a strategic decision
    involving billions of dollars. Max-flow analysis identifies
    bottlenecks that constrain throughput, while MST provides the
    minimum-cost connectivity backbone (Simchi-Levi et al., 2014).

References:
    Simchi-Levi, D., Kaminsky, P. & Simchi-Levi, E. (2014). Designing
    and Managing the Supply Chain. 3rd ed. McGraw-Hill.

    Magnanti, T.L. & Wong, R.T. (1984). Network design and
    transportation planning: Models and algorithms. Transportation
    Science, 18(1), 1-55.
    https://doi.org/10.1287/trsc.18.1.1
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
    "Raw Material Supplier",   # 0 (source)
    "Processing Plant Alpha",  # 1
    "Processing Plant Beta",   # 2
    "Processing Plant Gamma",  # 3
    "Distribution Center East",# 4
    "Distribution Center West",# 5
    "Retail Hub",              # 6 (sink)
]

# Transport links with capacity (trucks/day) and cost ($/truck)
TRANSPORT_LINKS = [
    # (from, to, capacity_trucks_per_day, cost_per_truck, distance_km)
    (0, 1, 50, 800, 120),   # Supplier → Alpha
    (0, 2, 40, 650, 95),    # Supplier → Beta
    (0, 3, 30, 900, 150),   # Supplier → Gamma
    (1, 2, 15, 300, 45),    # Alpha → Beta (cross-link)
    (1, 4, 35, 550, 80),    # Alpha → DC-East
    (2, 4, 25, 500, 70),    # Beta → DC-East
    (2, 5, 30, 600, 90),    # Beta → DC-West
    (3, 5, 25, 700, 110),   # Gamma → DC-West
    (3, 2, 10, 400, 60),    # Gamma → Beta (cross-link)
    (4, 6, 45, 450, 65),    # DC-East → Hub
    (5, 6, 40, 500, 75),    # DC-West → Hub
    (4, 5, 10, 250, 35),    # DC-East → DC-West (cross-link)
]


def create_supply_chain_data() -> dict:
    """Create supply chain network data.

    Returns:
        Dictionary with network data for all three analyses.
    """
    n = len(FACILITIES)

    # For max flow: directed edges with capacities
    flow_edges = [(f, t, cap) for f, t, cap, _, _ in TRANSPORT_LINKS]

    # For shortest path: directed edges with distances
    sp_edges = [(f, t, float(dist)) for f, t, _, _, dist in TRANSPORT_LINKS]

    # For MST: undirected edges with costs (backbone network)
    mst_edges = []
    seen = set()
    for f, t, _, cost, _ in TRANSPORT_LINKS:
        key = (min(f, t), max(f, t))
        if key not in seen:
            mst_edges.append((f, t, float(cost)))
            seen.add(key)

    return {
        "n": n,
        "facilities": FACILITIES,
        "flow_edges": flow_edges,
        "sp_edges": sp_edges,
        "mst_edges": mst_edges,
        "transport_links": TRANSPORT_LINKS,
    }


def solve_supply_chain(verbose: bool = True) -> dict:
    """Analyze supply chain network using Max Flow, MST, and Shortest Path.

    Returns:
        Dictionary with analysis results.
    """
    data = create_supply_chain_data()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    results = {}

    # ── Max Flow Analysis ────────────────────────────────────────────────
    mf_inst_mod = _load_mod(
        "mf_inst_sc", os.path.join(loc_dir, "max_flow", "instance.py")
    )
    mf_ek_mod = _load_mod(
        "mf_ek_sc", os.path.join(loc_dir, "max_flow", "exact", "edmonds_karp.py")
    )

    mf_instance = mf_inst_mod.MaxFlowInstance.from_edges(
        data["n"], source=0, sink=6,
        edges=data["flow_edges"], name="supply_chain",
    )

    mf_sol = mf_ek_mod.edmonds_karp(mf_instance)
    results["max_flow"] = {
        "max_throughput": mf_sol.max_flow,
        "flow_matrix": mf_sol.flow_matrix,
        "min_cut": mf_sol.min_cut,
    }

    # ── MST Analysis ─────────────────────────────────────────────────────
    mst_inst_mod = _load_mod(
        "mst_inst_sc", os.path.join(loc_dir, "min_spanning_tree", "instance.py")
    )
    mst_alg_mod = _load_mod(
        "mst_alg_sc",
        os.path.join(loc_dir, "min_spanning_tree", "exact", "mst_algorithms.py"),
    )

    mst_instance = mst_inst_mod.MSTInstance.from_edges(
        data["n"], data["mst_edges"], name="supply_backbone",
    )

    mst_sol = mst_alg_mod.kruskal(mst_instance)
    results["mst"] = {
        "total_cost": mst_sol.total_weight,
        "backbone_links": mst_sol.tree_edges,
    }

    # ── Shortest Path Analysis ───────────────────────────────────────────
    sp_inst_mod = _load_mod(
        "sp_inst_sc", os.path.join(loc_dir, "shortest_path", "instance.py")
    )
    sp_dj_mod = _load_mod(
        "sp_dj_sc",
        os.path.join(loc_dir, "shortest_path", "exact", "dijkstra.py"),
    )

    sp_instance = sp_inst_mod.ShortestPathInstance.from_edges(
        data["n"], data["sp_edges"], name="supply_express",
    )

    sp_sol = sp_dj_mod.dijkstra(sp_instance, source=0, target=6)
    results["shortest_path"] = {
        "distance": sp_sol.distance,
        "path": sp_sol.path,
    }

    if verbose:
        print("=" * 70)
        print("SUPPLY CHAIN NETWORK ANALYSIS")
        print(f"  {data['n']} facilities, {len(data['transport_links'])} links")
        print("=" * 70)

        # Max Flow
        print("\n--- 1. MAXIMUM THROUGHPUT (Max Flow) ---")
        print(f"  Max throughput: {mf_sol.max_flow:.0f} trucks/day")
        print("  Flow on each link:")
        for f, t, cap, cost, dist in TRANSPORT_LINKS:
            flow = mf_sol.flow_matrix[f][t]
            utilization = (flow / cap * 100) if cap > 0 else 0
            bar = "█" * int(utilization / 5) + "░" * (20 - int(utilization / 5))
            print(f"    {FACILITIES[f]:28s} → {FACILITIES[t]:28s}: "
                  f"{flow:5.0f}/{cap} [{bar}] {utilization:.0f}%")

        s_set, t_set = mf_sol.min_cut
        print(f"\n  BOTTLENECK (Min-Cut):")
        print(f"    Source side: {[FACILITIES[i] for i in s_set]}")
        print(f"    Sink side:   {[FACILITIES[i] for i in t_set]}")
        print("    Saturated links crossing the cut:")
        for u in s_set:
            for v in t_set:
                if mf_instance.capacity_matrix[u][v] > 0:
                    print(f"      {FACILITIES[u]} → {FACILITIES[v]}: "
                          f"capacity = {mf_instance.capacity_matrix[u][v]:.0f}")

        # MST
        print("\n--- 2. MINIMUM-COST BACKBONE NETWORK (MST) ---")
        print(f"  Total backbone cost: ${mst_sol.total_weight:,.0f}/truck")
        print("  Selected links:")
        for u, v, w in mst_sol.tree_edges:
            print(f"    {FACILITIES[u]:28s} — {FACILITIES[v]:28s}: ${w:,.0f}")

        # Shortest Path
        print("\n--- 3. EXPRESS SHIPPING ROUTE (Shortest Path) ---")
        path_names = [FACILITIES[i] for i in sp_sol.path]
        print(f"  Shortest route: {sp_sol.distance:.0f} km")
        print(f"  Path: {' → '.join(path_names)}")

    return results


if __name__ == "__main__":
    solve_supply_chain()

"""
Real-World Application: Irrigation Network Design and Capacity Analysis.

Domain: Agricultural water management / Irrigation infrastructure
Model: MST (Minimum Spanning Tree) + Max Flow

Scenario:
    A farm operation has 1 water source (well/pump station) and 8 field
    connection points (irrigation nodes) spread across the property. Two
    optimization problems must be solved:

    Problem 1 (MST): Design the minimum-cost pipe network connecting the
    water source to all field nodes. Pipe installation cost depends on
    distance and terrain. The MST gives the cheapest connected network.

    Problem 2 (Max Flow): Given the pipe network with flow capacities
    (gallons per minute), verify that total water throughput from the
    source to the main collection/distribution manifold meets all field
    demands during peak irrigation periods.

Real-world considerations modeled:
    - Pipe installation cost varies with distance and terrain type
    - Pipe capacity depends on diameter and pressure constraints
    - Peak irrigation demand during hot summer months
    - Gravity-fed vs. pressurized segments
    - Network redundancy for critical crop areas
    - Min-cut analysis identifies system bottlenecks

Industry context:
    Agriculture accounts for 70% of global freshwater withdrawals (FAO,
    2020). Efficient irrigation network design saves 25-40% of water
    compared to unplanned layouts. In the US, irrigated farmland produces
    54% of crop value on only 28% of harvested acreage (USDA, 2019).
    Optimized pipe networks reduce installation costs by 15-30% over
    ad-hoc designs and ensure adequate flow to all zones.

References:
    Keller, J. & Bliesner, R.D. (1990). Sprinkle and Trickle Irrigation.
    Van Nostrand Reinhold, New York.
    https://doi.org/10.1007/978-1-4757-1425-8

    Lamaddalena, N. & Sagardoy, J.A. (2000). Performance Analysis of
    On-Demand Pressurized Irrigation Systems. FAO Irrigation and
    Drainage Paper 59, Rome.

    Valiantzas, J.D. (2005). Modified Hazen-Williams and Darcy-Weisbach
    equations for friction and local head losses along irrigation
    laterals. Journal of Irrigation and Drainage Engineering, 131(4),
    342-350.
    https://doi.org/10.1061/(ASCE)0733-9437(2005)131:4(342)
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


# -- Domain Data ---------------------------------------------------------------

# Node 0 = water source (well/pump station); nodes 1-8 = field connection points
NODES = [
    {"id": 0, "name": "Well Pump Station",       "type": "source",  "coords": (50, 50), "demand_gpm": 0},
    {"id": 1, "name": "North Pivot Center",       "type": "pivot",   "coords": (45, 85), "demand_gpm": 120},
    {"id": 2, "name": "Northeast Drip Zone",      "type": "drip",    "coords": (75, 80), "demand_gpm": 60},
    {"id": 3, "name": "East Sprinkler Block",     "type": "sprink",  "coords": (85, 55), "demand_gpm": 90},
    {"id": 4, "name": "Southeast Flood Basin",    "type": "flood",   "coords": (70, 25), "demand_gpm": 150},
    {"id": 5, "name": "South Drip Vineyard",      "type": "drip",    "coords": (45, 15), "demand_gpm": 45},
    {"id": 6, "name": "Southwest Orchard Block",  "type": "sprink",  "coords": (20, 30), "demand_gpm": 80},
    {"id": 7, "name": "West Pasture Trough",      "type": "trough",  "coords": (15, 60), "demand_gpm": 30},
    {"id": 8, "name": "Distribution Manifold",    "type": "manifold","coords": (50, 50), "demand_gpm": 0},
]

# Potential pipe segments: (from, to, install_cost_$K, capacity_gpm)
# Cost depends on distance, terrain, pipe diameter
PIPE_SEGMENTS = [
    (0, 1, 18, 200),  (0, 2, 22, 150),  (0, 3, 20, 180),
    (0, 7, 12, 100),  (0, 6, 16, 120),
    (1, 2, 15, 100),  (1, 7, 14, 80),
    (2, 3, 12, 90),   (2, 1, 15, 100),
    (3, 4, 14, 130),  (3, 8, 16, 180),
    (4, 5, 10, 100),  (4, 8, 18, 150),
    (5, 6, 11, 70),   (5, 8, 15, 120),
    (6, 7, 13, 80),   (6, 5, 11, 70),
    (7, 1, 14, 80),   (7, 8, 16, 100),
    (0, 8, 5, 500),   # main trunk line from pump to manifold
    (1, 8, 20, 120),  (2, 8, 18, 90),
    (6, 8, 17, 80),   (7, 0, 12, 100),
]

# Edges for MST (undirected, use installation cost as weight)
MST_EDGES = [
    (0, 1, 18.0),  (0, 2, 22.0),  (0, 3, 20.0),  (0, 7, 12.0),
    (0, 6, 16.0),  (0, 5, 19.0),  (0, 4, 21.0),  (0, 8, 5.0),
    (1, 2, 15.0),  (1, 7, 14.0),  (1, 8, 20.0),
    (2, 3, 12.0),  (2, 8, 18.0),
    (3, 4, 14.0),  (3, 8, 16.0),
    (4, 5, 10.0),  (4, 8, 18.0),
    (5, 6, 11.0),  (5, 8, 15.0),
    (6, 7, 13.0),  (6, 8, 17.0),
    (7, 8, 16.0),
]


def create_instance() -> dict:
    """Create irrigation network instance data.

    Returns:
        Dictionary with MST and Max Flow instance data.
    """
    n = len(NODES)
    coords = np.array([nd["coords"] for nd in NODES], dtype=float)

    return {
        "n": n,
        "nodes": NODES,
        "mst_edges": MST_EDGES,
        "pipe_segments": PIPE_SEGMENTS,
        "coords": coords,
    }


def solve(verbose: bool = True) -> dict:
    """Solve irrigation network design using MST and Max Flow.

    Returns:
        Dictionary with MST and max flow results.
    """
    data = create_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    results = {}

    # -- Problem 1: MST (minimum-cost pipe network) ----------------------------
    mst_inst = _load_mod("mst_inst_irr", os.path.join(loc_dir, "min_spanning_tree", "instance.py"))
    mst_alg = _load_mod("mst_alg_irr", os.path.join(loc_dir, "min_spanning_tree", "exact", "mst_algorithms.py"))

    mst_instance = mst_inst.MSTInstance.from_edges(
        n=data["n"],
        edges=data["mst_edges"],
        name="irrigation_network",
    )

    kruskal_sol = mst_alg.kruskal(mst_instance)
    prim_sol = mst_alg.prim(mst_instance)

    results["mst"] = {
        "Kruskal": {
            "total_cost": kruskal_sol.total_weight,
            "tree_edges": kruskal_sol.tree_edges,
        },
        "Prim": {
            "total_cost": prim_sol.total_weight,
            "tree_edges": prim_sol.tree_edges,
        },
    }

    # -- Problem 2: Max Flow (water throughput verification) --------------------
    mf_inst = _load_mod("mf_inst_irr", os.path.join(loc_dir, "max_flow", "instance.py"))
    mf_ek = _load_mod("mf_ek_irr", os.path.join(loc_dir, "max_flow", "exact", "edmonds_karp.py"))

    # Build directed flow network: source=0 (pump), sink=8 (manifold)
    flow_edges = [(u, v, float(cap)) for u, v, cost, cap in data["pipe_segments"]]

    mf_instance = mf_inst.MaxFlowInstance.from_edges(
        n=data["n"],
        source=0,   # Well Pump Station
        sink=8,     # Distribution Manifold
        edges=flow_edges,
        name="irrigation_flow",
    )

    mf_sol = mf_ek.edmonds_karp(mf_instance)
    s_set, t_set = mf_sol.min_cut

    total_demand = sum(nd["demand_gpm"] for nd in NODES)

    results["max_flow"] = {
        "max_throughput_gpm": mf_sol.max_flow,
        "total_demand_gpm": total_demand,
        "demand_met": mf_sol.max_flow >= total_demand,
        "flow_matrix": mf_sol.flow_matrix,
        "min_cut": (s_set, t_set),
    }

    if verbose:
        print("=" * 70)
        print("IRRIGATION NETWORK DESIGN & CAPACITY ANALYSIS")
        print(f"  {data['n']} nodes ({data['n'] - 2} field zones + source + manifold)")
        print(f"  Total peak demand: {total_demand} GPM")
        print("=" * 70)

        # Node summary
        print("\n  Irrigation nodes:")
        for nd in NODES:
            demand_str = f"{nd['demand_gpm']} GPM" if nd["demand_gpm"] > 0 else "---"
            print(f"    [{nd['id']}] {nd['name']:28s} ({nd['type']:8s}) {demand_str:>8s}")

        # MST results
        best_method = "Kruskal"
        mst_res = results["mst"][best_method]
        print(f"\n--- PIPE NETWORK DESIGN (MST — {best_method}) ---")
        print(f"  Total installation cost: ${mst_res['total_cost']:.0f}K")
        print(f"  Pipe segments ({len(mst_res['tree_edges'])} connections):")
        for u, v, cost in mst_res["tree_edges"]:
            print(f"    {NODES[u]['name']:28s} <-> "
                  f"{NODES[v]['name']:28s}: ${cost:.0f}K")

        # Prim comparison
        prim_res = results["mst"]["Prim"]
        print(f"\n  Prim verification: ${prim_res['total_cost']:.0f}K "
              f"({'matches' if abs(prim_res['total_cost'] - mst_res['total_cost']) < 0.01 else 'differs'})")

        # Max Flow results
        mf_res = results["max_flow"]
        print(f"\n--- WATER THROUGHPUT ANALYSIS (Max Flow) ---")
        print(f"  Maximum throughput: {mf_res['max_throughput_gpm']:.0f} GPM")
        print(f"  Total field demand: {mf_res['total_demand_gpm']} GPM")
        status = "SUFFICIENT" if mf_res["demand_met"] else "INSUFFICIENT"
        surplus = mf_res["max_throughput_gpm"] - mf_res["total_demand_gpm"]
        print(f"  Status: {status} (surplus: {surplus:.0f} GPM)")

        # Min-cut bottleneck
        print(f"  Bottleneck (min-cut):")
        print(f"    Source side: {[NODES[i]['name'][:20] for i in s_set]}")
        print(f"    Sink side:   {[NODES[i]['name'][:20] for i in t_set]}")

        # Active flows
        print("  Active pipe flows:")
        fm = mf_res["flow_matrix"]
        for u, v, cost, cap in data["pipe_segments"]:
            flow = fm[u][v]
            if flow > 0:
                util = flow / cap * 100
                print(f"    {NODES[u]['name'][:22]:22s} -> "
                      f"{NODES[v]['name'][:22]:22s}: "
                      f"{flow:.0f}/{cap} GPM ({util:.0f}%)")

    return results


if __name__ == "__main__":
    solve()

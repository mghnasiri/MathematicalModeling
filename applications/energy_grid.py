"""
Real-World Application: Energy Grid Resource Allocation.

Domain: Power generation / Electricity market operations
Model: Linear Programming (unit commitment) + Network Flow (grid transmission)

Scenario:
    A regional electricity grid operator manages 6 power plants supplying
    8 demand districts through a transmission network. The operator must:

    1. **Generation Planning (LP):** Determine how much each plant generates
       to minimize total fuel cost, subject to capacity limits, minimum
       generation levels, and total demand satisfaction.
    2. **Grid Flow Analysis (Max Flow):** Determine maximum power that can
       flow from generation region to demand region through the transmission
       network (cable capacities, transformer limits).
    3. **Backup Network (MST):** Find the minimum-cost backbone network
       that keeps all nodes connected for emergency routing.

Real-world considerations modeled:
    - Heterogeneous generation costs (coal, gas, nuclear, wind, solar)
    - Transmission line capacity constraints
    - Minimum generation requirements (nuclear/coal ramp constraints)
    - Renewable intermittency (wind/solar capacity factors)

Industry context:
    Economic dispatch is solved every 5-15 minutes in real electricity
    markets. LP-based unit commitment saves utilities 1-5% of annual
    fuel costs (~$50-500M for large ISOs) (Hobbs et al., 2001).

References:
    Hobbs, B.F., Rothkopf, M.H., O'Neill, R.P. & Chao, H. (2001).
    The Next Generation of Electric Power Unit Commitment Models.
    Springer. https://doi.org/10.1007/978-1-4757-3427-7

    Wood, A.J., Wollenberg, B.F. & Sheble, G.B. (2014). Power
    Generation, Operation, and Control. 3rd ed. Wiley.
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

POWER_PLANTS = [
    {"name": "Oakville Coal",     "type": "coal",    "capacity": 800,
     "min_gen": 200, "cost_per_mwh": 45.0},
    {"name": "Riverside Gas",     "type": "gas",     "capacity": 600,
     "min_gen": 50,  "cost_per_mwh": 65.0},
    {"name": "Lakeside Nuclear",  "type": "nuclear", "capacity": 1200,
     "min_gen": 900, "cost_per_mwh": 12.0},
    {"name": "Hilltop Wind Farm", "type": "wind",    "capacity": 400,
     "min_gen": 0,   "cost_per_mwh": 5.0},
    {"name": "Desert Solar",      "type": "solar",   "capacity": 350,
     "min_gen": 0,   "cost_per_mwh": 8.0},
    {"name": "Valley Gas Peaker", "type": "gas",     "capacity": 300,
     "min_gen": 0,   "cost_per_mwh": 85.0},
]

DEMAND_DISTRICTS = [
    {"name": "Metro Center",      "demand": 500},
    {"name": "Industrial Zone",   "demand": 400},
    {"name": "North Suburbs",     "demand": 250},
    {"name": "South Suburbs",     "demand": 200},
    {"name": "University Town",   "demand": 150},
    {"name": "Harbor District",   "demand": 300},
    {"name": "Tech Corridor",     "demand": 350},
    {"name": "Airport Region",    "demand": 200},
]

# Transmission network (node indices: 0-5 = plants, 6-13 = districts)
TRANSMISSION_LINES = [
    # Plant → district connections (capacity in MW)
    (0, 6, 400),   # Coal → Metro
    (0, 7, 350),   # Coal → Industrial
    (1, 6, 300),   # Gas → Metro
    (1, 8, 250),   # Gas → North Suburbs
    (2, 7, 500),   # Nuclear → Industrial
    (2, 9, 300),   # Nuclear → South Suburbs
    (2, 11, 400),  # Nuclear → Harbor
    (3, 8, 200),   # Wind → North Suburbs
    (3, 10, 150),  # Wind → University
    (4, 9, 200),   # Solar → South Suburbs
    (4, 12, 250),  # Solar → Tech Corridor
    (5, 12, 200),  # Peaker → Tech Corridor
    (5, 13, 150),  # Peaker → Airport
    # Inter-district transfer links
    (6, 7, 200),   # Metro ↔ Industrial
    (6, 8, 150),   # Metro ↔ North
    (7, 11, 180),  # Industrial ↔ Harbor
    (8, 10, 120),  # North ↔ University
    (9, 11, 100),  # South ↔ Harbor
    (11, 12, 200), # Harbor ↔ Tech
    (12, 13, 150), # Tech ↔ Airport
]


def create_dispatch_instance(capacity_factor: float = 0.7) -> dict:
    """Create an economic dispatch LP instance.

    Args:
        capacity_factor: Fraction of renewable capacity available
            (models intermittency; 0.7 = 70% availability).

    Returns:
        Dictionary with LP formulation data.
    """
    n_plants = len(POWER_PLANTS)
    total_demand = sum(d["demand"] for d in DEMAND_DISTRICTS)

    # Effective capacities (renewables reduced by capacity factor)
    effective_caps = []
    for plant in POWER_PLANTS:
        if plant["type"] in ("wind", "solar"):
            effective_caps.append(plant["capacity"] * capacity_factor)
        else:
            effective_caps.append(float(plant["capacity"]))

    return {
        "n_plants": n_plants,
        "plant_names": [p["name"] for p in POWER_PLANTS],
        "plant_types": [p["type"] for p in POWER_PLANTS],
        "costs": np.array([p["cost_per_mwh"] for p in POWER_PLANTS]),
        "capacities": np.array(effective_caps),
        "min_generation": np.array([p["min_gen"] for p in POWER_PLANTS], dtype=float),
        "total_demand": float(total_demand),
    }


def create_grid_network() -> dict:
    """Create the transmission grid network for flow analysis.

    Returns:
        Dictionary with network data for max flow.
    """
    # Total nodes: 6 plants + 8 districts + 2 (super-source, super-sink)
    n = 6 + 8 + 2  # 16 nodes
    source = 14  # super-source
    sink = 15    # super-sink

    edges = []
    # Super-source → plants (capacity = plant capacity)
    for i, plant in enumerate(POWER_PLANTS):
        edges.append((source, i, plant["capacity"]))

    # Transmission lines (shift district indices by 0)
    for u, v, cap in TRANSMISSION_LINES:
        edges.append((u, v, cap))

    # Districts → super-sink (capacity = demand)
    for i, district in enumerate(DEMAND_DISTRICTS):
        edges.append((6 + i, sink, district["demand"]))

    return {
        "n": n,
        "source": source,
        "sink": sink,
        "edges": edges,
        "node_names": (
            [p["name"] for p in POWER_PLANTS]
            + [d["name"] for d in DEMAND_DISTRICTS]
            + ["Super-Source", "Super-Sink"]
        ),
    }


def create_backbone_network() -> dict:
    """Create network for MST backbone analysis.

    Returns:
        Dictionary with undirected weighted edges for MST.
    """
    # Use all physical nodes (plants + districts), with line distances as weights
    n = 6 + 8  # 14 nodes
    edges = []

    # Use transmission line capacity inversely as cost (higher cap = lower cost/MW)
    for u, v, cap in TRANSMISSION_LINES:
        # Cost inversely proportional to capacity (infrastructure cost metric)
        cost = 1000.0 / cap * 100  # normalized
        edges.append((u, v, cost))

    return {
        "n": n,
        "edges": edges,
        "node_names": (
            [p["name"] for p in POWER_PLANTS]
            + [d["name"] for d in DEMAND_DISTRICTS]
        ),
    }


def solve_energy_grid(verbose: bool = True) -> dict:
    """Solve the energy grid optimization problem.

    Returns:
        Dictionary with dispatch, flow, and backbone results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    lp_dir = os.path.join(base_dir, "problems", "continuous", "linear_programming")
    mf_dir = os.path.join(base_dir, "problems", "location_network", "max_flow")
    mst_dir = os.path.join(base_dir, "problems", "location_network", "min_spanning_tree")

    lp_inst_mod = _load_mod("lp_inst_app", os.path.join(lp_dir, "instance.py"))
    lp_solve_mod = _load_mod("lp_solve_app", os.path.join(lp_dir, "exact", "lp_solver.py"))
    mf_inst_mod = _load_mod("mf_inst_app", os.path.join(mf_dir, "instance.py"))
    mf_solve_mod = _load_mod("mf_solve_app", os.path.join(mf_dir, "exact", "edmonds_karp.py"))
    mst_inst_mod = _load_mod("mst_inst_app", os.path.join(mst_dir, "instance.py"))
    mst_solve_mod = _load_mod("mst_solve_app", os.path.join(mst_dir, "exact", "mst_algorithms.py"))

    results = {}

    # ── Economic Dispatch (LP) ───────────────────────────────────────────
    dispatch_data = create_dispatch_instance()
    n = dispatch_data["n_plants"]

    # Objective: minimize total generation cost
    c = dispatch_data["costs"]

    # Inequality constraints: x_i <= capacity_i (upper bounds)
    # Also: -x_i <= -min_gen_i (lower bounds, only for plants with min_gen > 0)
    A_ub_rows = []
    b_ub_vals = []

    for i in range(n):
        if dispatch_data["min_generation"][i] > 0:
            row = np.zeros(n)
            row[i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(-dispatch_data["min_generation"][i])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    # Equality: total generation = total demand
    A_eq = np.ones((1, n))
    b_eq = np.array([dispatch_data["total_demand"]])

    bounds = [(0.0, cap) for cap in dispatch_data["capacities"]]

    lp_instance = lp_inst_mod.LPInstance(
        n=n, c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
    )

    lp_sol = lp_solve_mod.solve_lp(lp_instance)
    results["dispatch"] = {
        "success": lp_sol.success,
        "total_cost": lp_sol.objective,
        "generation": lp_sol.x.tolist() if lp_sol.success else [],
        "plant_names": dispatch_data["plant_names"],
        "total_demand": dispatch_data["total_demand"],
    }

    # ── Grid Max Flow ────────────────────────────────────────────────────
    grid_data = create_grid_network()
    capacity_matrix = np.zeros((grid_data["n"], grid_data["n"]))
    for u, v, cap in grid_data["edges"]:
        capacity_matrix[u][v] = cap

    mf_instance = mf_inst_mod.MaxFlowInstance.from_edges(
        n=grid_data["n"],
        source=grid_data["source"],
        sink=grid_data["sink"],
        edges=[(u, v, float(cap)) for u, v, cap in grid_data["edges"]],
    )

    mf_sol = mf_solve_mod.edmonds_karp(mf_instance)
    min_cut_s, min_cut_t = mf_sol.min_cut if mf_sol.min_cut else ([], [])
    results["max_flow"] = {
        "max_throughput": mf_sol.max_flow,
        "min_cut": (list(min_cut_s), list(min_cut_t)),
        "total_demand": dispatch_data["total_demand"],
    }

    # ── Backbone Network (MST) ───────────────────────────────────────────
    backbone_data = create_backbone_network()
    backbone_n = backbone_data["n"]

    mst_instance = mst_inst_mod.MSTInstance.from_edges(
        n=backbone_n,
        edges=[(u, v, w) for u, v, w in backbone_data["edges"]],
    )

    kruskal_sol = mst_solve_mod.kruskal(mst_instance)
    results["mst"] = {
        "total_cost": kruskal_sol.total_weight,
        "backbone_links": kruskal_sol.tree_edges,
        "n_links": len(kruskal_sol.tree_edges),
    }

    if verbose:
        print("=" * 70)
        print("ENERGY GRID RESOURCE ALLOCATION")
        print(f"  {n} power plants, {len(DEMAND_DISTRICTS)} demand districts")
        print(f"  Total demand: {dispatch_data['total_demand']:.0f} MW")
        print("=" * 70)

        if lp_sol.success:
            print("\n--- Economic Dispatch (LP) ---")
            print(f"  Minimum cost: ${lp_sol.objective:,.2f}/hr")
            for i in range(n):
                gen = lp_sol.x[i]
                cap = dispatch_data["capacities"][i]
                ptype = dispatch_data["plant_types"][i]
                util = gen / cap * 100 if cap > 0 else 0
                print(f"    {dispatch_data['plant_names'][i]:25s} "
                      f"{gen:7.1f}/{cap:7.1f} MW ({util:4.1f}%) "
                      f"[{ptype}] ${gen * dispatch_data['costs'][i]:,.0f}/hr")

        print(f"\n--- Grid Transmission (Max Flow) ---")
        print(f"  Max throughput: {mf_sol.max_flow:.0f} MW")
        can_meet = mf_sol.max_flow >= dispatch_data["total_demand"]
        print(f"  Can meet demand: {'Yes' if can_meet else 'NO — bottleneck!'}")

        print(f"\n--- Backbone Network (MST) ---")
        print(f"  Total backbone cost: {kruskal_sol.total_weight:.1f}")
        print(f"  Links: {len(kruskal_sol.tree_edges)}")

    return results


if __name__ == "__main__":
    solve_energy_grid()

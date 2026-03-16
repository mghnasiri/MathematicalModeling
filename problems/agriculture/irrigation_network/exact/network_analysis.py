"""
Irrigation Network Analysis Algorithms

Analyzes the irrigation network using three graph algorithms:
1. MST (Kruskal/Prim) — minimum-cost pipe backbone
2. Max Flow (Edmonds-Karp) — maximum water throughput + bottleneck
3. Shortest Path (Dijkstra) — fastest water delivery to fields

Complexity:
    - Kruskal: O(E log E)
    - Prim: O(E log V)
    - Edmonds-Karp: O(V * E^2)
    - Dijkstra: O((V+E) log V)

References:
    Kruskal, J.B. (1956). On the shortest spanning subtree of a graph
    and the traveling salesman problem. Proceedings of the American
    Mathematical Society, 7(1), 48-50.

    Edmonds, J. & Karp, R.M. (1972). Theoretical improvements in
    algorithmic efficiency for network flow problems. Journal of the ACM,
    19(2), 248-264. https://doi.org/10.1145/321694.321699

    Dijkstra, E.W. (1959). A note on two problems in connexion with
    graphs. Numerische Mathematik, 1, 269-271.
    https://doi.org/10.1007/BF01386390
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


_inst = _load_mod(
    "irr_inst",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
IrrigationNetworkInstance = _inst.IrrigationNetworkInstance
IrrigationNetworkSolution = _inst.IrrigationNetworkSolution


def _get_network_modules():
    """Load network optimization modules."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    loc_dir = os.path.join(base_dir, "problems", "location_network")

    mst_inst = _load_mod(
        "mst_inst_irr", os.path.join(loc_dir, "min_spanning_tree", "instance.py")
    )
    mst_alg = _load_mod(
        "mst_alg_irr",
        os.path.join(loc_dir, "min_spanning_tree", "exact", "mst_algorithms.py"),
    )
    mf_inst = _load_mod(
        "mf_inst_irr", os.path.join(loc_dir, "max_flow", "instance.py")
    )
    mf_ek = _load_mod(
        "mf_ek_irr", os.path.join(loc_dir, "max_flow", "exact", "edmonds_karp.py")
    )
    sp_inst = _load_mod(
        "sp_inst_irr", os.path.join(loc_dir, "shortest_path", "instance.py")
    )
    sp_dj = _load_mod(
        "sp_dj_irr",
        os.path.join(loc_dir, "shortest_path", "exact", "dijkstra.py"),
    )
    return mst_inst, mst_alg, mf_inst, mf_ek, sp_inst, sp_dj


def solve_irrigation_network(
    instance: IrrigationNetworkInstance,
) -> IrrigationNetworkSolution:
    """Analyze irrigation network using MST, Max Flow, and Shortest Path.

    Args:
        instance: IrrigationNetworkInstance to analyze.

    Returns:
        IrrigationNetworkSolution with all three analyses.
    """
    mst_inst, mst_alg, mf_inst, mf_ek, sp_inst, sp_dj = _get_network_modules()

    # -- MST: minimum-cost pipe backbone --
    mst_instance = mst_inst.MSTInstance.from_edges(
        instance.n_nodes, instance.get_mst_edges(),
        name="irrigation_backbone",
    )
    kruskal_sol = mst_alg.kruskal(mst_instance)

    # -- Max Flow: pump → reservoir throughput --
    mf_instance = mf_inst.MaxFlowInstance.from_edges(
        n=instance.n_nodes,
        source=instance.source,
        sink=instance.sink,
        edges=instance.get_flow_edges(),
        name="irrigation_flow",
    )
    mf_sol = mf_ek.edmonds_karp(mf_instance)
    s_set, t_set = mf_sol.min_cut

    # -- Shortest Path: fastest delivery to each field --
    sp_instance = sp_inst.ShortestPathInstance.from_edges(
        n=instance.n_nodes,
        edges=instance.get_sp_edges(),
        name="irrigation_delivery",
    )
    shortest_paths = {}
    for target in instance.field_nodes:
        sol = sp_dj.dijkstra(sp_instance, source=instance.source, target=target)
        shortest_paths[target] = (sol.path, sol.distance)

    return IrrigationNetworkSolution(
        mst_cost=kruskal_sol.total_weight,
        mst_edges=kruskal_sol.tree_edges,
        max_flow=mf_sol.max_flow,
        min_cut=(list(s_set), list(t_set)),
        shortest_paths=shortest_paths,
    )


if __name__ == "__main__":
    inst = IrrigationNetworkInstance.standard_farm()
    sol = solve_irrigation_network(inst)
    print("=== Irrigation Network Analysis ===\n")
    print(f"MST backbone cost: ${sol.mst_cost:,.0f}")
    print(f"  Backbone pipes: {len(sol.mst_edges)}")
    print(f"\nMax flow: {sol.max_flow:,.0f} L/hr")
    print(f"  Total demand: {inst.total_field_demand:,.0f} L/hr")
    print(f"\nShortest paths from pump:")
    for target, (path, dist) in sol.shortest_paths.items():
        node_name = inst.nodes[target].name
        print(f"  -> {node_name}: {dist:.0f} m via {path}")

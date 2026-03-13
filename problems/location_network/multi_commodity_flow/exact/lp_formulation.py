"""Multi-Commodity Flow LP formulation via scipy.optimize.linprog.

Algorithm: LP relaxation of the multi-commodity flow problem.
Complexity: Polynomial (interior-point or simplex on LP).

Variables: f_{k,e} = flow of commodity k on edge e
Constraints:
    - Capacity: sum_k f_{k,e} <= cap_e for each edge e
    - Flow conservation: sum_in - sum_out = demand at sink, -demand at source
    - Non-negativity: f_{k,e} >= 0

References:
    Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993).
    Network Flows: Theory, Algorithms, and Applications.
"""
import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import linprog

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "mcf_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

MultiCommodityFlowInstance = _inst.MultiCommodityFlowInstance
MultiCommodityFlowSolution = _inst.MultiCommodityFlowSolution


def solve_mcf_lp(instance: MultiCommodityFlowInstance) -> MultiCommodityFlowSolution:
    """Solve the multi-commodity flow problem using LP.

    Args:
        instance: A MultiCommodityFlowInstance.

    Returns:
        MultiCommodityFlowSolution with flow assignments.
    """
    n_nodes = instance.n_nodes
    edges = instance.edges
    n_edges = len(edges)
    n_commodities = len(instance.commodities)

    # Variables: f[k * n_edges + e] = flow of commodity k on edge e
    n_vars = n_commodities * n_edges

    # Objective: minimize total flow (can also be zero-cost feasibility)
    c = np.ones(n_vars)

    # Build adjacency: for each node, list of (edge_index, is_outgoing)
    out_edges: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    in_edges: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for e_idx, (u, v) in enumerate(edges):
        out_edges[u].append(e_idx)
        in_edges[v].append(e_idx)

    # Inequality constraints: capacity constraints
    # sum_k f_{k,e} <= cap_e for each edge e
    A_ub_rows = []
    b_ub = []
    for e_idx in range(n_edges):
        row = np.zeros(n_vars)
        for k in range(n_commodities):
            row[k * n_edges + e_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(instance.capacities[e_idx])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub) if b_ub else None

    # Equality constraints: flow conservation per commodity per node
    A_eq_rows = []
    b_eq = []
    for k, commodity in enumerate(instance.commodities):
        for node in range(n_nodes):
            row = np.zeros(n_vars)
            # outflow - inflow = net
            for e_idx in out_edges[node]:
                row[k * n_edges + e_idx] = 1.0
            for e_idx in in_edges[node]:
                row[k * n_edges + e_idx] = -1.0

            if node == commodity.source:
                net = commodity.demand   # source pushes out demand
            elif node == commodity.sink:
                net = -commodity.demand  # sink absorbs demand
            else:
                net = 0.0

            A_eq_rows.append(row)
            b_eq.append(net)

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq)

    # Variable bounds: f >= 0
    bounds = [(0, None)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        flows: dict[int, dict[int, float]] = {}
        total_flow = 0.0
        for k in range(n_commodities):
            flows[k] = {}
            for e_idx in range(n_edges):
                val = result.x[k * n_edges + e_idx]
                if val > 1e-8:
                    flows[k][e_idx] = val
                    total_flow += val

        return MultiCommodityFlowSolution(
            flows=flows, total_flow=total_flow, feasible=True
        )
    else:
        return MultiCommodityFlowSolution(
            flows={}, total_flow=0.0, feasible=False
        )


if __name__ == "__main__":
    inst = MultiCommodityFlowInstance.random(n_nodes=5, n_edges=8, n_commodities=2)
    sol = solve_mcf_lp(inst)
    print(f"Instance: {inst.n_nodes} nodes, {len(inst.edges)} edges, "
          f"{len(inst.commodities)} commodities")
    print(sol)

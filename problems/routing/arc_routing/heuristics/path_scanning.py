"""Path-Scanning heuristic for the Capacitated Arc Routing Problem (CARP).

Algorithm: Build routes greedily. At each step, extend the current route by
selecting the nearest unserved required edge that fits within remaining
capacity. When no edge fits, return to depot and start a new route.

Complexity: O(R^2 * V) where R = number of required edges.

References:
    Golden, B. L., DeArmon, J. S., & Baker, E. K. (1983). Computational
    experiments with algorithms for a class of routing problems. Computers
    & Operations Research, 10(1), 47-59.
"""
import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "carp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

CARPInstance = _inst.CARPInstance
CARPSolution = _inst.CARPSolution


def path_scanning(instance: CARPInstance) -> CARPSolution:
    """Path-scanning heuristic for CARP.

    Builds routes one at a time. For each route, greedily selects the
    nearest unserved required edge from the current position, as long as
    the vehicle capacity is not exceeded.

    Args:
        instance: A CARPInstance.

    Returns:
        CARPSolution with routes serving all required edges.
    """
    dist = instance.shortest_paths()
    depot = instance.depot
    required = set(instance.required_edges)

    if not required:
        return CARPSolution(routes=[], total_cost=0.0, feasible=True)

    routes: list[list[tuple[int, tuple[int, int]]]] = []
    served: set[int] = set()
    total_cost = 0.0

    while served != required:
        route: list[tuple[int, tuple[int, int]]] = []
        current_node = depot
        remaining_cap = instance.capacity
        route_cost = 0.0

        while True:
            # Find nearest unserved required edge that fits
            best_edge = None
            best_dist = np.inf
            best_node = None
            best_direction = None

            for e_idx in required - served:
                u, v = instance.edges[e_idx]
                demand = instance.demands[e_idx]
                if demand > remaining_cap:
                    continue

                # Can approach from u or v side
                for start, end in [(u, v), (v, u)]:
                    d = dist[current_node][start]
                    if d < best_dist:
                        best_dist = d
                        best_edge = e_idx
                        best_node = end
                        best_direction = (start, end)

            if best_edge is None:
                break

            # Travel to the edge start and traverse it
            start, end = best_direction
            route_cost += dist[current_node][start]  # deadhead to start
            route_cost += instance.costs[best_edge]   # traverse edge
            current_node = end
            remaining_cap -= instance.demands[best_edge]
            served.add(best_edge)
            route.append((best_edge, best_direction))

        # Return to depot
        route_cost += dist[current_node][depot]
        total_cost += route_cost
        if route:
            routes.append(route)

    # Verify feasibility
    all_served = served == required
    cap_ok = True
    for route in routes:
        load = sum(instance.demands[e_idx] for e_idx, _ in route)
        if load > instance.capacity + 1e-9:
            cap_ok = False
            break

    return CARPSolution(
        routes=routes, total_cost=total_cost,
        feasible=all_served and cap_ok
    )


if __name__ == "__main__":
    inst = CARPInstance.random()
    sol = path_scanning(inst)
    print(f"Instance: {inst.n_nodes} nodes, {inst.n_required} required edges")
    print(sol)

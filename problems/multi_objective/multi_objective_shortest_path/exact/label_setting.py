"""Label-Setting algorithm for Multi-Objective Shortest Path.

Algorithm: Multi-objective Dijkstra with dominance pruning. Maintains
a set of Pareto-optimal labels (cost vectors) at each node. A label
is dominated if another label at the same node is component-wise <= with
at least one strict inequality.

Complexity: O(|P| * E * log(|P| * V)) where |P| is the number of
Pareto-optimal paths (can be exponential in worst case).

References:
    Martins, E. Q. V. (1984). On a multicriteria shortest path problem.
    European Journal of Operational Research, 16(2), 236-245.

    Hansen, P. (1980). Bicriterion path problems. In Multiple Criteria
    Decision Making Theory and Application (pp. 109-127). Springer.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import heapq


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "mosp_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MultiObjectiveSPInstance = _inst.MultiObjectiveSPInstance
MOSPSolution = _inst.MOSPSolution


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Check if cost vector a dominates b (a <= b componentwise, a != b).

    Args:
        a: First cost vector.
        b: Second cost vector.

    Returns:
        True if a dominates b.
    """
    at_least_one_strict = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_strict = True
    return at_least_one_strict


def _is_dominated_by_set(cost: tuple[float, ...],
                         label_set: list[tuple[float, ...]]) -> bool:
    """Check if cost is dominated by any label in the set."""
    for lbl in label_set:
        if dominates(lbl, cost):
            return True
    return False


def label_setting(instance: MultiObjectiveSPInstance) -> MOSPSolution:
    """Multi-objective label-setting shortest path algorithm.

    Finds all Pareto-optimal paths from source to target.

    Args:
        instance: A MultiObjectiveSPInstance.

    Returns:
        A MOSPSolution with all Pareto-optimal paths and their costs.
    """
    adj = instance.get_adjacency()
    source = instance.source
    target = instance.target
    n_obj = instance.n_objectives

    # Labels at each node: list of (cost_vector, path)
    # Use a priority queue with sum of costs as priority
    initial_cost = tuple(0.0 for _ in range(n_obj))
    # (priority, cost_vector, node, path)
    pq: list[tuple[float, tuple[float, ...], int, list[int]]] = []
    heapq.heappush(pq, (0.0, initial_cost, source, [source]))

    # Pareto-optimal labels at each node (just cost vectors for pruning)
    node_labels: dict[int, list[tuple[float, ...]]] = {i: [] for i in range(instance.n)}

    # Results at target
    pareto_paths: list[list[int]] = []
    pareto_costs: list[tuple[float, ...]] = []

    while pq:
        _, cost, node, path = heapq.heappop(pq)

        # Check if dominated at this node
        if _is_dominated_by_set(cost, node_labels[node]):
            continue

        # Remove dominated labels at this node
        node_labels[node] = [
            lbl for lbl in node_labels[node] if not dominates(cost, lbl)
        ]
        node_labels[node].append(cost)

        if node == target:
            # Check not dominated by existing target solutions
            if not _is_dominated_by_set(cost, pareto_costs):
                # Remove dominated target solutions
                new_paths = []
                new_costs = []
                for p, c in zip(pareto_paths, pareto_costs):
                    if not dominates(cost, c):
                        new_paths.append(p)
                        new_costs.append(c)
                new_paths.append(path)
                new_costs.append(cost)
                pareto_paths = new_paths
                pareto_costs = new_costs
            continue

        # Expand neighbors
        for neighbor, edge_costs in adj[node]:
            if neighbor in path:
                continue  # avoid cycles
            new_cost = tuple(cost[i] + edge_costs[i] for i in range(n_obj))

            # Prune if dominated at neighbor
            if _is_dominated_by_set(new_cost, node_labels[neighbor]):
                continue
            # Prune if dominated by target solutions
            if _is_dominated_by_set(new_cost, pareto_costs):
                continue

            priority = sum(new_cost)
            heapq.heappush(pq, (priority, new_cost, neighbor, path + [neighbor]))

    return MOSPSolution(
        pareto_paths=pareto_paths,
        pareto_costs=pareto_costs,
    )


if __name__ == "__main__":
    inst = MultiObjectiveSPInstance.random(n=6, n_objectives=2)
    sol = label_setting(inst)
    print(f"Instance: {inst.n} nodes, {len(inst.edges)} edges, "
          f"{inst.n_objectives} objectives")
    print(f"Solution: {sol}")
    for path, cost in zip(sol.pareto_paths, sol.pareto_costs):
        print(f"  Path: {path}, Cost: {cost}")

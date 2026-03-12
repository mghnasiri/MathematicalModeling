"""
Bron-Kerbosch Algorithm for Maximum Clique.

Problem: Maximum Clique (MC)
Complexity: O(3^(n/3)) worst case for maximal cliques

Finds the maximum clique using the Bron-Kerbosch algorithm with
pivot selection to prune the search tree.

References:
    Bron, C. & Kerbosch, J. (1973). Finding all cliques of an
    undirected graph. Communications of the ACM, 16(9), 575-577.
    https://doi.org/10.1145/362342.362367
"""

from __future__ import annotations

import os
import sys
import importlib.util


def _load_parent(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "mc_instance_bk",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
MaxCliqueInstance = _inst.MaxCliqueInstance
MaxCliqueSolution = _inst.MaxCliqueSolution


def bron_kerbosch(instance: MaxCliqueInstance) -> MaxCliqueSolution:
    """Find maximum clique using Bron-Kerbosch with pivoting.

    Args:
        instance: A MaxCliqueInstance.

    Returns:
        MaxCliqueSolution with the largest clique found.
    """
    adj = instance.adj
    best_clique: list[int] = []

    def _bk(R: set[int], P: set[int], X: set[int]):
        nonlocal best_clique

        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return

        # Choose pivot with max connections to P
        pivot = max(P | X, key=lambda v: len(adj[v] & P))

        for v in list(P - adj[pivot]):
            _bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    _bk(set(), set(range(instance.n_vertices)), set())

    return MaxCliqueSolution(clique=sorted(best_clique), size=len(best_clique))


def greedy_clique(instance: MaxCliqueInstance) -> MaxCliqueSolution:
    """Greedy maximum clique: start with highest-degree vertex, extend.

    Args:
        instance: A MaxCliqueInstance.

    Returns:
        MaxCliqueSolution (not necessarily optimal).
    """
    adj = instance.adj

    best_clique: list[int] = []

    # Try starting from each vertex
    order = sorted(range(instance.n_vertices),
                   key=lambda v: len(adj[v]), reverse=True)

    for start in order[:min(len(order), instance.n_vertices)]:
        clique = {start}
        candidates = adj[start].copy()

        while candidates:
            # Pick candidate with most connections to current clique
            best_v = max(candidates, key=lambda v: len(adj[v] & clique))
            if adj[best_v] >= clique:  # v adjacent to all clique members
                clique.add(best_v)
                candidates &= adj[best_v]
            else:
                candidates.discard(best_v)

        if len(clique) > len(best_clique):
            best_clique = sorted(clique)

    return MaxCliqueSolution(clique=best_clique, size=len(best_clique))


if __name__ == "__main__":
    inst = MaxCliqueInstance.complete(5)
    sol = bron_kerbosch(inst)
    print(f"K5 max clique: {sol}")

    inst2 = MaxCliqueInstance.petersen()
    sol2 = bron_kerbosch(inst2)
    print(f"Petersen max clique: {sol2}")

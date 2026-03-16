"""
Cold Storage and Packaging Optimization Algorithms

Solves two packing/cutting problems:
1. Cold storage bin packing: FF, FFD, BFD, GA
2. Packaging film cutting stock: Greedy, FFD-based

Complexity:
    - FF/FFD/BFD: O(n^2) / O(n log n)
    - GA: O(pop_size * generations * n)
    - Greedy CSP: O(m * max_demand)

References:
    Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation
    algorithms for bin packing. PWS Publishing, 46-93.
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
    "cs_inst",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
ColdStorageInstance = _inst.ColdStorageInstance
ColdStorageSolution = _inst.ColdStorageSolution
PackagingInstance = _inst.PackagingInstance
PackagingSolution = _inst.PackagingSolution


def _get_packing_modules():
    """Load packing/cutting stock solver modules."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    pack_dir = os.path.join(base_dir, "problems", "packing")

    bp_inst = _load_mod(
        "bp_inst_cs", os.path.join(pack_dir, "bin_packing", "instance.py")
    )
    bp_ff = _load_mod(
        "bp_ff_cs",
        os.path.join(pack_dir, "bin_packing", "heuristics", "first_fit.py"),
    )
    bp_ga = _load_mod(
        "bp_ga_cs",
        os.path.join(pack_dir, "bin_packing", "metaheuristics",
                     "genetic_algorithm.py"),
    )
    csp_inst = _load_mod(
        "csp_inst_cs",
        os.path.join(pack_dir, "cutting_stock", "instance.py"),
    )
    csp_gr = _load_mod(
        "csp_gr_cs",
        os.path.join(pack_dir, "cutting_stock", "heuristics", "greedy_csp.py"),
    )
    return bp_inst, bp_ff, bp_ga, csp_inst, csp_gr


def solve_cold_storage(
    instance: ColdStorageInstance,
    seed: int = 42,
) -> dict[str, ColdStorageSolution]:
    """Solve cold storage bin packing with multiple methods.

    Args:
        instance: ColdStorageInstance to solve.
        seed: Random seed for GA.

    Returns:
        Dict mapping method name to ColdStorageSolution.
    """
    bp_inst, bp_ff, bp_ga, _, _ = _get_packing_modules()
    sizes = instance.get_weights()

    bp_instance = bp_inst.BinPackingInstance(
        n=instance.n_lots,
        sizes=sizes,
        capacity=instance.storage_capacity_kg,
        name="cold_storage",
    )

    results = {}
    for method_name, solver in [
        ("FF", bp_ff.first_fit),
        ("FFD", bp_ff.first_fit_decreasing),
        ("BFD", bp_ff.best_fit_decreasing),
    ]:
        sol = solver(bp_instance)
        results[method_name] = ColdStorageSolution(
            n_units=sol.num_bins,
            bins=sol.bins,
            method=method_name,
        )

    ga_sol = bp_ga.genetic_algorithm(bp_instance, seed=seed)
    results["GA"] = ColdStorageSolution(
        n_units=ga_sol.num_bins,
        bins=ga_sol.bins,
        method="GA",
    )

    return results


def solve_packaging(
    instance: PackagingInstance,
) -> dict[str, PackagingSolution]:
    """Solve packaging film cutting stock with multiple methods.

    Args:
        instance: PackagingInstance to solve.

    Returns:
        Dict mapping method name to PackagingSolution.
    """
    _, _, _, csp_inst, csp_gr = _get_packing_modules()
    lengths = instance.get_lengths()
    demands = instance.get_demands()

    csp_instance = csp_inst.CuttingStockInstance(
        m=instance.n_types,
        stock_length=instance.roll_length_cm,
        lengths=lengths,
        demands=demands,
        name="packaging_film",
    )

    total_material = float(np.sum(lengths * demands))
    results = {}

    for method_name, solver in [
        ("Greedy", csp_gr.greedy_largest_first),
        ("FFD", csp_gr.ffd_based),
    ]:
        sol = solver(csp_instance)
        total_stock = sol.num_rolls * instance.roll_length_cm
        waste = total_stock - total_material
        waste_pct = waste / total_stock * 100 if total_stock > 0 else 0.0

        results[method_name] = PackagingSolution(
            n_rolls=sol.num_rolls,
            waste_cm=waste,
            waste_pct=waste_pct,
            patterns=sol.patterns,
            method=method_name,
        )

    return results


if __name__ == "__main__":
    print("=== Cold Storage & Packaging Optimization ===\n")

    cs = ColdStorageInstance.packing_house()
    cs_results = solve_cold_storage(cs)
    print(f"Cold Storage ({cs.n_lots} lots, {cs.total_weight:.0f} kg):")
    for name, sol in cs_results.items():
        print(f"  {name}: {sol.n_units} units")

    pkg = PackagingInstance.standard()
    pkg_results = solve_packaging(pkg)
    print(f"\nPackaging ({pkg.n_types} sheet types):")
    for name, sol in pkg_results.items():
        print(f"  {name}: {sol.n_rolls} rolls, waste={sol.waste_pct:.1f}%")

"""
Real-World Application: Agriculture Cold Storage & Packaging Optimization.

Domain: Agriculture / Produce packing house operations
Models: Bin Packing + Cutting Stock

Scenario:
    A produce packing house receives harvested crops and must solve two
    logistics problems:

    1. Cold Storage Bin Packing: Pack 20 produce lots (pallets of varying
       sizes in kg) into cold storage units with 1000 kg capacity each.
       Different produce types have different pallet weights. Minimize
       the number of cold storage units needed to reduce energy costs.

    2. Packaging Film Cutting Stock: Cut packaging film rolls (200 cm
       stock length) into wrapping sheets for different produce box sizes.
       Four sheet sizes are needed in varying quantities. Minimize the
       number of film rolls used to reduce material waste.

Real-world considerations modeled:
    - Heterogeneous produce lot sizes (tomatoes, berries, lettuce, vegetables)
    - Cold storage capacity constraints (weight-limited)
    - Packaging material waste minimization
    - Multiple cutting patterns for film rolls

Industry context:
    Post-harvest losses in fresh produce supply chains range from 20-40%
    in developing countries and 5-15% in developed nations (FAO, 2019).
    Efficient cold storage packing reduces energy costs by 15-30%, while
    optimized cutting stock for packaging materials can reduce film waste
    by 10-20% compared to naive cutting approaches.

References:
    FAO (2019). The State of Food and Agriculture: Moving Forward on
    Food Loss and Waste Reduction. Rome: Food and Agriculture
    Organization of the United Nations.
    https://doi.org/10.4060/ca6030en

    Brosnan, T. & Sun, D.-W. (2001). Precooling techniques and
    applications for horticultural products — a review. International
    Journal of Refrigeration, 24(2), 154-170.
    https://doi.org/10.1016/S0140-7007(00)00017-7

    Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation
    algorithms for bin packing: A survey. In: Hochbaum, D.S. (ed)
    Approximation Algorithms for NP-hard Problems. PWS Publishing,
    46-93.
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

COLD_STORAGE_CAPACITY_KG = 1000  # kg per cold storage unit

# Produce lot types with typical pallet weight ranges (kg)
PRODUCE_TYPES = {
    "tomato":     {"weight_range": (150, 300), "label": "Tomato pallet"},
    "berry":      {"weight_range": (80, 150),  "label": "Berry crate"},
    "lettuce":    {"weight_range": (100, 200), "label": "Lettuce bin"},
    "mixed_veg":  {"weight_range": (50, 120),  "label": "Mixed veg box"},
}

# Packaging film specifications
FILM_ROLL_LENGTH_CM = 200  # cm per stock roll

SHEET_TYPES = {
    "large":  {"length_cm": 60, "demand": 25, "label": "Large box wrap"},
    "medium": {"length_cm": 45, "demand": 40, "label": "Medium box wrap"},
    "small":  {"length_cm": 30, "demand": 50, "label": "Small box wrap"},
    "mini":   {"length_cm": 20, "demand": 30, "label": "Mini tray wrap"},
}


def create_produce_lots(n_lots: int = 20, seed: int = 42) -> dict:
    """Generate realistic produce lot data for cold storage packing.

    Args:
        n_lots: Number of produce lots to generate.
        seed: Random seed.

    Returns:
        Dictionary with produce lot data.
    """
    rng = np.random.default_rng(seed)

    # Distribution: tomatoes most common, then lettuce, mixed veg, berries
    type_probs = {
        "tomato": 0.30, "berry": 0.20,
        "lettuce": 0.25, "mixed_veg": 0.25,
    }
    type_names = list(type_probs.keys())
    probs = list(type_probs.values())

    lots = []
    for i in range(n_lots):
        ptype = rng.choice(type_names, p=probs)
        info = PRODUCE_TYPES[ptype]
        weight = int(rng.integers(info["weight_range"][0],
                                  info["weight_range"][1] + 1))
        lots.append({
            "id": f"lot-{i:03d}",
            "type": ptype,
            "label": info["label"],
            "weight_kg": weight,
        })

    return {
        "n_lots": n_lots,
        "lots": lots,
        "storage_capacity": COLD_STORAGE_CAPACITY_KG,
    }


def create_packaging_instance() -> dict:
    """Create the packaging film cutting stock data.

    Returns:
        Dictionary with sheet type data.
    """
    sheet_data = []
    for name, info in SHEET_TYPES.items():
        sheet_data.append({
            "name": name,
            "label": info["label"],
            "length_cm": info["length_cm"],
            "demand": info["demand"],
        })

    return {
        "roll_length": FILM_ROLL_LENGTH_CM,
        "sheets": sheet_data,
    }


def solve_cold_storage_packing(verbose: bool = True) -> dict:
    """Solve cold storage bin packing and packaging film cutting stock.

    Returns:
        Dictionary with results from both problems.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pack_dir = os.path.join(base_dir, "problems", "packing")

    results = {}

    # == Part 1: Cold Storage Bin Packing ======================================

    bp_inst_mod = _load_mod(
        "bp_inst_agri",
        os.path.join(pack_dir, "bin_packing", "instance.py"),
    )
    bp_ff_mod = _load_mod(
        "bp_ff_agri",
        os.path.join(pack_dir, "bin_packing", "heuristics", "first_fit.py"),
    )
    bp_ga_mod = _load_mod(
        "bp_ga_agri",
        os.path.join(pack_dir, "bin_packing", "metaheuristics",
                     "genetic_algorithm.py"),
    )

    data = create_produce_lots(n_lots=20, seed=42)
    sizes = np.array([lot["weight_kg"] for lot in data["lots"]], dtype=float)

    bp_instance = bp_inst_mod.BinPackingInstance(
        n=data["n_lots"],
        sizes=sizes,
        capacity=float(data["storage_capacity"]),
        name="cold_storage",
    )

    ff_sol = bp_ff_mod.first_fit(bp_instance)
    ffd_sol = bp_ff_mod.first_fit_decreasing(bp_instance)
    bfd_sol = bp_ff_mod.best_fit_decreasing(bp_instance)
    ga_sol = bp_ga_mod.genetic_algorithm(bp_instance, seed=42)

    lb_l1 = bp_instance.lower_bound_l1()
    lb_l2 = bp_instance.lower_bound_l2()
    total_weight = float(np.sum(sizes))

    results["bin_packing"] = {
        "total_weight_kg": total_weight,
        "lower_bound_l1": lb_l1,
        "lower_bound_l2": lb_l2,
    }
    for name, sol in [("FF", ff_sol), ("FFD", ffd_sol),
                      ("BFD", bfd_sol), ("GA", ga_sol)]:
        results["bin_packing"][name] = {
            "n_units": sol.num_bins,
            "bins": sol.bins,
        }

    if verbose:
        print("=" * 70)
        print("AGRICULTURE COLD STORAGE & PACKAGING OPTIMIZATION")
        print("=" * 70)

        # Produce lot summary
        type_counts: dict[str, int] = {}
        type_weights: dict[str, int] = {}
        for lot in data["lots"]:
            t = lot["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            type_weights[t] = type_weights.get(t, 0) + lot["weight_kg"]

        print(f"\n  Produce lots: {data['n_lots']}")
        print(f"  Cold storage unit capacity: {data['storage_capacity']} kg")
        print(f"  Total produce weight: {total_weight:.0f} kg")
        print(f"\n  Lot breakdown:")
        for t in PRODUCE_TYPES:
            if t in type_counts:
                print(f"    {PRODUCE_TYPES[t]['label']:20s}: "
                      f"{type_counts[t]:2d} lots, "
                      f"{type_weights[t]:5d} kg total")

        print(f"\n  Lower bounds: L1 = {lb_l1}, L2 = {lb_l2}")

        print("\n--- COLD STORAGE BIN PACKING (minimize units) ---")
        for method in ["FF", "FFD", "BFD", "GA"]:
            res = results["bin_packing"][method]
            n_units = res["n_units"]
            utilization = total_weight / (n_units * data["storage_capacity"]) * 100
            print(f"\n  {method}: {n_units} cold storage units "
                  f"(utilization = {utilization:.1f}%)")
            for i, bin_items in enumerate(res["bins"]):
                bin_weight = sum(sizes[j] for j in bin_items)
                items_desc = [f"{data['lots'][j]['type']}({int(sizes[j])}kg)"
                              for j in bin_items]
                free = data["storage_capacity"] - bin_weight
                print(f"    Unit {i+1}: {bin_weight:4.0f}/{data['storage_capacity']} kg "
                      f"({free:4.0f} free) -- [{', '.join(items_desc)}]")

    # == Part 2: Packaging Film Cutting Stock ==================================

    csp_inst_mod = _load_mod(
        "csp_inst_agri",
        os.path.join(pack_dir, "cutting_stock", "instance.py"),
    )
    csp_gr_mod = _load_mod(
        "csp_gr_agri",
        os.path.join(pack_dir, "cutting_stock", "heuristics", "greedy_csp.py"),
    )

    pkg_data = create_packaging_instance()
    lengths = np.array([s["length_cm"] for s in pkg_data["sheets"]], dtype=float)
    demands = np.array([s["demand"] for s in pkg_data["sheets"]], dtype=int)

    csp_instance = csp_inst_mod.CuttingStockInstance(
        m=len(pkg_data["sheets"]),
        stock_length=float(pkg_data["roll_length"]),
        lengths=lengths,
        demands=demands,
        name="packaging_film",
    )

    greedy_sol = csp_gr_mod.greedy_largest_first(csp_instance)
    ffd_csp_sol = csp_gr_mod.ffd_based(csp_instance)

    csp_lb = csp_instance.lower_bound()
    total_material = float(np.sum(lengths * demands))

    results["cutting_stock"] = {
        "total_material_cm": total_material,
        "lower_bound": csp_lb,
    }
    for name, sol in [("Greedy", greedy_sol), ("FFD", ffd_csp_sol)]:
        total_stock = sol.num_rolls * pkg_data["roll_length"]
        waste = total_stock - total_material
        waste_pct = waste / total_stock * 100
        results["cutting_stock"][name] = {
            "n_rolls": sol.num_rolls,
            "waste_cm": waste,
            "waste_pct": waste_pct,
            "patterns": sol.patterns,
        }

    if verbose:
        print("\n" + "=" * 70)
        print("PACKAGING FILM CUTTING STOCK (minimize rolls)")
        print("=" * 70)

        print(f"\n  Film roll length: {pkg_data['roll_length']} cm")
        print(f"  Sheet types required:")
        for s in pkg_data["sheets"]:
            print(f"    {s['label']:20s}: {s['length_cm']:3d} cm x {s['demand']:3d} units")
        print(f"\n  Total material needed: {total_material:.0f} cm")
        print(f"  Lower bound (rolls): {csp_lb}")

        sheet_names = [s["name"] for s in pkg_data["sheets"]]

        for method in ["Greedy", "FFD"]:
            res = results["cutting_stock"][method]
            print(f"\n  {method}: {res['n_rolls']} rolls "
                  f"(waste = {res['waste_cm']:.0f} cm, {res['waste_pct']:.1f}%)")
            print(f"    Cutting patterns:")
            for p_idx, (pattern, freq) in enumerate(res["patterns"]):
                parts = []
                for k in range(len(sheet_names)):
                    if pattern[k] > 0:
                        parts.append(f"{pattern[k]}x{sheet_names[k]}"
                                     f"({int(lengths[k])}cm)")
                used = float(np.dot(pattern, lengths))
                trim = pkg_data["roll_length"] - used
                print(f"      Pattern {p_idx+1} (x{freq}): "
                      f"[{', '.join(parts)}] "
                      f"= {used:.0f} cm used, {trim:.0f} cm trim")

    return results


if __name__ == "__main__":
    solve_cold_storage_packing()

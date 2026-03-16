"""
Real-World Application: Post-Harvest Grain Silo Allocation.

Domain: Agricultural storage management / Grain elevator operations
Model: Bin Packing (1D)

Scenario:
    After harvest, a grain elevator must store 15 grain lots in storage
    silos. Each silo has a 50-ton capacity. Grain lots range from 5 to
    45 tons and come from different fields with different grain types.
    Different grain types (wheat, corn, soybean, barley) cannot be mixed
    in the same silo due to quality control and traceability requirements.
    However, lots of the same grain type from different fields can share
    a silo if capacity allows.

    The goal is to minimize the total number of silos used, freeing up
    remaining silos for future deliveries or other commodities.

    The problem is modeled as 1D bin packing: each grain lot is an "item"
    with size equal to its tonnage, and each silo is a "bin" with capacity
    50 tons. We solve per grain type to enforce the no-mixing constraint.

Real-world considerations modeled:
    - Silo capacity constraints (50 tons per silo)
    - Grain type segregation (quality control requirement)
    - Variable lot sizes based on field yield and acreage
    - Minimizing occupied storage to maximize operational flexibility
    - Comparison of heuristic vs. metaheuristic packing strategies

Industry context:
    Efficient silo utilization reduces storage costs by 10-20% at
    commercial grain elevators (Maier & Bakker-Arkema, 2002). The
    average US grain elevator handles 50,000-200,000 bushels across
    10-30 bins. Grain segregation for identity-preserved (IP) crops
    adds complexity, as organic, non-GMO, and conventional lots must
    not be mixed. Poor bin allocation leads to unnecessary fumigation,
    quality downgrades, and lost premium pricing.

References:
    Maier, D.E. & Bakker-Arkema, F.W. (2002). Grain storage systems
    design. In: Handbook of Postharvest Technology, Marcel Dekker,
    New York, 517-574.

    Coffman, E.G., Garey, M.R. & Johnson, D.S. (1997). Approximation
    algorithms for bin packing: A survey. In: Hochbaum, D.S. (ed)
    Approximation Algorithms for NP-Hard Problems, PWS Publishing,
    46-93.

    Reed, C.R., Doyungan, S., Ioerger, B. & Getchell, A. (2007).
    Response of storage molds to different initial moisture contents
    of maize (corn) stored at 25C, and effect on respiration rate and
    nutrient composition. Journal of Stored Products Research, 43(4),
    443-458.
    https://doi.org/10.1016/j.jspr.2006.12.006
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

SILO_CAPACITY = 50  # tons per silo

# 15 grain lots from various fields
GRAIN_LOTS = [
    {"id": "LOT-001", "field": "Anderson Farm",    "crop": "wheat",   "tons": 38},
    {"id": "LOT-002", "field": "Anderson Farm",    "crop": "wheat",   "tons": 25},
    {"id": "LOT-003", "field": "Baker Ranch",      "crop": "corn",    "tons": 45},
    {"id": "LOT-004", "field": "Baker Ranch",      "crop": "corn",    "tons": 30},
    {"id": "LOT-005", "field": "Clark Farms",      "crop": "soybean", "tons": 22},
    {"id": "LOT-006", "field": "Clark Farms",      "crop": "soybean", "tons": 18},
    {"id": "LOT-007", "field": "Davis Homestead",  "crop": "wheat",   "tons": 15},
    {"id": "LOT-008", "field": "Evans Estate",     "crop": "corn",    "tons": 35},
    {"id": "LOT-009", "field": "Foster Fields",    "crop": "barley",  "tons": 12},
    {"id": "LOT-010", "field": "Foster Fields",    "crop": "barley",  "tons": 28},
    {"id": "LOT-011", "field": "Green Acres",      "crop": "soybean", "tons": 8},
    {"id": "LOT-012", "field": "Harris Farm",      "crop": "wheat",   "tons": 42},
    {"id": "LOT-013", "field": "Irving Ranch",     "crop": "corn",    "tons": 20},
    {"id": "LOT-014", "field": "Johnson Fields",   "crop": "barley",  "tons": 5},
    {"id": "LOT-015", "field": "Kelly Farm",       "crop": "soybean", "tons": 10},
]


def create_instance() -> dict:
    """Create grain silo packing instance data.

    Returns:
        Dictionary with bin packing instance data per grain type.
    """
    # Group lots by crop type (cannot mix types)
    crop_groups = {}
    for lot in GRAIN_LOTS:
        crop = lot["crop"]
        if crop not in crop_groups:
            crop_groups[crop] = []
        crop_groups[crop].append(lot)

    return {
        "n_lots": len(GRAIN_LOTS),
        "silo_capacity": SILO_CAPACITY,
        "lots": GRAIN_LOTS,
        "crop_groups": crop_groups,
    }


def solve(verbose: bool = True) -> dict:
    """Solve grain silo allocation using bin packing.

    Returns:
        Dictionary with packing results per grain type and overall.
    """
    data = create_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bp_dir = os.path.join(base_dir, "problems", "packing", "bin_packing")

    bp_inst_mod = _load_mod("bp_inst_silo", os.path.join(bp_dir, "instance.py"))
    bp_ff_mod = _load_mod("bp_ff_silo", os.path.join(bp_dir, "heuristics", "first_fit.py"))
    bp_ga_mod = _load_mod("bp_ga_silo", os.path.join(bp_dir, "metaheuristics", "genetic_algorithm.py"))

    results = {"by_crop": {}, "totals": {}}

    total_silos_ff = 0
    total_silos_ffd = 0
    total_silos_bfd = 0
    total_silos_ga = 0

    for crop, lots in data["crop_groups"].items():
        sizes = np.array([lot["tons"] for lot in lots], dtype=float)
        n = len(lots)

        bp_instance = bp_inst_mod.BinPackingInstance(
            n=n,
            sizes=sizes,
            capacity=data["silo_capacity"],
            name=f"silo_{crop}",
        )

        ff_sol = bp_ff_mod.first_fit(bp_instance)
        ffd_sol = bp_ff_mod.first_fit_decreasing(bp_instance)
        bfd_sol = bp_ff_mod.best_fit_decreasing(bp_instance)
        ga_sol = bp_ga_mod.genetic_algorithm(
            bp_instance, pop_size=30, generations=100, seed=42
        )

        crop_results = {}
        for name, sol in [("FF", ff_sol), ("FFD", ffd_sol),
                          ("BFD", bfd_sol), ("GA", ga_sol)]:
            crop_results[name] = {
                "n_silos": sol.num_bins,
                "bins": sol.bins,
            }

        results["by_crop"][crop] = {
            "lots": lots,
            "sizes": sizes.tolist(),
            "total_tons": float(sizes.sum()),
            "solutions": crop_results,
        }

        total_silos_ff += crop_results["FF"]["n_silos"]
        total_silos_ffd += crop_results["FFD"]["n_silos"]
        total_silos_bfd += crop_results["BFD"]["n_silos"]
        total_silos_ga += crop_results["GA"]["n_silos"]

    results["totals"] = {
        "FF": total_silos_ff,
        "FFD": total_silos_ffd,
        "BFD": total_silos_bfd,
        "GA": total_silos_ga,
    }

    if verbose:
        total_tons = sum(lot["tons"] for lot in GRAIN_LOTS)
        print("=" * 70)
        print("POST-HARVEST GRAIN SILO ALLOCATION (BIN PACKING)")
        print(f"  {data['n_lots']} grain lots, {total_tons} tons total")
        print(f"  Silo capacity: {SILO_CAPACITY} tons each")
        print(f"  Grain types: {', '.join(data['crop_groups'].keys())}")
        print("=" * 70)

        # Lot inventory
        print("\n  Grain lot inventory:")
        for lot in GRAIN_LOTS:
            print(f"    {lot['id']}: {lot['field']:20s} — "
                  f"{lot['crop']:8s} {lot['tons']:2d}t")

        # Per-crop results
        for crop, crop_data in results["by_crop"].items():
            n_lots = len(crop_data["lots"])
            crop_tons = crop_data["total_tons"]
            min_silos = max(1, int(np.ceil(crop_tons / SILO_CAPACITY)))
            print(f"\n--- {crop.upper()} ({n_lots} lots, {crop_tons:.0f}t, "
                  f"min silos: {min_silos}) ---")

            # Show best result (FFD typically)
            best_method = min(crop_data["solutions"],
                              key=lambda k: crop_data["solutions"][k]["n_silos"])
            best = crop_data["solutions"][best_method]

            methods_summary = ", ".join(
                f"{m}={s['n_silos']}" for m, s in crop_data["solutions"].items()
            )
            print(f"  Methods: {methods_summary}")
            print(f"  Best ({best_method}) silo assignments:")

            for i, bin_items in enumerate(best["bins"]):
                bin_tons = sum(crop_data["sizes"][j] for j in bin_items)
                free = SILO_CAPACITY - bin_tons
                lot_names = [crop_data["lots"][j]["id"] for j in bin_items]
                util = bin_tons / SILO_CAPACITY * 100
                print(f"    Silo {i+1}: {bin_tons:.0f}/{SILO_CAPACITY}t "
                      f"({util:.0f}%, {free:.0f}t free) — "
                      f"[{', '.join(lot_names)}]")

        # Overall summary
        print(f"\n--- OVERALL SILO USAGE ---")
        print(f"  Theoretical minimum: "
              f"{max(1, int(np.ceil(total_tons / SILO_CAPACITY)))}")
        for method, count in results["totals"].items():
            utilization = total_tons / (count * SILO_CAPACITY) * 100
            print(f"  {method:3s}: {count} silos "
                  f"(avg utilization: {utilization:.1f}%)")

        best_total = min(results["totals"].values())
        worst_total = max(results["totals"].values())
        if worst_total > best_total:
            saved = worst_total - best_total
            print(f"\n  Optimization saves up to {saved} silo(s) vs. naive packing")

    return results


if __name__ == "__main__":
    solve()

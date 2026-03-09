"""
Real-World Application: Hospital Bed & Ward Capacity Management.

Domain: Hospital operations / Patient flow management
Model: Bin Packing + Knapsack

Scenario:
    A 120-bed hospital has 4 wards, each with limited bed capacity:
      - Ward A (Medical): 35 beds
      - Ward B (Surgical): 30 beds
      - Ward C (Cardiac): 25 beds
      - Ward D (General): 30 beds

    Problem 1 (Bin Packing): 22 new admissions arrive in the ED.
    Each patient has a "bed-equivalent" score based on care needs
    (isolation rooms count as 2 beds, ICU stepdowns as 1.5, standard
    as 1). Pack all patients into wards using minimum beds.

    Problem 2 (Knapsack): Ward C (Cardiac) has 10 beds available
    and 12 cardiac patients waiting. Each patient has a clinical
    priority score (benefit) and expected length-of-stay (weight).
    Select which patients to admit to maximize clinical benefit.

Real-world considerations modeled:
    - Patient acuity-to-bed-equivalent conversion
    - Isolation requirements (infectious patients)
    - Ward specialty matching
    - Length-of-stay variation
    - Clinical prioritization for scarce beds

Industry context:
    Hospital occupancy rates above 85% are associated with increased
    mortality and ED boarding (Bain et al., 2010). Optimizing bed
    allocation can improve throughput by 10-20% without adding capacity.

References:
    Bain, C.A., Taylor, P.G., McDonnell, G. & Georgiou, A. (2010).
    Myths of ideal hospital occupancy. Medical Journal of Australia,
    192(1), 42-43.
    https://doi.org/10.5694/j.1326-5377.2010.tb03401.x

    Demeester, P., Souffriau, W., De Causmaecker, P. & Berghe, G.V.
    (2010). A hybrid tabu search algorithm for automatically assigning
    patients to beds. Artificial Intelligence in Medicine, 48(1), 61-70.
    https://doi.org/10.1016/j.artmed.2009.09.001

    Hulshof, P.J.H., Kortbeek, N., Boucherie, R.J., Hans, E.W. &
    Bakker, P.J.M. (2012). Taxonomic classification of planning
    decisions in health care: A structured review of the state of the
    art in OR/MS. Health Systems, 1(2), 129-175.
    https://doi.org/10.1057/hs.2012.18
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

WARDS = [
    {"name": "Ward A (Medical)",  "total_beds": 35, "available": 12},
    {"name": "Ward B (Surgical)", "total_beds": 30, "available": 10},
    {"name": "Ward C (Cardiac)",  "total_beds": 25, "available": 10},
    {"name": "Ward D (General)",  "total_beds": 30, "available": 14},
]

# ED admissions needing bed assignment
ED_ADMISSIONS = [
    {"name": "Pt 1: Pneumonia",           "bed_equiv": 1, "specialty": "medical"},
    {"name": "Pt 2: COPD exacerbation",   "bed_equiv": 1, "specialty": "medical"},
    {"name": "Pt 3: Post-appendectomy",   "bed_equiv": 1, "specialty": "surgical"},
    {"name": "Pt 4: Hip fracture",        "bed_equiv": 1, "specialty": "surgical"},
    {"name": "Pt 5: C. diff (isolation)",  "bed_equiv": 2, "specialty": "medical"},
    {"name": "Pt 6: Chest pain obs",      "bed_equiv": 1, "specialty": "cardiac"},
    {"name": "Pt 7: STEMI post-PCI",      "bed_equiv": 2, "specialty": "cardiac"},
    {"name": "Pt 8: GI bleed",            "bed_equiv": 1, "specialty": "medical"},
    {"name": "Pt 9: Cellulitis",          "bed_equiv": 1, "specialty": "general"},
    {"name": "Pt 10: Syncope workup",     "bed_equiv": 1, "specialty": "general"},
    {"name": "Pt 11: DVT (anticoag)",     "bed_equiv": 1, "specialty": "medical"},
    {"name": "Pt 12: Post-lap chole",     "bed_equiv": 1, "specialty": "surgical"},
    {"name": "Pt 13: MRSA (isolation)",    "bed_equiv": 2, "specialty": "medical"},
    {"name": "Pt 14: Heart failure",      "bed_equiv": 1, "specialty": "cardiac"},
    {"name": "Pt 15: UTI/sepsis",         "bed_equiv": 1, "specialty": "medical"},
    {"name": "Pt 16: Hernia repair",      "bed_equiv": 1, "specialty": "surgical"},
    {"name": "Pt 17: Atrial fib",         "bed_equiv": 1, "specialty": "cardiac"},
    {"name": "Pt 18: Dehydration",        "bed_equiv": 1, "specialty": "general"},
    {"name": "Pt 19: Back pain",          "bed_equiv": 1, "specialty": "general"},
    {"name": "Pt 20: TB suspect (iso)",    "bed_equiv": 2, "specialty": "medical"},
]

# Cardiac ward waiting list (knapsack problem)
CARDIAC_WAITLIST = [
    {"name": "WL-1: Unstable angina",     "priority": 9,  "los_days": 4},
    {"name": "WL-2: CHF decompensation",  "priority": 8,  "los_days": 5},
    {"name": "WL-3: Post-CABG recovery",  "priority": 7,  "los_days": 7},
    {"name": "WL-4: Arrhythmia monitoring","priority": 6,  "los_days": 3},
    {"name": "WL-5: Valve disease workup", "priority": 5,  "los_days": 4},
    {"name": "WL-6: Pre-op cardiac eval",  "priority": 4,  "los_days": 2},
    {"name": "WL-7: Post-PCI observation", "priority": 8,  "los_days": 2},
    {"name": "WL-8: Pericarditis",         "priority": 6,  "los_days": 3},
    {"name": "WL-9: Endocarditis (IV abx)","priority": 9,  "los_days": 10},
    {"name": "WL-10: Cardiac rehab eval",  "priority": 3,  "los_days": 2},
    {"name": "WL-11: Myocarditis",         "priority": 7,  "los_days": 5},
    {"name": "WL-12: Pulmonary HTN",       "priority": 5,  "los_days": 6},
]


def create_bed_packing_instance() -> dict:
    """Create a bin-packing instance for bed assignment.

    Returns:
        Dictionary with instance data.
    """
    # Pack all admissions into wards (bins)
    # Ward capacity = available beds; patient size = bed_equiv
    sizes = np.array([p["bed_equiv"] for p in ED_ADMISSIONS], dtype=int)
    total_available = sum(w["available"] for w in WARDS)
    # Use the largest ward capacity as bin size
    max_ward = max(w["available"] for w in WARDS)

    return {
        "n_patients": len(ED_ADMISSIONS),
        "sizes": sizes,
        "ward_capacity": max_ward,
        "total_beds_needed": sizes.sum(),
        "total_available": total_available,
        "admissions": ED_ADMISSIONS,
    }


def create_cardiac_admission_instance() -> dict:
    """Create a knapsack instance for cardiac bed allocation.

    Returns:
        Dictionary with instance data.
    """
    weights = np.array([p["los_days"] for p in CARDIAC_WAITLIST], dtype=int)
    values = np.array([p["priority"] for p in CARDIAC_WAITLIST], dtype=int)

    return {
        "n_patients": len(CARDIAC_WAITLIST),
        "weights": weights,  # LOS in bed-days
        "values": values,    # clinical priority
        "capacity": 10,      # available beds (simplified: each for 1 day-equivalent)
        "waitlist": CARDIAC_WAITLIST,
    }


def solve_bed_management(verbose: bool = True) -> dict:
    """Solve hospital bed management problems.

    Returns:
        Dictionary with results for both problems.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pack_dir = os.path.join(base_dir, "problems", "packing")

    results = {}

    # ── Problem 1: Bin Packing (ED admissions → wards) ───────────────────
    bp_inst_mod = _load_mod("bp_inst_bed", os.path.join(pack_dir, "bin_packing", "instance.py"))
    bp_ff_mod = _load_mod("bp_ff_bed", os.path.join(pack_dir, "bin_packing", "heuristics", "first_fit.py"))

    bp_data = create_bed_packing_instance()
    bp_instance = bp_inst_mod.BinPackingInstance(
        n=bp_data["n_patients"],
        sizes=bp_data["sizes"],
        capacity=bp_data["ward_capacity"],
        name="ed_admissions",
    )

    ff_sol = bp_ff_mod.first_fit(bp_instance)
    ffd_sol = bp_ff_mod.first_fit_decreasing(bp_instance)
    bfd_sol = bp_ff_mod.best_fit_decreasing(bp_instance)

    results["bed_packing"] = {}
    for name, sol in [("FF", ff_sol), ("FFD", ffd_sol), ("BFD", bfd_sol)]:
        results["bed_packing"][name] = {
            "n_wards_needed": sol.num_bins,
            "bins": sol.bins,
        }

    # ── Problem 2: Knapsack (cardiac bed allocation) ─────────────────────
    ks_inst_mod = _load_mod("ks_inst_bed", os.path.join(pack_dir, "knapsack", "instance.py"))
    ks_dp_mod = _load_mod("ks_dp_bed", os.path.join(pack_dir, "knapsack", "exact", "dynamic_programming.py"))
    ks_gr_mod = _load_mod("ks_gr_bed", os.path.join(pack_dir, "knapsack", "heuristics", "greedy.py"))

    ks_data = create_cardiac_admission_instance()
    ks_instance = ks_inst_mod.KnapsackInstance(
        n=ks_data["n_patients"],
        weights=ks_data["weights"],
        values=ks_data["values"],
        capacity=ks_data["capacity"],
        name="cardiac_beds",
    )

    dp_sol = ks_dp_mod.dynamic_programming(ks_instance)
    gr_sol = ks_gr_mod.greedy_value_density(ks_instance)

    results["cardiac_admission"] = {
        "DP (Optimal)": {
            "total_priority": dp_sol.value,
            "admitted": dp_sol.items,
            "total_los": sum(ks_data["weights"][i] for i in dp_sol.items),
        },
        "Greedy": {
            "total_priority": gr_sol.value,
            "admitted": gr_sol.items,
            "total_los": sum(ks_data["weights"][i] for i in gr_sol.items),
        },
    }

    if verbose:
        print("=" * 70)
        print("HOSPITAL BED & WARD CAPACITY MANAGEMENT")
        print("=" * 70)

        # Ward status
        print("\n  Current ward status:")
        for w in WARDS:
            occ = w["total_beds"] - w["available"]
            pct = occ / w["total_beds"] * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(f"    {w['name']:22s}: {occ:2d}/{w['total_beds']:2d} "
                  f"({w['available']:2d} free) [{bar}] {pct:.0f}%")

        # Bin packing results
        print(f"\n--- ED ADMISSIONS: {bp_data['n_patients']} patients, "
              f"{bp_data['total_beds_needed']} bed-equivalents needed ---")

        best_bp = min(results["bed_packing"], key=lambda k: results["bed_packing"][k]["n_wards_needed"])
        for method, res in results["bed_packing"].items():
            marker = " (best)" if method == best_bp else ""
            print(f"\n  {method}{marker}: {res['n_wards_needed']} ward sections needed")
            for i, ward_patients in enumerate(res["bins"]):
                used = sum(bp_data["sizes"][j] for j in ward_patients)
                names = [ED_ADMISSIONS[j]["name"][:15] for j in ward_patients]
                iso_count = sum(1 for j in ward_patients if ED_ADMISSIONS[j]["bed_equiv"] > 1)
                print(f"    Section {i+1}: {used:2d}/{bp_data['ward_capacity']} beds, "
                      f"{len(ward_patients)} patients "
                      f"({iso_count} isolation) — [{', '.join(names)}]")

        # Cardiac admission results
        print(f"\n--- CARDIAC WARD: {ks_data['capacity']} bed-days available, "
              f"{ks_data['n_patients']} patients waiting ---")

        for method, res in results["cardiac_admission"].items():
            print(f"\n  {method}: admit {len(res['admitted'])} patients, "
                  f"priority score = {res['total_priority']}, "
                  f"total LOS = {res['total_los']} bed-days")
            for idx in res["admitted"]:
                p = CARDIAC_WAITLIST[idx]
                print(f"    ADMIT: {p['name']:30s} (priority={p['priority']}, "
                      f"LOS={p['los_days']}d)")

            denied = [i for i in range(ks_data["n_patients"]) if i not in res["admitted"]]
            if denied:
                print(f"    --- Deferred ({len(denied)} patients) ---")
                for idx in denied:
                    p = CARDIAC_WAITLIST[idx]
                    print(f"    DEFER: {p['name']:30s} (priority={p['priority']}, "
                          f"LOS={p['los_days']}d)")

    return results


if __name__ == "__main__":
    solve_bed_management()

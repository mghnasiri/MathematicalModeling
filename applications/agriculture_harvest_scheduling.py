"""
Real-World Application: Harvest Processing Line Scheduling.

Domain: Post-harvest grain handling / Grain elevator operations
Model: Permutation Flow Shop (Fm | prmu | Cmax)

Scenario:
    A grain elevator receives 10 grain lots during the fall harvest.
    Each lot must pass through 4 sequential processing stages:

    1. Field cutting & loading — combine harvesting and truck loading
    2. Transport to silo — trucking from field to grain elevator
    3. Cleaning & grading — removing debris, foreign material, grading
    4. Drying & storage — reducing moisture to safe storage levels

    The order in which lots are processed through the entire line is
    fixed (permutation schedule). The objective is to find the lot
    sequence that minimizes the total completion time (makespan),
    ensuring all grain is safely stored before adverse weather
    (rain, frost) degrades quality.

    Processing times vary by lot size, grain type, moisture content
    at harvest, and distance from field to elevator.

Real-world considerations modeled:
    - Sequential processing stages (flow shop structure)
    - Variable processing times by grain type and lot size
    - Permutation constraint (lots processed in same order at each stage)
    - Makespan minimization (race against weather deterioration)
    - Comparison of constructive heuristic vs. metaheuristic

Industry context:
    Approximately 10% of global grain harvest is lost to post-harvest
    handling delays and quality degradation (FAO, 2019). In North
    America, harvest timeliness costs average $5-15 per acre per day
    of delay. Efficient scheduling of harvest processing can reduce
    total handling time by 15-25%, preserving grain quality and
    reducing drying costs (which represent 30-40% of post-harvest
    expense).

References:
    FAO (2019). The State of Food and Agriculture: Moving Forward on
    Food Loss and Waste Reduction. Rome: Food and Agriculture
    Organization of the United Nations.

    Nawaz, M., Enscore, E.E. & Ham, I. (1983). A heuristic algorithm
    for the m-machine, n-job flow-shop sequencing problem. Omega,
    11(1), 91-95. https://doi.org/10.1016/0305-0483(83)90088-9

    Ruiz, R. & Stuetzle, T. (2007). A simple and effective iterated
    greedy algorithm for the permutation flowshop scheduling problem.
    European Journal of Operational Research, 177(3), 2033-2049.
    https://doi.org/10.1016/j.ejor.2005.12.009

    Bochtis, D.D., Sorensen, C.G. & Busato, P. (2014). Advances in
    agricultural machinery management: A review. Biosystems Engineering,
    126, 69-81. https://doi.org/10.1016/j.biosystemseng.2014.07.012
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

# 4 processing stages
STAGES = [
    "Field Cutting & Loading",
    "Transport to Silo",
    "Cleaning & Grading",
    "Drying & Storage",
]

# 10 grain lots with characteristics affecting processing times
GRAIN_LOTS = [
    {"name": "Lot A - Corn (North)",      "crop": "Corn",     "tonnes": 180,
     "moisture": 22.0, "distance_km": 12, "debris": "low"},
    {"name": "Lot B - Wheat (East)",       "crop": "Wheat",    "tonnes": 120,
     "moisture": 16.5, "distance_km": 8,  "debris": "medium"},
    {"name": "Lot C - Soybean (South)",    "crop": "Soybean",  "tonnes": 95,
     "moisture": 15.0, "distance_km": 15, "debris": "high"},
    {"name": "Lot D - Corn (Central)",     "crop": "Corn",     "tonnes": 220,
     "moisture": 24.0, "distance_km": 5,  "debris": "low"},
    {"name": "Lot E - Barley (West)",      "crop": "Barley",   "tonnes": 85,
     "moisture": 14.5, "distance_km": 20, "debris": "low"},
    {"name": "Lot F - Wheat (North)",      "crop": "Wheat",    "tonnes": 150,
     "moisture": 17.0, "distance_km": 10, "debris": "medium"},
    {"name": "Lot G - Canola (East)",      "crop": "Canola",   "tonnes": 60,
     "moisture": 12.0, "distance_km": 18, "debris": "high"},
    {"name": "Lot H - Corn (South)",       "crop": "Corn",     "tonnes": 200,
     "moisture": 23.5, "distance_km": 7,  "debris": "medium"},
    {"name": "Lot I - Soybean (Central)",  "crop": "Soybean",  "tonnes": 110,
     "moisture": 14.0, "distance_km": 3,  "debris": "low"},
    {"name": "Lot J - Oats (West)",        "crop": "Oats",     "tonnes": 70,
     "moisture": 13.5, "distance_km": 22, "debris": "medium"},
]

# Cost parameters
DRYING_COST_PER_TONNE_PER_POINT = 3.50  # $/tonne per moisture point removed
QUALITY_LOSS_PER_HOUR_DELAY = 0.10       # % quality degradation per hour
AVG_GRAIN_VALUE_PER_TONNE = 280.0        # average market value


def _compute_processing_times() -> np.ndarray:
    """Compute processing time matrix (m x n) for all lots and stages.

    Stage 1 (Cutting): depends on tonnes and crop type
    Stage 2 (Transport): depends on distance and tonnes
    Stage 3 (Cleaning): depends on tonnes and debris level
    Stage 4 (Drying): depends on tonnes and moisture content

    Returns:
        Processing time matrix of shape (4, 10) in minutes.
    """
    n = len(GRAIN_LOTS)
    m = len(STAGES)
    pt = np.zeros((m, n), dtype=int)

    # Crop-specific cutting rate (tonnes per hour)
    cutting_rates = {"Corn": 35, "Wheat": 45, "Soybean": 30,
                     "Barley": 50, "Canola": 25, "Oats": 55}

    for j, lot in enumerate(GRAIN_LOTS):
        # Stage 1: Field cutting (minutes)
        rate = cutting_rates.get(lot["crop"], 40)
        pt[0, j] = int((lot["tonnes"] / rate) * 60)

        # Stage 2: Transport (minutes) - round trip at 40 km/h avg
        trips = max(1, lot["tonnes"] // 25)  # 25-tonne truck loads
        pt[1, j] = int(trips * (2 * lot["distance_km"] / 40) * 60)

        # Stage 3: Cleaning & grading (minutes)
        cleaning_rate = 50  # tonnes per hour base rate
        debris_factor = {"low": 1.0, "medium": 1.3, "high": 1.7}
        factor = debris_factor.get(lot["debris"], 1.0)
        pt[2, j] = int((lot["tonnes"] / cleaning_rate) * factor * 60)

        # Stage 4: Drying & storage (minutes)
        # Target moisture: 14% for corn, 13.5% for wheat/barley/oats,
        # 12% for soybean, 10% for canola
        target_moisture = {"Corn": 14.0, "Wheat": 13.5, "Soybean": 12.0,
                          "Barley": 13.5, "Canola": 10.0, "Oats": 13.5}
        target = target_moisture.get(lot["crop"], 13.0)
        points_to_remove = max(0, lot["moisture"] - target)
        # Drying rate: ~1 point per hour per 20 tonnes
        drying_hours = (points_to_remove * lot["tonnes"]) / 20.0 / 4.0
        # Storage handling: 15 min per 50 tonnes
        storage_min = (lot["tonnes"] / 50) * 15
        pt[3, j] = int(drying_hours * 60 + storage_min)

    return pt


def create_harvest_instance() -> dict:
    """Create a flow shop instance for harvest processing.

    Returns:
        Dictionary with processing times, lot details, and stage info.
    """
    processing_times = _compute_processing_times()
    n = len(GRAIN_LOTS)
    m = len(STAGES)

    total_tonnes = sum(lot["tonnes"] for lot in GRAIN_LOTS)

    return {
        "n_lots": n,
        "n_stages": m,
        "processing_times": processing_times,
        "lots": GRAIN_LOTS,
        "stages": STAGES,
        "total_tonnes": total_tonnes,
    }


def solve_harvest_scheduling(verbose: bool = True) -> dict:
    """Solve harvest processing scheduling using flow shop algorithms.

    Applies NEH (constructive heuristic) and Iterated Greedy
    (metaheuristic) to find the best lot processing sequence.

    Args:
        verbose: Whether to print detailed results.

    Returns:
        Dictionary with scheduling results from each method.
    """
    data = create_harvest_instance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs_dir = os.path.join(
        base_dir, "problems", "scheduling", "flow_shop"
    )

    fs_inst_mod = _load_mod(
        "fs_inst_harv", os.path.join(fs_dir, "instance.py")
    )
    fs_neh_mod = _load_mod(
        "fs_neh_harv", os.path.join(fs_dir, "heuristics", "neh.py")
    )
    fs_ig_mod = _load_mod(
        "fs_ig_harv",
        os.path.join(fs_dir, "metaheuristics", "iterated_greedy.py"),
    )

    instance = fs_inst_mod.FlowShopInstance(
        n=data["n_lots"],
        m=data["n_stages"],
        processing_times=data["processing_times"],
    )

    # NEH constructive heuristic
    neh_sol = fs_neh_mod.neh(instance)

    # Iterated Greedy metaheuristic
    ig_sol = fs_ig_mod.iterated_greedy(
        instance, max_iterations=500, d=3, seed=42
    )

    results = {}
    for method_name, sol in [("NEH", neh_sol), ("Iterated Greedy", ig_sol)]:
        # Compute completion times for detailed analysis
        ct = fs_inst_mod.compute_completion_times(instance, sol.permutation)

        results[method_name] = {
            "makespan": sol.makespan,
            "sequence": sol.permutation,
            "completion_times": ct,
        }

    if verbose:
        total_processing = data["processing_times"].sum()
        print("=" * 70)
        print("HARVEST PROCESSING LINE SCHEDULING (Flow Shop)")
        print(f"  {data['n_lots']} grain lots, {data['n_stages']} stages, "
              f"{data['total_tonnes']:,d} total tonnes")
        print(f"  Stages: {' -> '.join(data['stages'])}")
        print("=" * 70)

        # Lot details
        print("\n  Grain lots:")
        print(f"    {'Lot':<30s} {'Crop':<10s} {'Tonnes':>6s} "
              f"{'Moist%':>6s} {'Dist':>5s} {'Debris':<6s}")
        for lot in GRAIN_LOTS:
            print(f"    {lot['name']:<30s} {lot['crop']:<10s} "
                  f"{lot['tonnes']:>6d} {lot['moisture']:>5.1f}% "
                  f"{lot['distance_km']:>4d}km {lot['debris']:<6s}")

        # Processing time matrix
        print(f"\n  Processing times (minutes):")
        print(f"    {'Stage':<25s}", end="")
        for j in range(data["n_lots"]):
            print(f" L{j:1d}", end="")
        print(f"  {'Total':>5s}")
        for i, stage in enumerate(STAGES):
            print(f"    {stage:<25s}", end="")
            row_total = 0
            for j in range(data["n_lots"]):
                pt = data["processing_times"][i, j]
                print(f" {pt:3d}", end="")
                row_total += pt
            print(f"  {row_total:>5d}")

        # Results per method
        best_method = min(results, key=lambda k: results[k]["makespan"])

        for method_name, res in results.items():
            makespan_hrs = res["makespan"] / 60
            marker = " << BEST" if method_name == best_method else ""
            print(f"\n  {method_name}{marker}: makespan = {res['makespan']} min "
                  f"({makespan_hrs:.1f} hours)")

            seq = res["sequence"]
            print("  Processing order:")
            for rank, lot_id in enumerate(seq):
                lot = GRAIN_LOTS[lot_id]
                finish = res["completion_times"][-1, rank]
                print(f"    {rank+1:2d}. {lot['name']:<30s} "
                      f"(finishes at t={finish:>5d} min = {finish/60:.1f}h)")

        # Improvement analysis
        neh_ms = results["NEH"]["makespan"]
        ig_ms = results["Iterated Greedy"]["makespan"]
        if ig_ms < neh_ms:
            improvement = neh_ms - ig_ms
            pct = improvement / neh_ms * 100
            hours_saved = improvement / 60
            # Economic impact of time saved
            quality_value = (hours_saved * QUALITY_LOSS_PER_HOUR_DELAY / 100
                            * data["total_tonnes"] * AVG_GRAIN_VALUE_PER_TONNE)
            print(f"\n  {'─' * 50}")
            print(f"  Iterated Greedy improves over NEH by {improvement} min "
                  f"({pct:.1f}%)")
            print(f"  Time saved: {hours_saved:.1f} hours")
            print(f"  Quality preservation value: ${quality_value:,.0f}")
        elif neh_ms == ig_ms:
            print(f"\n  {'─' * 50}")
            print(f"  Both methods found the same makespan: {neh_ms} min")

        # Drying cost estimate for best solution
        total_drying_cost = 0
        for lot in GRAIN_LOTS:
            target = {"Corn": 14.0, "Wheat": 13.5, "Soybean": 12.0,
                      "Barley": 13.5, "Canola": 10.0, "Oats": 13.5}
            t = target.get(lot["crop"], 13.0)
            points = max(0, lot["moisture"] - t)
            total_drying_cost += points * lot["tonnes"] * DRYING_COST_PER_TONNE_PER_POINT

        print(f"  Estimated total drying cost: ${total_drying_cost:,.0f}")
        print(f"  Total grain value: "
              f"${data['total_tonnes'] * AVG_GRAIN_VALUE_PER_TONNE:,.0f}")

    return {"harvest_scheduling": results}


if __name__ == "__main__":
    solve_harvest_scheduling()

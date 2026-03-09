"""
Real-World Application: Cloud Resource Packing & VM Placement.

Domain: Cloud computing / Data center capacity planning
Models: Bin Packing + 0-1 Knapsack

Scenario:
    A cloud provider manages physical servers, each with 64 GB RAM.
    Customers request VMs of various sizes. Two problems:

    1. Bin Packing: Pack all 20 VM requests into minimum physical
       servers to minimize hardware cost.
    2. Knapsack: Given a single server with limited capacity, select
       the most profitable VMs to host (maximize revenue per server).

Real-world considerations modeled:
    - Heterogeneous VM sizes (small, medium, large, xlarge)
    - Server capacity constraints (RAM-limited)
    - Revenue optimization per physical server
    - Fragmentation analysis (wasted capacity)

Industry context:
    Server utilization in data centers averages 12-18% (Barroso et al., 2013).
    Bin-packing-based VM placement can improve utilization to 60-85%,
    saving millions in hardware and energy costs.

References:
    Barroso, L.A., Clidaras, J. & Hölzle, U. (2013). The Datacenter
    as a Computer: Designing Warehouse-Scale Machines. 2nd ed.
    Morgan & Claypool.
    https://doi.org/10.2200/S00516ED2V01Y201306CAC024

    Beloglazov, A., Abawajy, J. & Buyya, R. (2012). Energy-aware
    resource allocation heuristics for efficient management of data
    centers for cloud computing. Future Generation Computer Systems,
    28(5), 755-768.
    https://doi.org/10.1016/j.future.2011.04.017

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


# ── Domain Data ──────────────────────────────────────────────────────────────

SERVER_CAPACITY_GB = 64  # GB RAM per physical server

# VM size tiers (typical cloud offerings)
VM_TIERS = {
    "nano":   {"ram_gb": 1,  "monthly_revenue": 15},
    "micro":  {"ram_gb": 2,  "monthly_revenue": 25},
    "small":  {"ram_gb": 4,  "monthly_revenue": 45},
    "medium": {"ram_gb": 8,  "monthly_revenue": 80},
    "large":  {"ram_gb": 16, "monthly_revenue": 150},
    "xlarge": {"ram_gb": 32, "monthly_revenue": 280},
}


def create_vm_requests(n_requests: int = 20, seed: int = 42) -> dict:
    """Generate realistic VM placement requests.

    Args:
        n_requests: Number of VM requests.
        seed: Random seed.

    Returns:
        Dictionary with VM request data.
    """
    rng = np.random.default_rng(seed)

    # Distribution: many small, fewer large (typical cloud workload)
    tier_probs = {
        "nano": 0.10, "micro": 0.20, "small": 0.30,
        "medium": 0.20, "large": 0.15, "xlarge": 0.05,
    }
    tier_names = list(tier_probs.keys())
    probs = list(tier_probs.values())

    requests = []
    for i in range(n_requests):
        tier = rng.choice(tier_names, p=probs)
        vm = VM_TIERS[tier]
        requests.append({
            "id": f"vm-{i:03d}",
            "tier": tier,
            "ram_gb": vm["ram_gb"],
            "monthly_revenue": vm["monthly_revenue"],
            "customer": f"customer-{rng.integers(100, 999)}",
        })

    return {
        "n_requests": n_requests,
        "requests": requests,
        "server_capacity": SERVER_CAPACITY_GB,
    }


def solve_vm_packing(verbose: bool = True) -> dict:
    """Solve VM placement using bin packing and knapsack.

    Returns:
        Dictionary with results.
    """
    data = create_vm_requests(n_requests=20, seed=42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pack_dir = os.path.join(base_dir, "problems", "packing")

    results = {}

    # ── Bin Packing: Pack ALL VMs into minimum servers ───────────────────
    bp_inst_mod = _load_mod(
        "bp_inst_cloud",
        os.path.join(pack_dir, "bin_packing", "instance.py"),
    )
    bp_ff_mod = _load_mod(
        "bp_ff_cloud",
        os.path.join(pack_dir, "bin_packing", "heuristics", "first_fit.py"),
    )

    sizes = np.array([r["ram_gb"] for r in data["requests"]], dtype=int)

    bp_instance = bp_inst_mod.BinPackingInstance(
        n=data["n_requests"],
        sizes=sizes,
        capacity=data["server_capacity"],
        name="vm_placement",
    )

    ff_sol = bp_ff_mod.first_fit(bp_instance)
    ffd_sol = bp_ff_mod.first_fit_decreasing(bp_instance)
    bfd_sol = bp_ff_mod.best_fit_decreasing(bp_instance)

    results["bin_packing"] = {}
    for name, sol in [("FF", ff_sol), ("FFD", ffd_sol), ("BFD", bfd_sol)]:
        results["bin_packing"][name] = {
            "n_servers": sol.num_bins,
            "bins": sol.bins,
        }

    # ── Knapsack: Best VMs for a single server ───────────────────────────
    ks_inst_mod = _load_mod(
        "ks_inst_cloud",
        os.path.join(pack_dir, "knapsack", "instance.py"),
    )
    ks_dp_mod = _load_mod(
        "ks_dp_cloud",
        os.path.join(pack_dir, "knapsack", "exact", "dynamic_programming.py"),
    )
    ks_gr_mod = _load_mod(
        "ks_gr_cloud",
        os.path.join(pack_dir, "knapsack", "heuristics", "greedy.py"),
    )

    weights = np.array([r["ram_gb"] for r in data["requests"]], dtype=int)
    values = np.array([r["monthly_revenue"] for r in data["requests"]], dtype=int)

    ks_instance = ks_inst_mod.KnapsackInstance(
        n=data["n_requests"],
        weights=weights,
        values=values,
        capacity=data["server_capacity"],
        name="vm_revenue",
    )

    dp_sol = ks_dp_mod.dynamic_programming(ks_instance)
    gr_sol = ks_gr_mod.greedy_value_density(ks_instance)

    results["knapsack"] = {
        "DP (Optimal)": {
            "revenue": dp_sol.value,
            "selected": dp_sol.items,
            "total_ram": sum(weights[i] for i in dp_sol.items),
        },
        "Greedy": {
            "revenue": gr_sol.value,
            "selected": gr_sol.items,
            "total_ram": sum(weights[i] for i in gr_sol.items),
        },
    }

    if verbose:
        print("=" * 70)
        print("CLOUD VM PLACEMENT & REVENUE OPTIMIZATION")
        print(f"  {data['n_requests']} VM requests, server capacity = "
              f"{data['server_capacity']} GB RAM")
        print("=" * 70)

        # VM request summary
        tier_counts = {}
        for r in data["requests"]:
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        total_ram = sum(r["ram_gb"] for r in data["requests"])
        total_revenue = sum(r["monthly_revenue"] for r in data["requests"])
        print(f"\n  VM requests: {tier_counts}")
        print(f"  Total RAM needed: {total_ram} GB")
        print(f"  Total monthly revenue: ${total_revenue}")
        min_servers = max(1, -(-total_ram // data["server_capacity"]))
        print(f"  Theoretical minimum servers: {min_servers}")

        # Bin Packing results
        print("\n--- BIN PACKING: Pack ALL VMs (minimize servers) ---")
        for method, res in results["bin_packing"].items():
            utilization = total_ram / (res["n_servers"] * data["server_capacity"]) * 100
            print(f"\n  {method}: {res['n_servers']} servers "
                  f"(utilization = {utilization:.1f}%)")
            for i, bin_items in enumerate(res["bins"]):
                bin_ram = sum(sizes[j] for j in bin_items)
                vms = [f"{data['requests'][j]['tier']}" for j in bin_items]
                free = data["server_capacity"] - bin_ram
                print(f"    Server {i+1}: {bin_ram:2d}/{data['server_capacity']} GB "
                      f"({free:2d} free) — [{', '.join(vms)}]")

        # Knapsack results
        print("\n--- KNAPSACK: Best VMs for ONE server (maximize revenue) ---")
        for method, res in results["knapsack"].items():
            print(f"\n  {method}: ${res['revenue']}/month, "
                  f"{res['total_ram']}/{data['server_capacity']} GB used")
            selected_vms = res["selected"]
            for j in selected_vms:
                r = data["requests"][j]
                density = r["monthly_revenue"] / r["ram_gb"]
                print(f"    {r['id']} ({r['tier']:6s}): "
                      f"{r['ram_gb']:2d} GB, ${r['monthly_revenue']}/mo "
                      f"(${density:.1f}/GB)")

    return results


if __name__ == "__main__":
    solve_vm_packing()

"""
Deterministic Farm-to-Market Delivery Routing

Solves the farm delivery problem as a standard CVRP using:
1. Clarke-Wright savings heuristic
2. Sweep algorithm (multi-start)

Complexity:
    - Clarke-Wright: O(n^2 log n)
    - Sweep: O(n log n) per start

References:
    Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. Operations Research,
    12(4), 568-581. https://doi.org/10.1287/opre.12.4.568

    Gillett, B.E. & Miller, L.R. (1974). A heuristic algorithm for the
    vehicle-dispatch problem. Operations Research, 22(2), 340-349.
    https://doi.org/10.1287/opre.22.2.340
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
    "fd_inst_det",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
FarmDeliveryInstance = _inst.FarmDeliveryInstance
FarmDeliverySolution = _inst.FarmDeliverySolution


def _get_cvrp_modules():
    """Load CVRP solver modules from the routing family."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    cvrp_dir = os.path.join(base_dir, "problems", "routing", "cvrp")

    cvrp_inst = _load_mod("cvrp_inst_fd", os.path.join(cvrp_dir, "instance.py"))
    cw_mod = _load_mod(
        "cvrp_cw_fd", os.path.join(cvrp_dir, "heuristics", "clarke_wright.py")
    )
    sweep_mod = _load_mod(
        "cvrp_sweep_fd", os.path.join(cvrp_dir, "heuristics", "sweep.py")
    )
    return cvrp_inst, cw_mod, sweep_mod


def _build_cvrp_instance(instance: FarmDeliveryInstance, seed: int = 42):
    """Convert FarmDeliveryInstance to CVRPInstance."""
    cvrp_inst, _, _ = _get_cvrp_modules()
    coords = instance.generate_coordinates(seed=seed)
    dist = instance.generate_distance_matrix(coords)
    demands = np.array(
        [dp.base_demand_kg for dp in instance.delivery_points], dtype=float
    )
    return cvrp_inst.CVRPInstance(
        n=instance.n_customers,
        coords=coords,
        demands=demands,
        capacity=instance.truck_capacity_kg,
        distance_matrix=dist,
        name=f"farm_{instance.name}",
    ), coords


def clarke_wright_delivery(
    instance: FarmDeliveryInstance,
    seed: int = 42,
) -> FarmDeliverySolution:
    """Solve farm delivery using Clarke-Wright savings.

    Args:
        instance: FarmDeliveryInstance to solve.
        seed: Random seed for coordinate generation.

    Returns:
        FarmDeliverySolution with Clarke-Wright routes.
    """
    _, cw_mod, _ = _get_cvrp_modules()
    cvrp, _ = _build_cvrp_instance(instance, seed=seed)
    sol = cw_mod.clarke_wright_savings(cvrp)

    return FarmDeliverySolution(
        routes=sol.routes,
        total_distance=sol.distance,
        n_vehicles=len(sol.routes),
        method="Clarke-Wright",
    )


def sweep_delivery(
    instance: FarmDeliveryInstance,
    num_starts: int = 12,
    seed: int = 42,
) -> FarmDeliverySolution:
    """Solve farm delivery using multi-start sweep algorithm.

    Args:
        instance: FarmDeliveryInstance to solve.
        num_starts: Number of starting angles to try.
        seed: Random seed for coordinate generation.

    Returns:
        FarmDeliverySolution with sweep routes.
    """
    _, _, sweep_mod = _get_cvrp_modules()
    cvrp, _ = _build_cvrp_instance(instance, seed=seed)
    sol = sweep_mod.sweep_multistart(cvrp, num_starts=num_starts)

    return FarmDeliverySolution(
        routes=sol.routes,
        total_distance=sol.distance,
        n_vehicles=len(sol.routes),
        method="Sweep",
    )


if __name__ == "__main__":
    inst = FarmDeliveryInstance.quebec_cooperative()
    print("=== Farm-to-Market Delivery (Deterministic CVRP) ===\n")

    sol_cw = clarke_wright_delivery(inst)
    print(f"Clarke-Wright: {sol_cw}")
    for i, route in enumerate(sol_cw.routes):
        print(f"  Route {i+1}: {route}")

    sol_sw = sweep_delivery(inst)
    print(f"\nSweep: {sol_sw}")

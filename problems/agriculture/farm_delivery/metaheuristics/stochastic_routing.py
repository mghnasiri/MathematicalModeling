"""
Stochastic Farm-to-Market Delivery Routing

Solves the farm delivery problem as a Stochastic VRP using:
1. Chance-constrained Clarke-Wright savings
2. Simulated Annealing with recourse penalty

Routes are designed to maintain P(overflow) <= alpha under demand
uncertainty, avoiding costly mid-route depot returns.

Complexity:
    - CC-Clarke-Wright: O(n^2 * S)
    - SA: O(max_iter * n * S)

References:
    Bertsimas, D.J. (1992). A vehicle routing problem with stochastic
    demand. Operations Research, 40(3), 574-585.
    https://doi.org/10.1287/opre.40.3.574
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
    "fd_inst_sto",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
FarmDeliveryInstance = _inst.FarmDeliveryInstance
FarmDeliverySolution = _inst.FarmDeliverySolution


def _get_svrp_modules():
    """Load Stochastic VRP solver modules."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )))
    svrp_dir = os.path.join(
        base_dir, "problems", "stochastic_robust", "stochastic_vrp"
    )

    svrp_inst = _load_mod("svrp_inst_fd", os.path.join(svrp_dir, "instance.py"))
    cc_cw = _load_mod(
        "svrp_cccw_fd",
        os.path.join(svrp_dir, "heuristics", "chance_constrained_cw.py"),
    )
    sa_mod = _load_mod(
        "svrp_sa_fd",
        os.path.join(svrp_dir, "metaheuristics", "simulated_annealing.py"),
    )
    return svrp_inst, cc_cw, sa_mod


def _build_svrp_instance(instance: FarmDeliveryInstance, seed: int = 42):
    """Convert FarmDeliveryInstance to StochasticVRPInstance."""
    svrp_inst, _, _ = _get_svrp_modules()
    coords = instance.generate_coordinates(seed=seed)
    demand_scenarios = instance.generate_demand_scenarios(seed=seed)
    demands = np.array(
        [dp.base_demand_kg for dp in instance.delivery_points], dtype=float
    )
    n_vehicles = int(np.ceil(demands.sum() / instance.truck_capacity_kg)) + 1

    return svrp_inst.StochasticVRPInstance(
        n_customers=instance.n_customers,
        coordinates=coords,
        demand_scenarios=demand_scenarios,
        vehicle_capacity=instance.truck_capacity_kg,
        n_vehicles=n_vehicles,
        alpha=instance.alpha,
    )


def chance_constrained_delivery(
    instance: FarmDeliveryInstance,
    seed: int = 42,
) -> FarmDeliverySolution:
    """Solve farm delivery using chance-constrained Clarke-Wright.

    Routes are constructed to ensure P(overflow) <= alpha for each route.

    Args:
        instance: FarmDeliveryInstance to solve.
        seed: Random seed.

    Returns:
        FarmDeliverySolution with chance-constrained routes.
    """
    _, cc_cw, _ = _get_svrp_modules()
    svrp = _build_svrp_instance(instance, seed=seed)
    sol = cc_cw.chance_constrained_savings(svrp)

    return FarmDeliverySolution(
        routes=sol.routes,
        total_distance=sol.total_distance,
        n_vehicles=sol.n_routes,
        method="CC-Clarke-Wright",
        expected_cost=sol.expected_total_cost,
        max_overflow_prob=sol.max_overflow_prob,
    )


def stochastic_sa_delivery(
    instance: FarmDeliveryInstance,
    max_iterations: int = 5000,
    seed: int = 42,
) -> FarmDeliverySolution:
    """Solve farm delivery using SA with recourse penalty.

    Args:
        instance: FarmDeliveryInstance to solve.
        max_iterations: SA iteration count.
        seed: Random seed.

    Returns:
        FarmDeliverySolution with SA-optimized routes.
    """
    _, _, sa_mod = _get_svrp_modules()
    svrp = _build_svrp_instance(instance, seed=seed)
    sol = sa_mod.simulated_annealing(svrp, max_iterations=max_iterations, seed=seed)

    return FarmDeliverySolution(
        routes=sol.routes,
        total_distance=sol.total_distance,
        n_vehicles=sol.n_routes,
        method="SA-Recourse",
        expected_cost=sol.expected_total_cost,
        max_overflow_prob=sol.max_overflow_prob,
    )


if __name__ == "__main__":
    inst = FarmDeliveryInstance.quebec_cooperative()
    print("=== Farm-to-Market Delivery (Stochastic VRP) ===\n")

    sol_cc = chance_constrained_delivery(inst)
    print(f"CC-Clarke-Wright: {sol_cc}")
    print(f"  Expected cost: {sol_cc.expected_cost:.1f}")
    print(f"  Max overflow: {sol_cc.max_overflow_prob:.3f}")

    sol_sa = stochastic_sa_delivery(inst)
    print(f"\nSA-Recourse: {sol_sa}")
    print(f"  Expected cost: {sol_sa.expected_cost:.1f}")
    print(f"  Max overflow: {sol_sa.max_overflow_prob:.3f}")

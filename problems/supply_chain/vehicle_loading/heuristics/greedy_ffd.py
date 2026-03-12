"""First-Fit Decreasing heuristic for Vehicle Loading.

Algorithm: Sort items by weight (decreasing). For each item, place it
in the first vehicle that has enough remaining weight AND volume capacity.
If no vehicle fits, open a new vehicle.

Complexity: O(n^2) worst case (n items, checking each vehicle).

References:
    Johnson, D. S. (1973). Near-optimal bin packing algorithms.
    Doctoral dissertation, MIT.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import numpy as np


def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_parent(
    "vl_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
VehicleLoadingInstance = _inst.VehicleLoadingInstance
VehicleLoadingSolution = _inst.VehicleLoadingSolution


def greedy_ffd(instance: VehicleLoadingInstance) -> VehicleLoadingSolution:
    """First-Fit Decreasing heuristic for vehicle loading.

    Sort items by weight (decreasing), then place each item in the
    first vehicle with sufficient remaining weight and volume.

    Args:
        instance: A VehicleLoadingInstance.

    Returns:
        A VehicleLoadingSolution.
    """
    # Sort items by weight, descending
    order = np.argsort(-instance.weights)

    vehicles: list[list[int]] = []
    remaining_weight: list[float] = []
    remaining_volume: list[float] = []

    for idx in order:
        w = instance.weights[idx]
        v = instance.volumes[idx]

        placed = False
        for vi in range(len(vehicles)):
            if remaining_weight[vi] >= w - 1e-9 and remaining_volume[vi] >= v - 1e-9:
                vehicles[vi].append(int(idx))
                remaining_weight[vi] -= w
                remaining_volume[vi] -= v
                placed = True
                break

        if not placed:
            vehicles.append([int(idx)])
            remaining_weight.append(instance.weight_capacity - w)
            remaining_volume.append(instance.volume_capacity - v)

    return VehicleLoadingSolution(
        vehicle_assignments=vehicles,
        n_vehicles=len(vehicles),
    )


def greedy_ffd_volume(instance: VehicleLoadingInstance) -> VehicleLoadingSolution:
    """First-Fit Decreasing by volume for vehicle loading.

    Same as FFD but sorts by volume instead of weight.

    Args:
        instance: A VehicleLoadingInstance.

    Returns:
        A VehicleLoadingSolution.
    """
    order = np.argsort(-instance.volumes)

    vehicles: list[list[int]] = []
    remaining_weight: list[float] = []
    remaining_volume: list[float] = []

    for idx in order:
        w = instance.weights[idx]
        v = instance.volumes[idx]

        placed = False
        for vi in range(len(vehicles)):
            if remaining_weight[vi] >= w - 1e-9 and remaining_volume[vi] >= v - 1e-9:
                vehicles[vi].append(int(idx))
                remaining_weight[vi] -= w
                remaining_volume[vi] -= v
                placed = True
                break

        if not placed:
            vehicles.append([int(idx)])
            remaining_weight.append(instance.weight_capacity - w)
            remaining_volume.append(instance.volume_capacity - v)

    return VehicleLoadingSolution(
        vehicle_assignments=vehicles,
        n_vehicles=len(vehicles),
    )


if __name__ == "__main__":
    inst = VehicleLoadingInstance.random(n_items=15)
    sol = greedy_ffd(inst)
    print(f"Instance: {inst.n_items} items, W_cap={inst.weight_capacity}, "
          f"V_cap={inst.volume_capacity}")
    print(f"Solution: {sol}")
    print(f"Valid: {inst.validate_loading(sol.vehicle_assignments)}")

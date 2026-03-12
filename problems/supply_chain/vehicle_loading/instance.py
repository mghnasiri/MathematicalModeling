"""Vehicle/Container Loading Problem — 1D weight + volume bin packing.

Problem: Pack items with weight and volume into vehicles with weight
and volume capacities, minimizing the number of vehicles used.

Complexity: NP-hard (generalizes 1D bin packing).

References:
    Christensen, H. I., Khan, A., Pokutta, S., & Tetali, P. (2017).
    Approximation and online algorithms for multidimensional bin packing:
    A survey. Computer Science Review, 24, 63-79.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleLoadingInstance:
    """Vehicle loading problem instance.

    Args:
        n_items: Number of items to load.
        weights: Weight of each item (length n_items).
        volumes: Volume of each item (length n_items).
        weight_capacity: Weight capacity of each vehicle.
        volume_capacity: Volume capacity of each vehicle.
    """
    n_items: int
    weights: np.ndarray
    volumes: np.ndarray
    weight_capacity: float
    volume_capacity: float

    @classmethod
    def random(cls, n_items: int = 15, seed: int = 42,
               weight_capacity: float = 100.0,
               volume_capacity: float = 100.0) -> VehicleLoadingInstance:
        """Generate a random vehicle loading instance.

        Args:
            n_items: Number of items.
            seed: Random seed.
            weight_capacity: Vehicle weight capacity.
            volume_capacity: Vehicle volume capacity.

        Returns:
            A random VehicleLoadingInstance.
        """
        rng = np.random.default_rng(seed)
        weights = rng.uniform(5, 40, size=n_items)
        volumes = rng.uniform(5, 40, size=n_items)
        return cls(
            n_items=n_items,
            weights=weights,
            volumes=volumes,
            weight_capacity=weight_capacity,
            volume_capacity=volume_capacity,
        )

    def validate_loading(self, vehicle_assignments: list[list[int]]) -> bool:
        """Check that a loading is feasible.

        Args:
            vehicle_assignments: List of vehicles, each a list of item indices.

        Returns:
            True if all capacity constraints are satisfied and all items loaded exactly once.
        """
        loaded = set()
        for vehicle_items in vehicle_assignments:
            total_w = sum(self.weights[i] for i in vehicle_items)
            total_v = sum(self.volumes[i] for i in vehicle_items)
            if total_w > self.weight_capacity + 1e-9:
                return False
            if total_v > self.volume_capacity + 1e-9:
                return False
            for i in vehicle_items:
                if i in loaded:
                    return False
                loaded.add(i)
        return loaded == set(range(self.n_items))


@dataclass
class VehicleLoadingSolution:
    """Solution to a vehicle loading problem.

    Args:
        vehicle_assignments: List of vehicles, each containing a list of item indices.
        n_vehicles: Number of vehicles used.
    """
    vehicle_assignments: list[list[int]]
    n_vehicles: int

    def __repr__(self) -> str:
        sizes = [len(v) for v in self.vehicle_assignments]
        return (f"VehicleLoadingSolution(vehicles={self.n_vehicles}, "
                f"items_per_vehicle={sizes})")

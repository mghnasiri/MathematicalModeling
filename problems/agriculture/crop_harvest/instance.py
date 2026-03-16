"""
Perishable Crop Harvest Planning Problem

Domain: Agriculture / perishable produce supply chain
Notation: NV_agri | stochastic demand, perishability | min E[cost]

A vegetable farm produces multiple perishable crops. Each day during peak
season, the farmer must decide how many kilograms of each crop to harvest
before demand from farmers markets, restaurants, and grocery stores is
revealed. Unharvested crop perishes in the field; excess harvested produce
is sold at salvage value to food processors or composted.

This is a direct application of the Newsvendor model to agricultural
harvest decisions, extended to multi-product with shared labor budget.

Complexity: O(S log S) per crop for critical fractile; O(n * S * B/step)
            for multi-product marginal allocation.

References:
    Minner, S. & Transchel, S. (2010). Periodic review inventory-control
    for perishable products under service-level constraints. OR Spectrum,
    32(4), 979-996. https://doi.org/10.1007/s00291-010-0196-1

    Ketzenberg, M. & Ferguson, M.E. (2008). Managing slow-moving
    perishables in the grocery industry. Production and Operations
    Management, 17(5), 513-521. https://doi.org/10.3401/poms.1080.0052
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class CropProfile:
    """Profile for a single perishable crop.

    Args:
        name: Crop name (e.g., "Tomatoes").
        unit_cost: Harvest + packing cost per kg ($/kg).
        selling_price: Revenue per kg at market ($/kg).
        salvage_value: Revenue per unsold kg ($/kg, food processor/compost).
        demand_mean: Expected daily demand in kg.
        demand_std: Standard deviation of daily demand in kg.
    """
    name: str
    unit_cost: float
    selling_price: float
    salvage_value: float
    demand_mean: float
    demand_std: float

    @property
    def overage_cost(self) -> float:
        """Cost per kg harvested but unsold: c - v."""
        return self.unit_cost - self.salvage_value

    @property
    def underage_cost(self) -> float:
        """Cost per kg of unmet demand: p - c."""
        return self.selling_price - self.unit_cost

    @property
    def critical_fractile(self) -> float:
        """Optimal service level: c_u / (c_u + c_o)."""
        cu = self.underage_cost
        co = self.overage_cost
        return cu / (cu + co)


@dataclass
class CropHarvestInstance:
    """Multi-crop perishable harvest planning instance.

    Args:
        crops: List of CropProfile for each crop.
        n_scenarios: Number of demand scenarios to generate.
        daily_labor_budget: Total daily labor/logistics budget ($).
        name: Instance name for identification.
    """
    crops: list[CropProfile]
    n_scenarios: int = 50
    daily_labor_budget: float = 2500.0
    name: str = "crop_harvest"

    @property
    def n_crops(self) -> int:
        return len(self.crops)

    def generate_scenarios(self, seed: int = 42) -> np.ndarray:
        """Generate demand scenarios for all crops.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (n_scenarios, n_crops) with demand values.
        """
        rng = np.random.default_rng(seed)
        scenarios = np.zeros((self.n_scenarios, self.n_crops))
        for i, crop in enumerate(self.crops):
            scenarios[:, i] = np.maximum(
                0.0,
                rng.normal(crop.demand_mean, crop.demand_std, self.n_scenarios),
            )
        return scenarios

    @classmethod
    def quebec_farm(cls, n_scenarios: int = 50) -> CropHarvestInstance:
        """Create the Quebec vegetable farm benchmark instance.

        8 perishable crops with realistic cost and demand parameters
        for a medium-sized Quebec vegetable farm during peak season.

        Args:
            n_scenarios: Number of demand scenarios.

        Returns:
            CropHarvestInstance with 8 crops.
        """
        crops = [
            CropProfile("Tomatoes", 1.50, 4.00, 0.50, 500.0, 100.0),
            CropProfile("Strawberries", 3.00, 8.00, 1.00, 200.0, 60.0),
            CropProfile("Lettuce", 0.80, 2.50, 0.20, 300.0, 80.0),
            CropProfile("Cucumbers", 0.60, 2.00, 0.15, 350.0, 90.0),
            CropProfile("Peppers", 1.80, 5.00, 0.60, 250.0, 70.0),
            CropProfile("Blueberries", 4.00, 10.00, 1.50, 150.0, 50.0),
            CropProfile("Zucchini", 0.50, 1.80, 0.10, 400.0, 120.0),
            CropProfile("Herbs", 2.50, 12.00, 0.80, 80.0, 25.0),
        ]
        return cls(
            crops=crops,
            n_scenarios=n_scenarios,
            daily_labor_budget=2500.0,
            name="quebec_vegetable_farm",
        )

    @classmethod
    def random(cls, n_crops: int = 5, n_scenarios: int = 50,
               seed: int = 42) -> CropHarvestInstance:
        """Generate a random crop harvest instance.

        Args:
            n_crops: Number of crops.
            n_scenarios: Number of demand scenarios.
            seed: Random seed.

        Returns:
            Random CropHarvestInstance.
        """
        rng = np.random.default_rng(seed)
        crop_names = [
            "Crop_A", "Crop_B", "Crop_C", "Crop_D", "Crop_E",
            "Crop_F", "Crop_G", "Crop_H", "Crop_I", "Crop_J",
        ]
        crops = []
        for i in range(n_crops):
            price = rng.uniform(2.0, 15.0)
            cost = rng.uniform(0.5, price * 0.6)
            salvage = rng.uniform(0.0, cost * 0.4)
            mean_d = rng.uniform(50.0, 600.0)
            std_d = mean_d * rng.uniform(0.15, 0.35)
            crops.append(CropProfile(
                name=crop_names[i % len(crop_names)],
                unit_cost=round(cost, 2),
                selling_price=round(price, 2),
                salvage_value=round(salvage, 2),
                demand_mean=round(mean_d, 1),
                demand_std=round(std_d, 1),
            ))
        budget = sum(c.unit_cost * c.demand_mean for c in crops) * 0.8
        return cls(
            crops=crops,
            n_scenarios=n_scenarios,
            daily_labor_budget=round(budget, 2),
            name=f"random_{n_crops}crops",
        )


@dataclass
class CropHarvestSolution:
    """Solution to the crop harvest planning problem.

    Args:
        harvest_quantities: Optimal harvest quantity per crop (kg).
        expected_profits: Expected profit per crop ($).
        service_levels: Service level (fill rate) per crop.
        total_expected_profit: Sum of expected profits.
        total_harvest_cost: Total harvest cost.
        budget_feasible: Whether solution fits within labor budget.
        method: Algorithm used.
    """
    harvest_quantities: list[float]
    expected_profits: list[float]
    service_levels: list[float]
    total_expected_profit: float
    total_harvest_cost: float
    budget_feasible: bool
    method: str

    def __repr__(self) -> str:
        return (f"CropHarvestSolution(method={self.method}, "
                f"profit=${self.total_expected_profit:.2f}, "
                f"cost=${self.total_harvest_cost:.2f}, "
                f"feasible={self.budget_feasible})")


if __name__ == "__main__":
    inst = CropHarvestInstance.quebec_farm()
    print(f"Quebec farm: {inst.n_crops} crops, budget=${inst.daily_labor_budget}")
    for crop in inst.crops:
        print(f"  {crop.name}: cost=${crop.unit_cost}, price=${crop.selling_price}, "
              f"CF={crop.critical_fractile:.3f}")
    scenarios = inst.generate_scenarios()
    print(f"Scenarios shape: {scenarios.shape}")

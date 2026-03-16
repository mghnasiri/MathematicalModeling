"""
Crop Rotation and Land Allocation Optimization Problem

Domain: Agricultural planning / Precision farming
Models: LP (revenue maximization) + Multi-Objective (revenue vs soil health)

A farm with multiple fields and candidate crops optimizes land allocation
subject to water, labor, diversity, and rotation constraints. Two models:

1. LP: Maximize total revenue with sensitivity analysis.
2. Multi-Objective: Pareto front trading off revenue vs nitrogen balance
   (soil health) via epsilon-constraint method.

Complexity: Polynomial (LP), O(n_points * LP) for Pareto front.

References:
    Bullock, D.G. (1992). Crop rotation. Critical Reviews in Plant
    Sciences, 11(4), 309-326.
    https://doi.org/10.1080/07352689209382349

    Dury, J. et al. (2012). Models to support cropping plan and crop
    rotation decisions. A review. Agronomy for Sustainable Development,
    32(2), 567-580. https://doi.org/10.1007/s13593-011-0037-x

    Detlefsen, N.K. & Jensen, A.L. (2007). Modelling optimal crop
    sequences using network flows. Agricultural Systems, 94(2), 566-572.
    https://doi.org/10.1016/j.agsy.2007.02.002
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class FieldProfile:
    """A farm field with its characteristics.

    Args:
        name: Field name.
        hectares: Field area in hectares.
        soil_quality: One of "high", "medium", "low".
    """
    name: str
    hectares: float
    soil_quality: str


@dataclass
class CropRotationInstance:
    """Crop rotation and land allocation instance.

    Args:
        fields: List of field profiles.
        crops: List of crop names.
        revenue_per_ton: Revenue per ton for each crop.
        yield_by_soil: Yield (tons/ha) by crop and soil quality.
        water_per_ha: Water requirement (m^3/ha) per crop.
        labor_per_ha: Labor requirement (hours/ha) per crop.
        nitrogen_effect: Nitrogen balance (kg N/ha) per crop.
        water_budget: Total water budget (m^3).
        labor_budget: Total labor budget (hours).
        max_crop_fraction: Maximum fraction of total area for any crop.
        rotation_forbidden: List of (field_idx, crop_name) forbidden pairs.
        name: Instance name.
    """
    fields: list[FieldProfile]
    crops: list[str]
    revenue_per_ton: dict[str, float]
    yield_by_soil: dict[str, dict[str, float]]
    water_per_ha: dict[str, float]
    labor_per_ha: dict[str, float]
    nitrogen_effect: dict[str, float]
    water_budget: float
    labor_budget: float
    max_crop_fraction: float = 0.40
    rotation_forbidden: list[tuple[int, str]] = field(default_factory=list)
    name: str = "crop_rotation"

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def n_crops(self) -> int:
        return len(self.crops)

    @property
    def total_hectares(self) -> float:
        return sum(f.hectares for f in self.fields)

    def get_yield(self, crop: str, field_idx: int) -> float:
        """Get expected yield (tons/ha) for crop on field."""
        soil = self.fields[field_idx].soil_quality
        return self.yield_by_soil[crop][soil]

    def get_revenue_per_ha(self, crop: str, field_idx: int) -> float:
        """Get expected revenue ($/ha) for crop on field."""
        return self.get_yield(crop, field_idx) * self.revenue_per_ton[crop]

    @classmethod
    def standard_farm(cls) -> CropRotationInstance:
        """Create the standard 6-field 5-crop benchmark instance.

        Returns:
            CropRotationInstance with realistic Quebec farm parameters.
        """
        fields = [
            FieldProfile("North Ridge", 35, "high"),
            FieldProfile("East Valley", 25, "medium"),
            FieldProfile("South Flat", 40, "high"),
            FieldProfile("West Hill", 15, "low"),
            FieldProfile("Central Plot", 20, "medium"),
            FieldProfile("Creek Bottom", 10, "high"),
        ]
        crops = ["corn", "wheat", "soybeans", "vegetables", "hay"]
        revenue_per_ton = {
            "corn": 180, "wheat": 220, "soybeans": 400,
            "vegetables": 600, "hay": 120,
        }
        yield_by_soil = {
            "corn": {"high": 10.5, "medium": 8.5, "low": 6.0},
            "wheat": {"high": 5.0, "medium": 4.0, "low": 3.0},
            "soybeans": {"high": 3.5, "medium": 3.0, "low": 2.2},
            "vegetables": {"high": 18.0, "medium": 14.0, "low": 9.0},
            "hay": {"high": 8.0, "medium": 7.0, "low": 5.5},
        }
        water_per_ha = {
            "corn": 5500, "wheat": 3000, "soybeans": 4000,
            "vegetables": 7000, "hay": 2500,
        }
        labor_per_ha = {
            "corn": 35, "wheat": 25, "soybeans": 20,
            "vegetables": 120, "hay": 15,
        }
        nitrogen_effect = {
            "corn": -80, "wheat": -30, "soybeans": 60,
            "vegetables": -50, "hay": 10,
        }
        rotation_forbidden = [
            (0, "corn"),      # North Ridge grew corn last year
            (2, "wheat"),     # South Flat grew wheat last year
            (4, "soybeans"),  # Central Plot grew soybeans last year
        ]
        return cls(
            fields=fields, crops=crops,
            revenue_per_ton=revenue_per_ton,
            yield_by_soil=yield_by_soil,
            water_per_ha=water_per_ha,
            labor_per_ha=labor_per_ha,
            nitrogen_effect=nitrogen_effect,
            water_budget=500_000,
            labor_budget=8_000,
            max_crop_fraction=0.40,
            rotation_forbidden=rotation_forbidden,
            name="standard_farm",
        )

    @classmethod
    def random(cls, n_fields: int = 4, n_crops: int = 3,
               seed: int = 42) -> CropRotationInstance:
        """Generate a random crop rotation instance.

        Args:
            n_fields: Number of fields.
            n_crops: Number of candidate crops.
            seed: Random seed.

        Returns:
            Random CropRotationInstance.
        """
        rng = np.random.default_rng(seed)
        qualities = ["high", "medium", "low"]
        fields = [
            FieldProfile(f"Field_{i}", float(rng.integers(10, 50)),
                         rng.choice(qualities))
            for i in range(n_fields)
        ]
        crop_names = ["crop_A", "crop_B", "crop_C", "crop_D", "crop_E"][:n_crops]
        revenue = {c: float(rng.integers(100, 700)) for c in crop_names}
        yields = {}
        for c in crop_names:
            base = rng.uniform(3, 15)
            yields[c] = {
                "high": round(base, 1),
                "medium": round(base * 0.75, 1),
                "low": round(base * 0.5, 1),
            }
        water = {c: float(rng.integers(2000, 8000)) for c in crop_names}
        labor = {c: float(rng.integers(15, 130)) for c in crop_names}
        nitrogen = {c: float(rng.integers(-100, 80)) for c in crop_names}
        total_ha = sum(f.hectares for f in fields)
        # Relax diversity to ensure feasibility with few crops
        max_frac = min(0.90, 1.0 / n_crops + 0.3)
        return cls(
            fields=fields, crops=crop_names,
            revenue_per_ton=revenue, yield_by_soil=yields,
            water_per_ha=water, labor_per_ha=labor,
            nitrogen_effect=nitrogen,
            water_budget=total_ha * 8000,
            labor_budget=total_ha * 130,
            max_crop_fraction=max_frac,
            name=f"random_{n_fields}f_{n_crops}c",
        )


@dataclass
class CropAllocation:
    """Allocation of crops to fields.

    Args:
        field_crops: Dict mapping field index to assigned crop name.
        fractions: Dict mapping field index to allocation fraction (0-1).
    """
    field_crops: dict[int, str]
    fractions: dict[int, float]


@dataclass
class CropRotationSolution:
    """Solution to the crop rotation problem.

    Args:
        success: Whether optimization succeeded.
        total_revenue: Total expected revenue ($).
        allocation: Crop allocation per field.
        total_water: Total water used (m^3).
        total_labor: Total labor used (hours).
        total_nitrogen: Net nitrogen balance (kg N).
        method: Algorithm used.
    """
    success: bool
    total_revenue: float
    allocation: CropAllocation
    total_water: float
    total_labor: float
    total_nitrogen: float
    method: str

    def __repr__(self) -> str:
        return (f"CropRotationSolution(method={self.method}, "
                f"revenue=${self.total_revenue:,.0f}, "
                f"nitrogen={self.total_nitrogen:+,.0f} kg N)")


@dataclass
class ParetoFrontSolution:
    """Pareto front for revenue vs soil health trade-off.

    Args:
        points: List of (revenue, nitrogen) tuples.
        allocations: List of CropAllocation for each Pareto point.
        n_points: Number of Pareto-efficient points.
        nitrogen_range: Tuple of (min, max) achievable nitrogen.
    """
    points: list[tuple[float, float]]
    allocations: list[CropAllocation]
    n_points: int
    nitrogen_range: tuple[float, float]

    def __repr__(self) -> str:
        return (f"ParetoFrontSolution(n_points={self.n_points}, "
                f"N_range=[{self.nitrogen_range[0]:+,.0f}, "
                f"{self.nitrogen_range[1]:+,.0f}] kg N)")


if __name__ == "__main__":
    inst = CropRotationInstance.standard_farm()
    print(f"Standard farm: {inst.n_fields} fields ({inst.total_hectares} ha), "
          f"{inst.n_crops} crops")
    for f in inst.fields:
        print(f"  {f.name}: {f.hectares} ha, soil={f.soil_quality}")

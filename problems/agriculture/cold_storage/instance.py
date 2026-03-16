"""
Agriculture Cold Storage & Packaging Optimization Problem

Domain: Agriculture / Produce packing house operations
Models: Bin Packing (cold storage) + Cutting Stock (packaging film)

A produce packing house solves two logistics problems:
1. Cold Storage: Pack produce lots into cold storage units (minimize units).
2. Packaging Film: Cut film rolls into wrapping sheets (minimize rolls).

Complexity: NP-hard (strongly) for both problems.

References:
    FAO (2019). The State of Food and Agriculture: Moving Forward on
    Food Loss and Waste Reduction. Rome: Food and Agriculture
    Organization of the United Nations.
    https://doi.org/10.4060/ca6030en

    Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation
    algorithms for bin packing: A survey. In: Hochbaum, D.S. (ed)
    Approximation Algorithms for NP-hard Problems. PWS Publishing, 46-93.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class ProduceLot:
    """A produce lot to be stored.

    Args:
        lot_id: Unique identifier.
        produce_type: Type of produce.
        weight_kg: Lot weight in kg.
    """
    lot_id: str
    produce_type: str
    weight_kg: float


@dataclass
class SheetType:
    """A packaging sheet type with demand.

    Args:
        name: Sheet size name.
        label: Human-readable label.
        length_cm: Sheet length in cm.
        demand: Number of sheets needed.
    """
    name: str
    label: str
    length_cm: float
    demand: int


@dataclass
class ColdStorageInstance:
    """Cold storage bin packing instance.

    Args:
        lots: List of produce lots.
        storage_capacity_kg: Weight capacity per cold storage unit.
        name: Instance name.
    """
    lots: list[ProduceLot]
    storage_capacity_kg: float = 1000.0
    name: str = "cold_storage"

    @property
    def n_lots(self) -> int:
        return len(self.lots)

    @property
    def total_weight(self) -> float:
        return sum(lot.weight_kg for lot in self.lots)

    def get_weights(self) -> np.ndarray:
        return np.array([lot.weight_kg for lot in self.lots], dtype=float)

    @classmethod
    def packing_house(cls, n_lots: int = 20, seed: int = 42) -> ColdStorageInstance:
        """Create a packing house benchmark instance.

        Generates produce lots from 4 types with realistic weight ranges.

        Args:
            n_lots: Number of produce lots.
            seed: Random seed.

        Returns:
            ColdStorageInstance with mixed produce.
        """
        rng = np.random.default_rng(seed)
        produce_types = {
            "tomato": {"weight_range": (150, 300), "label": "Tomato pallet", "prob": 0.30},
            "berry": {"weight_range": (80, 150), "label": "Berry crate", "prob": 0.20},
            "lettuce": {"weight_range": (100, 200), "label": "Lettuce bin", "prob": 0.25},
            "mixed_veg": {"weight_range": (50, 120), "label": "Mixed veg box", "prob": 0.25},
        }
        type_names = list(produce_types.keys())
        probs = [produce_types[t]["prob"] for t in type_names]

        lots = []
        for i in range(n_lots):
            ptype = rng.choice(type_names, p=probs)
            info = produce_types[ptype]
            weight = int(rng.integers(info["weight_range"][0],
                                      info["weight_range"][1] + 1))
            lots.append(ProduceLot(
                lot_id=f"lot-{i:03d}",
                produce_type=ptype,
                weight_kg=float(weight),
            ))
        return cls(lots=lots, storage_capacity_kg=1000.0, name="packing_house")


@dataclass
class PackagingInstance:
    """Packaging film cutting stock instance.

    Args:
        sheet_types: List of sheet types with demands.
        roll_length_cm: Length of stock film rolls.
        name: Instance name.
    """
    sheet_types: list[SheetType]
    roll_length_cm: float = 200.0
    name: str = "packaging_film"

    @property
    def n_types(self) -> int:
        return len(self.sheet_types)

    def get_lengths(self) -> np.ndarray:
        return np.array([s.length_cm for s in self.sheet_types], dtype=float)

    def get_demands(self) -> np.ndarray:
        return np.array([s.demand for s in self.sheet_types], dtype=int)

    @classmethod
    def standard(cls) -> PackagingInstance:
        """Create the standard packaging film benchmark instance.

        4 sheet sizes for different produce box types.

        Returns:
            PackagingInstance with 4 sheet types.
        """
        sheets = [
            SheetType("large", "Large box wrap", 60, 25),
            SheetType("medium", "Medium box wrap", 45, 40),
            SheetType("small", "Small box wrap", 30, 50),
            SheetType("mini", "Mini tray wrap", 20, 30),
        ]
        return cls(sheet_types=sheets, roll_length_cm=200.0, name="standard_film")


@dataclass
class ColdStorageSolution:
    """Solution to the cold storage packing problem.

    Args:
        n_units: Number of cold storage units used.
        bins: List of item index lists per bin.
        method: Algorithm used.
    """
    n_units: int
    bins: list[list[int]]
    method: str

    def __repr__(self) -> str:
        return (f"ColdStorageSolution(method={self.method}, "
                f"units={self.n_units})")


@dataclass
class PackagingSolution:
    """Solution to the packaging film cutting problem.

    Args:
        n_rolls: Number of film rolls used.
        waste_cm: Total waste in cm.
        waste_pct: Waste as percentage of total stock.
        patterns: List of (pattern, frequency) tuples.
        method: Algorithm used.
    """
    n_rolls: int
    waste_cm: float
    waste_pct: float
    patterns: list[tuple[np.ndarray, int]]
    method: str

    def __repr__(self) -> str:
        return (f"PackagingSolution(method={self.method}, "
                f"rolls={self.n_rolls}, waste={self.waste_pct:.1f}%)")


if __name__ == "__main__":
    cs = ColdStorageInstance.packing_house()
    print(f"Cold storage: {cs.n_lots} lots, {cs.total_weight:.0f} kg total")

    pkg = PackagingInstance.standard()
    print(f"Packaging: {pkg.n_types} sheet types, roll={pkg.roll_length_cm} cm")

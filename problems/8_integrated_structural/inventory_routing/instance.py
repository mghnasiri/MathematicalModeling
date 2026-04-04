"""
Inventory-Routing Problem (IRP)

Jointly optimizes inventory replenishment and vehicle routing over a
multi-period planning horizon. A supplier at a depot uses a fleet of
capacitated vehicles to deliver product to customers with deterministic
demand rates, aiming to minimize total routing cost plus inventory
holding cost while avoiding stockouts.

Notation: IRP | det. demand, capacity | min routing + holding

Complexity: NP-hard (generalizes both VRP and lot-sizing).

References:
    - Federgruen, A. & Zipkin, P. (1984). A combined vehicle routing
      and inventory allocation problem. Oper. Res., 32(5), 1019-1037.
      https://doi.org/10.1287/opre.32.5.1019
    - Campbell, A., Clarke, L., Kleywegt, A. & Savelsbergh, M. (1998).
      The inventory routing problem. In: Fleet Management and Logistics,
      pp. 95-113. Springer.
      https://doi.org/10.1007/978-1-4615-5755-5_4
    - Coelho, L.C., Cordeau, J.-F. & Laporte, G. (2014). Thirty years
      of inventory routing. Transp. Sci., 48(1), 1-19.
      https://doi.org/10.1287/trsc.2013.0472
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class IRPInstance:
    """Inventory-Routing Problem instance.

    A depot (node 0) supplies n customers over T periods using K vehicles
    of capacity Q. Each customer has a deterministic demand rate, holding
    cost, storage capacity, and initial inventory.

    Args:
        n_customers: Number of customers (not counting depot).
        T: Number of periods in the planning horizon.
        demands: (n,) array of per-period demand rates for each customer.
        holding_costs: (n,) array of per-unit per-period holding costs.
        storage_capacities: (n,) array of maximum inventory at each customer.
        initial_inventory: (n,) array of starting inventory levels.
        coordinates: (n+1, 2) array of node coordinates; depot at index 0.
        vehicle_capacity: Capacity Q of each vehicle.
        n_vehicles: Number of vehicles K available per period.
        name: Optional instance name.
    """
    n_customers: int
    T: int
    demands: np.ndarray
    holding_costs: np.ndarray
    storage_capacities: np.ndarray
    initial_inventory: np.ndarray
    coordinates: np.ndarray
    vehicle_capacity: float
    n_vehicles: int
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.holding_costs = np.asarray(self.holding_costs, dtype=float)
        self.storage_capacities = np.asarray(self.storage_capacities, dtype=float)
        self.initial_inventory = np.asarray(self.initial_inventory, dtype=float)
        self.coordinates = np.asarray(self.coordinates, dtype=float)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j.

        Args:
            i: First node index (0 = depot).
            j: Second node index (0 = depot).

        Returns:
            Euclidean distance.
        """
        diff = self.coordinates[i] - self.coordinates[j]
        return float(np.sqrt(np.dot(diff, diff)))

    def distance_matrix(self) -> np.ndarray:
        """Full (n+1) x (n+1) Euclidean distance matrix.

        Returns:
            Symmetric distance matrix including depot (index 0).
        """
        n = len(self.coordinates)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(i, j)
                dm[i, j] = d
                dm[j, i] = d
        return dm

    def route_distance(self, route: list[int]) -> float:
        """Total distance of a route: depot -> customers -> depot.

        Args:
            route: List of customer indices (1-indexed, no depot).

        Returns:
            Total Euclidean distance of the route.
        """
        if not route:
            return 0.0
        total = self.distance(0, route[0])
        for i in range(len(route) - 1):
            total += self.distance(route[i], route[i + 1])
        total += self.distance(route[-1], 0)
        return total

    @classmethod
    def small_instance(cls) -> IRPInstance:
        """Built-in small example: 5 customers, 3 periods.

        Layout: depot at center, customers around it.
        Moderate demands so that not every customer needs delivery every period.

        Returns:
            A small IRPInstance for testing and demonstration.
        """
        return cls(
            n_customers=5,
            T=3,
            demands=np.array([4.0, 3.0, 5.0, 2.0, 6.0]),
            holding_costs=np.array([1.0, 1.5, 0.8, 2.0, 1.2]),
            storage_capacities=np.array([15.0, 12.0, 18.0, 10.0, 20.0]),
            initial_inventory=np.array([8.0, 6.0, 10.0, 5.0, 12.0]),
            coordinates=np.array([
                [50.0, 50.0],   # depot
                [20.0, 80.0],   # customer 1
                [80.0, 80.0],   # customer 2
                [20.0, 20.0],   # customer 3
                [80.0, 20.0],   # customer 4
                [50.0, 90.0],   # customer 5
            ]),
            vehicle_capacity=30.0,
            n_vehicles=2,
            name="small_5_3",
        )

    @classmethod
    def random(cls, n_customers: int = 10, T: int = 5,
               n_vehicles: int = 3, seed: int = 42) -> IRPInstance:
        """Generate a random IRP instance.

        Args:
            n_customers: Number of customers.
            T: Number of periods.
            n_vehicles: Number of vehicles.
            seed: Random seed for reproducibility.

        Returns:
            A random IRPInstance.
        """
        rng = np.random.default_rng(seed)

        coordinates = rng.uniform(0, 100, (n_customers + 1, 2))
        demands = rng.uniform(2, 10, n_customers)
        holding_costs = rng.uniform(0.5, 3.0, n_customers)
        storage_capacities = demands * rng.uniform(3, 6, n_customers)
        initial_inventory = rng.uniform(0.3, 0.8, n_customers) * storage_capacities
        vehicle_capacity = float(demands.sum() * 0.6)

        return cls(
            n_customers=n_customers,
            T=T,
            demands=demands,
            holding_costs=holding_costs,
            storage_capacities=storage_capacities,
            initial_inventory=initial_inventory,
            coordinates=coordinates,
            vehicle_capacity=vehicle_capacity,
            n_vehicles=n_vehicles,
            name=f"random_{n_customers}_{T}",
        )


@dataclass
class IRPSolution:
    """Solution to the Inventory-Routing Problem.

    For each period t in [0, T-1], the solution specifies a set of routes
    (each route is a list of customer indices, 1-indexed) and delivery
    quantities for each customer visited.

    Args:
        routes_per_period: List of length T. Each element is a list of routes
            for that period. Each route is a list of customer indices (1-indexed).
        deliveries_per_period: List of length T. Each element is a dict mapping
            customer index (1-indexed) -> delivery quantity for that period.
        routing_cost: Total routing distance across all periods.
        holding_cost: Total holding cost across all periods and customers.
        total_cost: routing_cost + holding_cost.
        inventory_levels: (T+1, n) array of inventory at end of each period.
            Row 0 is initial inventory, row t is inventory after period t.
    """
    routes_per_period: list[list[list[int]]]
    deliveries_per_period: list[dict[int, float]]
    routing_cost: float
    holding_cost: float
    total_cost: float
    inventory_levels: np.ndarray | None = None

    def __repr__(self) -> str:
        T = len(self.routes_per_period)
        total_routes = sum(len(rr) for rr in self.routes_per_period)
        return (
            f"IRPSolution(periods={T}, routes={total_routes}, "
            f"routing={self.routing_cost:.1f}, "
            f"holding={self.holding_cost:.1f}, "
            f"total={self.total_cost:.1f})"
        )


def compute_cost(
    instance: IRPInstance,
    routes_per_period: list[list[list[int]]],
    deliveries_per_period: list[dict[int, float]],
) -> IRPSolution:
    """Compute total cost (routing + holding) for a given IRP solution.

    Simulates inventory dynamics over the planning horizon, computing
    routing distances per period and end-of-period holding costs.

    Args:
        instance: IRPInstance.
        routes_per_period: Routes for each period.
        deliveries_per_period: Delivery quantities for each period.

    Returns:
        IRPSolution with costs and inventory levels populated.
        Stockouts are penalized with a high cost rather than raising errors.
    """
    n = instance.n_customers
    T = instance.T

    # Track inventory: row 0 = initial, row t = end of period t
    inventory = np.zeros((T + 1, n))
    inventory[0] = instance.initial_inventory.copy()

    total_routing = 0.0
    total_holding = 0.0
    stockout_penalty = 0.0

    for t in range(T):
        # Start-of-period inventory
        inv_start = inventory[t].copy()

        # Add deliveries
        deliveries = deliveries_per_period[t] if t < len(deliveries_per_period) else {}
        for cust, qty in deliveries.items():
            inv_start[cust - 1] += qty  # customers are 1-indexed

        # Subtract demand
        inv_end = inv_start - instance.demands

        # Penalize stockouts instead of raising error (soft constraint)
        if np.any(inv_end < -1e-9):
            stockout_qty = float(np.sum(np.maximum(0.0, -inv_end)))
            stockout_penalty += stockout_qty * 1000.0  # high penalty per unit

        # Clip small numerical errors
        inv_end = np.maximum(inv_end, 0.0)
        inventory[t + 1] = inv_end

        # Holding cost: based on end-of-period inventory
        total_holding += float(np.dot(inv_end, instance.holding_costs))

        # Routing cost for this period
        period_routes = routes_per_period[t] if t < len(routes_per_period) else []
        for route in period_routes:
            total_routing += instance.route_distance(route)

    total_cost = total_routing + total_holding + stockout_penalty

    return IRPSolution(
        routes_per_period=routes_per_period,
        deliveries_per_period=deliveries_per_period,
        routing_cost=total_routing,
        holding_cost=total_holding,
        total_cost=total_cost,
        inventory_levels=inventory,
    )


# ── Benchmark instances ──────────────────────────────────────────────────────


def small_5_3() -> IRPInstance:
    """Small benchmark: 5 customers, 3 periods.

    Returns:
        IRPInstance with 5 customers and 3-period horizon.
    """
    return IRPInstance.small_instance()


def medium_10_5() -> IRPInstance:
    """Medium benchmark: 10 customers, 5 periods.

    Returns:
        IRPInstance with 10 customers and 5-period horizon.
    """
    return IRPInstance.random(n_customers=10, T=5, n_vehicles=3, seed=100)


if __name__ == "__main__":
    inst = IRPInstance.small_instance()
    print(f"Instance: {inst.name}")
    print(f"  Customers: {inst.n_customers}, Periods: {inst.T}")
    print(f"  Demands: {inst.demands}")
    print(f"  Holding costs: {inst.holding_costs}")
    print(f"  Storage caps: {inst.storage_capacities}")
    print(f"  Initial inv: {inst.initial_inventory}")
    print(f"  Vehicle cap: {inst.vehicle_capacity}, Vehicles: {inst.n_vehicles}")
    print(f"  Distance matrix shape: {inst.distance_matrix().shape}")
    print()

    inst2 = IRPInstance.random(n_customers=8, T=4, seed=77)
    print(f"Random instance: {inst2.name}")
    print(f"  Customers: {inst2.n_customers}, Periods: {inst2.T}")
    print(f"  Demands: {inst2.demands.round(2)}")

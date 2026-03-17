"""
Economic Order Quantity (EOQ) — Instance and Solution definitions.

The classic EOQ model determines the optimal order quantity that minimizes
the total inventory cost, consisting of ordering costs and holding costs.
Extensions include EOQ with backorders and EOQ with quantity discounts.

Complexity: O(1) for basic EOQ, O(B log B) for quantity discounts with
B breakpoints.

References:
    Harris, F.W. (1913). How many parts to make at once. Factory, The
    Magazine of Management, 10(2), 135-136, 152.

    Wilson, R.H. (1934). A scientific routine for stock control.
    Harvard Business Review, 13(1), 116-128.

    Zipkin, P.H. (2000). Foundations of Inventory Management.
    McGraw-Hill, New York.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EOQInstance:
    """Economic Order Quantity problem instance.

    Attributes:
        demand_rate: Annual demand rate D (units per time period).
        ordering_cost: Fixed cost per order K.
        holding_cost: Holding cost per unit per time period h.
        backorder_cost: Backorder cost per unit per time period b (0 means no backorders).
        discount_breaks: Quantity discount breakpoints, shape (B,).
            Each entry is the minimum quantity for that price tier.
        discount_prices: Unit price for each discount tier, shape (B,).
        name: Optional instance name.
    """

    demand_rate: float
    ordering_cost: float
    holding_cost: float
    backorder_cost: float = 0.0
    discount_breaks: np.ndarray | None = None
    discount_prices: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if self.ordering_cost < 0:
            raise ValueError("ordering_cost must be non-negative")
        if self.holding_cost <= 0:
            raise ValueError("holding_cost must be positive")
        if self.backorder_cost < 0:
            raise ValueError("backorder_cost must be non-negative")

        if self.discount_breaks is not None:
            self.discount_breaks = np.asarray(self.discount_breaks, dtype=float)
            self.discount_prices = np.asarray(self.discount_prices, dtype=float)
            if self.discount_breaks.shape != self.discount_prices.shape:
                raise ValueError(
                    "discount_breaks and discount_prices must have same shape"
                )

    @classmethod
    def random(
        cls,
        demand_range: tuple[float, float] = (100.0, 10000.0),
        ordering_cost_range: tuple[float, float] = (10.0, 500.0),
        holding_cost_range: tuple[float, float] = (0.5, 10.0),
        seed: int | None = None,
    ) -> EOQInstance:
        """Generate a random EOQ instance.

        Args:
            demand_range: Range for annual demand rate.
            ordering_cost_range: Range for ordering cost.
            holding_cost_range: Range for holding cost per unit per period.
            seed: Random seed for reproducibility.

        Returns:
            A random EOQInstance.
        """
        rng = np.random.default_rng(seed)
        demand = rng.uniform(*demand_range)
        ordering = rng.uniform(*ordering_cost_range)
        holding = rng.uniform(*holding_cost_range)

        return cls(
            demand_rate=round(demand, 2),
            ordering_cost=round(ordering, 2),
            holding_cost=round(holding, 2),
            name="random_eoq",
        )

    def total_cost(self, Q: float) -> float:
        """Compute total annual cost for order quantity Q (without purchase cost).

        Total cost = (D/Q)*K + (Q/2)*h for basic EOQ.

        Args:
            Q: Order quantity.

        Returns:
            Total annual cost.
        """
        if Q <= 0:
            return float("inf")
        ordering = (self.demand_rate / Q) * self.ordering_cost
        holding = (Q / 2.0) * self.holding_cost
        return ordering + holding

    def total_cost_with_backorders(self, Q: float, S: float) -> float:
        """Compute total cost with backorders.

        Args:
            Q: Order quantity.
            S: Maximum inventory level (Q - max backorder).

        Returns:
            Total annual cost with backorders.
        """
        if Q <= 0:
            return float("inf")
        b = self.backorder_cost
        h = self.holding_cost
        ordering = (self.demand_rate / Q) * self.ordering_cost
        holding = (S ** 2 / (2.0 * Q)) * h
        backorder = ((Q - S) ** 2 / (2.0 * Q)) * b
        return ordering + holding + backorder


@dataclass
class EOQSolution:
    """Solution to an EOQ instance.

    Attributes:
        order_quantity: Optimal order quantity Q*.
        total_cost: Total annual cost at Q*.
        reorder_point: Maximum inventory level (equals Q for basic EOQ).
        num_orders: Number of orders per period (D/Q).
        cycle_time: Time between orders (Q/D).
    """

    order_quantity: float
    total_cost: float
    reorder_point: float = 0.0
    num_orders: float = 0.0
    cycle_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"EOQSolution(Q*={self.order_quantity:.2f}, "
            f"cost={self.total_cost:.2f}, "
            f"orders/period={self.num_orders:.2f}, "
            f"cycle_time={self.cycle_time:.4f})"
        )


# ── Benchmark instances ──────────────────────────────────────────────────────


def textbook_eoq() -> EOQInstance:
    """Classic textbook EOQ instance.

    D=1000, K=50, h=2 => Q*=sqrt(2*1000*50/2)=sqrt(50000)=223.61
    """
    return EOQInstance(
        demand_rate=1000.0,
        ordering_cost=50.0,
        holding_cost=2.0,
        name="textbook",
    )


def backorder_eoq() -> EOQInstance:
    """EOQ instance with backorders allowed.

    D=1200, K=100, h=5, b=25
    """
    return EOQInstance(
        demand_rate=1200.0,
        ordering_cost=100.0,
        holding_cost=5.0,
        backorder_cost=25.0,
        name="backorder",
    )


def discount_eoq() -> EOQInstance:
    """EOQ instance with quantity discounts.

    D=10000, K=49, h=20% of price.
    Price tiers: [0, 300, 500] => [$5.00, $4.50, $3.90]
    """
    return EOQInstance(
        demand_rate=10000.0,
        ordering_cost=49.0,
        holding_cost=1.0,  # will be adjusted by price
        discount_breaks=np.array([0.0, 300.0, 500.0]),
        discount_prices=np.array([5.0, 4.50, 3.90]),
        name="discount",
    )


if __name__ == "__main__":
    inst = textbook_eoq()
    print(f"{inst.name}: D={inst.demand_rate}, K={inst.ordering_cost}, h={inst.holding_cost}")
    Q_star = np.sqrt(2 * inst.demand_rate * inst.ordering_cost / inst.holding_cost)
    print(f"  EOQ = {Q_star:.2f}")
    print(f"  Total cost = {inst.total_cost(Q_star):.2f}")

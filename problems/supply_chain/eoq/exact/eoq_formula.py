"""
Economic Order Quantity — Exact closed-form solutions.

Problem: EOQ (Economic Order Quantity)
Complexity: O(1) for basic and backorder models, O(B log B) for discounts.

Implements three variants:
1. Classic EOQ: Q* = sqrt(2*D*K / h)
2. EOQ with backorders: Q* = sqrt(2*D*K/h * (h+b)/b)
3. EOQ with quantity discounts: evaluate EOQ per price tier

References:
    Harris, F.W. (1913). How many parts to make at once.
    Factory, The Magazine of Management, 10(2), 135-136, 152.

    Hadley, G. & Whitin, T.M. (1963). Analysis of Inventory Systems.
    Prentice-Hall, Englewood Cliffs, NJ.

    Zipkin, P.H. (2000). Foundations of Inventory Management.
    McGraw-Hill, New York.
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("eoq_instance", os.path.join(_parent_dir, "instance.py"))
EOQInstance = _inst.EOQInstance
EOQSolution = _inst.EOQSolution


def classic_eoq(instance: EOQInstance) -> EOQSolution:
    """Compute the classic Economic Order Quantity.

    Q* = sqrt(2 * D * K / h)

    Total cost at Q* = sqrt(2 * D * K * h)

    Args:
        instance: EOQInstance with demand_rate, ordering_cost, holding_cost.

    Returns:
        EOQSolution with optimal order quantity and total cost.
    """
    D = instance.demand_rate
    K = instance.ordering_cost
    h = instance.holding_cost

    Q_star = np.sqrt(2.0 * D * K / h)
    total_cost = instance.total_cost(Q_star)
    num_orders = D / Q_star
    cycle_time = Q_star / D

    return EOQSolution(
        order_quantity=float(Q_star),
        total_cost=float(total_cost),
        reorder_point=float(Q_star),
        num_orders=float(num_orders),
        cycle_time=float(cycle_time),
    )


def eoq_with_backorders(instance: EOQInstance) -> EOQSolution:
    """Compute EOQ with planned backorders.

    Q* = sqrt(2*D*K/h) * sqrt((h+b)/b)
    S* = Q* * b/(h+b)   (maximum inventory level)

    Args:
        instance: EOQInstance with backorder_cost > 0.

    Returns:
        EOQSolution with optimal order quantity and total cost.

    Raises:
        ValueError: If backorder_cost is zero.
    """
    D = instance.demand_rate
    K = instance.ordering_cost
    h = instance.holding_cost
    b = instance.backorder_cost

    if b <= 0:
        raise ValueError("backorder_cost must be positive for backorder model")

    Q_star = np.sqrt(2.0 * D * K / h) * np.sqrt((h + b) / b)
    S_star = Q_star * b / (h + b)

    total_cost = instance.total_cost_with_backorders(Q_star, S_star)
    num_orders = D / Q_star
    cycle_time = Q_star / D

    return EOQSolution(
        order_quantity=float(Q_star),
        total_cost=float(total_cost),
        reorder_point=float(S_star),
        num_orders=float(num_orders),
        cycle_time=float(cycle_time),
    )


def eoq_with_discounts(
    instance: EOQInstance,
    holding_pct: float = 0.2,
) -> EOQSolution:
    """Compute EOQ with all-units quantity discounts.

    For each price tier, compute the EOQ using h_i = holding_pct * price_i.
    If the EOQ falls within the valid range for that tier, it is a candidate.
    Otherwise, use the tier's minimum quantity as the candidate.
    Select the candidate with the lowest total cost (including purchase cost).

    Args:
        instance: EOQInstance with discount_breaks and discount_prices.
        holding_pct: Holding cost as a fraction of unit price.

    Returns:
        EOQSolution with optimal order quantity and total cost (including purchase).

    Raises:
        ValueError: If discount data is not provided.
    """
    if instance.discount_breaks is None or instance.discount_prices is None:
        raise ValueError("discount_breaks and discount_prices required")

    D = instance.demand_rate
    K = instance.ordering_cost
    breaks = instance.discount_breaks
    prices = instance.discount_prices
    n_tiers = len(breaks)

    best_Q = None
    best_cost = float("inf")

    for i in range(n_tiers):
        price_i = prices[i]
        h_i = holding_pct * price_i

        # EOQ for this tier
        Q_i = np.sqrt(2.0 * D * K / h_i)

        # Valid range for this tier
        lower = breaks[i]
        upper = breaks[i + 1] if i + 1 < n_tiers else float("inf")

        if Q_i < lower:
            Q_i = lower
        elif Q_i > upper:
            # This tier's EOQ is above the tier range; skip
            continue

        # Total cost = purchase + ordering + holding
        purchase = D * price_i
        ordering = (D / Q_i) * K
        holding = (Q_i / 2.0) * h_i
        total = purchase + ordering + holding

        if total < best_cost:
            best_cost = total
            best_Q = Q_i

    if best_Q is None:
        raise ValueError("No feasible solution found")

    num_orders = D / best_Q
    cycle_time = best_Q / D

    return EOQSolution(
        order_quantity=float(best_Q),
        total_cost=float(best_cost),
        reorder_point=float(best_Q),
        num_orders=float(num_orders),
        cycle_time=float(cycle_time),
    )


if __name__ == "__main__":
    from instance import textbook_eoq, backorder_eoq, discount_eoq

    # Classic EOQ
    inst = textbook_eoq()
    sol = classic_eoq(inst)
    print(f"Classic EOQ: {sol}")

    # EOQ with backorders
    inst_bo = backorder_eoq()
    sol_bo = eoq_with_backorders(inst_bo)
    print(f"EOQ with backorders: {sol_bo}")

    # EOQ with discounts
    inst_disc = discount_eoq()
    sol_disc = eoq_with_discounts(inst_disc)
    print(f"EOQ with discounts: {sol_disc}")

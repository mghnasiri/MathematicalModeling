"""
Multi-Product Newsvendor Heuristics

When multiple products share a budget or capacity constraint, the
single-product critical fractile no longer applies. These heuristics
use marginal analysis and greedy allocation.

Complexity: O(n * S * max_iter) for marginal allocation.

References:
    - Hadley, G. & Whitin, T.M. (1963). Analysis of Inventory Systems.
      Prentice-Hall, Chapter 9.
    - Lau, H.-S. & Lau, A.H.-L. (1996). The newsstand problem: A capacitated
      multiple-product single-period inventory problem. EJOR, 94(1), 29-42.
      https://doi.org/10.1016/0377-2217(95)00192-1
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

import importlib.util

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("nv_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
NewsvendorInstance = _inst.NewsvendorInstance
NewsvendorSolution = _inst.NewsvendorSolution


@dataclass
class MultiProductInstance:
    """Multi-product newsvendor with a shared budget constraint.

    Args:
        products: List of NewsvendorInstance for each product.
        budget: Total budget for ordering across all products.
    """
    products: list[NewsvendorInstance]
    budget: float

    @property
    def n_products(self) -> int:
        return len(self.products)


@dataclass
class MultiProductSolution:
    """Solution to the multi-product newsvendor.

    Args:
        order_quantities: Order quantity for each product.
        total_expected_profit: Sum of expected profits across products.
        total_cost: Total ordering cost.
        budget_used: Fraction of budget used.
    """
    order_quantities: list[float]
    total_expected_profit: float
    total_cost: float
    budget_used: float

    def __repr__(self) -> str:
        return (f"MultiProductSolution(Q={self.order_quantities}, "
                f"E[profit]={self.total_expected_profit:.2f}, "
                f"budget_used={self.budget_used:.1%})")


def marginal_allocation(instance: MultiProductInstance,
                        step: float = 1.0) -> MultiProductSolution:
    """Greedy marginal allocation for budget-constrained newsvendor.

    Iteratively allocate one unit to the product with the highest
    marginal expected profit per dollar, until budget is exhausted.

    Args:
        instance: MultiProductInstance to solve.
        step: Allocation step size.

    Returns:
        MultiProductSolution with greedy allocation.
    """
    n = instance.n_products
    quantities = np.zeros(n)
    budget_remaining = instance.budget

    # Precompute current expected profits
    profits = np.array([p.expected_profit(0) for p in instance.products])

    while budget_remaining > 0:
        best_product = -1
        best_marginal = -np.inf

        for i in range(n):
            cost_for_step = instance.products[i].unit_cost * step
            if cost_for_step > budget_remaining + 1e-9:
                continue
            new_profit = instance.products[i].expected_profit(quantities[i] + step)
            marginal = (new_profit - profits[i]) / cost_for_step
            if marginal > best_marginal:
                best_marginal = marginal
                best_product = i

        if best_product < 0 or best_marginal <= 0:
            break

        quantities[best_product] += step
        cost_used = instance.products[best_product].unit_cost * step
        budget_remaining -= cost_used
        profits[best_product] = instance.products[best_product].expected_profit(
            quantities[best_product]
        )

    total_cost = sum(
        quantities[i] * instance.products[i].unit_cost for i in range(n)
    )

    return MultiProductSolution(
        order_quantities=[float(q) for q in quantities],
        total_expected_profit=float(profits.sum()),
        total_cost=total_cost,
        budget_used=total_cost / instance.budget if instance.budget > 0 else 0,
    )


def independent_then_scale(instance: MultiProductInstance) -> MultiProductSolution:
    """Solve each product independently, then scale down to fit budget.

    First compute unconstrained optimal Q* for each product via critical
    fractile. If total cost exceeds budget, scale all quantities
    proportionally.

    Args:
        instance: MultiProductInstance to solve.

    Returns:
        MultiProductSolution with scaled quantities.
    """
    _cf = _load_parent("nv_cf", os.path.join(os.path.dirname(__file__), "..", "exact", "critical_fractile.py"))
    critical_fractile = _cf.critical_fractile

    n = instance.n_products
    unconstrained = []
    for p in instance.products:
        sol = critical_fractile(p)
        unconstrained.append(sol.order_quantity)

    quantities = np.array(unconstrained)
    total_cost = sum(quantities[i] * instance.products[i].unit_cost for i in range(n))

    if total_cost > instance.budget:
        scale = instance.budget / total_cost
        quantities = quantities * scale
        total_cost = instance.budget

    profits = sum(
        instance.products[i].expected_profit(quantities[i]) for i in range(n)
    )

    return MultiProductSolution(
        order_quantities=[float(q) for q in quantities],
        total_expected_profit=profits,
        total_cost=total_cost,
        budget_used=total_cost / instance.budget if instance.budget > 0 else 0,
    )


if __name__ == "__main__":
    products = [
        NewsvendorInstance(
            unit_cost=5.0, selling_price=12.0, salvage_value=1.0,
            demand_scenarios=np.array([30, 40, 50, 60, 70]),
        ),
        NewsvendorInstance(
            unit_cost=3.0, selling_price=8.0, salvage_value=0.5,
            demand_scenarios=np.array([20, 30, 40, 50, 60]),
        ),
    ]
    mp = MultiProductInstance(products=products, budget=300.0)
    sol_ma = marginal_allocation(mp, step=5.0)
    print(f"Marginal allocation: {sol_ma}")
    sol_is = independent_then_scale(mp)
    print(f"Independent + scale: {sol_is}")

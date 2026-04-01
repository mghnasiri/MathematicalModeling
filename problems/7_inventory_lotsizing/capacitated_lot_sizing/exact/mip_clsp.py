"""
MIP Formulation for the Capacitated Lot Sizing Problem (CLSP).

Problem: CLSP — Capacitated Lot Sizing
Notation: min Σ_t (K_t·y_t + v_t·x_t + h_t·I_t)
Complexity: NP-hard (Florian, Lenstra & Rinnooy Kan, 1980)

Formulation (standard single-item CLSP):
    Variables:
        x_t ≥ 0      : production quantity in period t
        I_t ≥ 0      : ending inventory in period t
        y_t ∈ {0, 1} : setup indicator (1 if production occurs in t)

    Minimize:
        Σ_{t=0}^{T-1} [ K_t · y_t + v_t · x_t + h_t · I_t ]

    Subject to:
        I_{t-1} + x_t = d_t + I_t       ∀t ∈ {0,...,T-1}   (flow balance)
        x_t ≤ C_t · y_t                  ∀t ∈ {0,...,T-1}   (setup linking)
        x_t, I_t ≥ 0                     ∀t
        y_t ∈ {0, 1}                     ∀t
        I_{-1} = 0                       (no initial inventory)

Uses SciPy's HiGHS solver via scipy.optimize.milp.

References:
    Pochet, Y. & Wolsey, L.A. (2006). Production Planning by Mixed
    Integer Programming. Springer, New York.
    https://doi.org/10.1007/0-387-33477-7

    Florian, M., Lenstra, J.K. & Rinnooy Kan, A.H.G. (1980).
    Deterministic production planning: Algorithms and complexity.
    Management Science, 26(7), 669-679.
    https://doi.org/10.1287/mnsc.26.7.669

    Barany, I., Van Roy, T.J. & Wolsey, L.A. (1984). Strong formulations
    for multi-item capacitated lot sizing. Management Science, 30(10),
    1255-1261. https://doi.org/10.1287/mnsc.30.10.1255
"""

from __future__ import annotations

import os
import sys
import importlib.util

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("cls_instance_mip", os.path.join(_parent_dir, "instance.py"))
CapLotSizingInstance = _inst.CapLotSizingInstance
CapLotSizingSolution = _inst.CapLotSizingSolution


def mip_clsp(
    instance: CapLotSizingInstance,
    time_limit: float = 60.0,
    initial_inventory: float = 0.0,
) -> CapLotSizingSolution:
    """Solve the CLSP exactly using Mixed Integer Programming.

    Variable layout (total = 3T variables):
        x[0..T-1]   : production quantities (continuous, ≥ 0)
        I[0..T-1]   : ending inventory (continuous, ≥ 0)
        y[0..T-1]   : setup indicators (binary)

    Args:
        instance: A CapLotSizingInstance to solve.
        time_limit: Maximum solver time in seconds.
        initial_inventory: Starting inventory (default 0).

    Returns:
        CapLotSizingSolution with optimal (or best found) plan.

    Raises:
        RuntimeError: If the solver fails to find a feasible solution.
    """
    T = instance.T
    d = instance.demands
    C = instance.capacities
    K = instance.fixed_costs
    v = instance.variable_costs
    h = instance.holding_costs

    # Variable indices
    # x: [0, T)   production
    # I: [T, 2T)  inventory
    # y: [2T, 3T) setup binary
    n_vars = 3 * T

    def x_idx(t: int) -> int:
        return t

    def i_idx(t: int) -> int:
        return T + t

    def y_idx(t: int) -> int:
        return 2 * T + t

    # ── Objective: min Σ (K_t · y_t + v_t · x_t + h_t · I_t) ──
    obj = np.zeros(n_vars)
    for t in range(T):
        obj[x_idx(t)] = v[t]
        obj[i_idx(t)] = h[t]
        obj[y_idx(t)] = K[t]

    # ── Variable bounds ──
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    integrality = np.zeros(n_vars, dtype=int)

    for t in range(T):
        ub[x_idx(t)] = C[t]          # x_t ≤ C_t
        ub[y_idx(t)] = 1.0           # y_t ≤ 1
        integrality[y_idx(t)] = 1    # y_t is binary

    bounds = Bounds(lb=lb, ub=ub)

    # ── Constraints (dense matrix approach) ──
    # 1) Flow balance (equality): I_{t-1} + x_t - I_t = d_t   for all t
    # 2) Setup linking (inequality): x_t - C_t · y_t ≤ 0      for all t

    A_eq_rows = []
    b_eq = []
    A_ub_rows = []
    b_ub = []

    for t in range(T):
        # Flow balance: x_t + I_{t-1} - I_t = d_t
        row = np.zeros(n_vars)
        row[x_idx(t)] = 1.0
        row[i_idx(t)] = -1.0
        if t > 0:
            row[i_idx(t - 1)] = 1.0
        A_eq_rows.append(row)
        b_eq.append(d[t] - (initial_inventory if t == 0 else 0.0))

    for t in range(T):
        # Setup linking: x_t - C_t · y_t ≤ 0
        row = np.zeros(n_vars)
        row[x_idx(t)] = 1.0
        row[y_idx(t)] = -C[t]
        A_ub_rows.append(row)
        b_ub.append(0.0)

    constraints = []
    A_eq = np.array(A_eq_rows)
    b_eq_arr = np.array(b_eq)
    constraints.append(LinearConstraint(A_eq, b_eq_arr, b_eq_arr))

    A_ub = np.array(A_ub_rows)
    b_ub_arr = np.array(b_ub)
    constraints.append(LinearConstraint(A_ub, -np.inf, b_ub_arr))

    # ── Solve ──
    result = milp(
        c=obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit},
    )

    if not result.success:
        raise RuntimeError(f"CLSP MIP solver failed: {result.message}")

    # ── Extract solution ──
    x_vals = result.x
    production = np.array([x_vals[x_idx(t)] for t in range(T)])
    # Snap near-integer values and clean floating-point noise
    rounded = np.round(production)
    production = np.where(np.abs(production - rounded) < 1e-4, rounded, production)
    production = np.where(production < 1e-6, 0.0, production)

    total_cost = instance.compute_cost(production)
    production_periods = [t for t in range(T) if production[t] > 1e-6]

    return CapLotSizingSolution(
        production=production,
        total_cost=total_cost,
        production_periods=production_periods,
    )


if __name__ == "__main__":
    from instance import tight_capacity_6, loose_capacity_4, variable_costs_8

    for name, factory in [
        ("Tight-6", tight_capacity_6),
        ("Loose-4", loose_capacity_4),
        ("Variable-8", variable_costs_8),
    ]:
        inst = factory()
        print(f"\n{'='*50}")
        print(f"Instance: {name} (T={inst.T})")
        print(f"  demands:    {inst.demands}")
        print(f"  capacities: {inst.capacities}")

        sol = mip_clsp(inst)
        print(f"  MIP solution: cost={sol.total_cost:.2f}")
        print(f"  production:   {sol.production}")
        print(f"  periods:      {sol.production_periods}")

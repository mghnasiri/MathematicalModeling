"""
Crop Rotation LP and Multi-Objective Optimization

Solves the crop rotation problem using:
1. LP: Revenue-maximizing allocation with sensitivity analysis
2. Pareto front: Epsilon-constraint for revenue vs nitrogen balance

Uses scipy.optimize.linprog (HiGHS backend) and the repository's
LP solver module for sensitivity analysis.

Complexity: Polynomial (LP), O(n_points * LP) for Pareto front.

References:
    Detlefsen, N.K. & Jensen, A.L. (2007). Modelling optimal crop
    sequences using network flows. Agricultural Systems, 94(2), 566-572.
    https://doi.org/10.1016/j.agsy.2007.02.002
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np
from scipy.optimize import linprog


def _load_mod(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod(
    "cr_inst",
    os.path.join(os.path.dirname(__file__), "..", "instance.py"),
)
CropRotationInstance = _inst.CropRotationInstance
CropRotationSolution = _inst.CropRotationSolution
CropAllocation = _inst.CropAllocation
ParetoFrontSolution = _inst.ParetoFrontSolution


def _build_lp_data(instance: CropRotationInstance) -> dict:
    """Build LP formulation matrices from instance."""
    nf = instance.n_fields
    nc = instance.n_crops
    n_vars = nf * nc

    def var_idx(f: int, c: int) -> int:
        return f * nc + c

    # Objective: maximize revenue => minimize -revenue
    c_obj = np.zeros(n_vars)
    for f in range(nf):
        ha = instance.fields[f].hectares
        for ci, crop in enumerate(instance.crops):
            c_obj[var_idx(f, ci)] = -instance.get_revenue_per_ha(crop, f) * ha

    # Revenue and nitrogen matrices
    revenue_matrix = np.zeros((nf, nc))
    nitrogen_matrix = np.zeros((nf, nc))
    for f in range(nf):
        ha = instance.fields[f].hectares
        for ci, crop in enumerate(instance.crops):
            revenue_matrix[f, ci] = instance.get_revenue_per_ha(crop, f) * ha
            nitrogen_matrix[f, ci] = instance.nitrogen_effect[crop] * ha

    # Inequality constraints
    ub_rows = []
    ub_rhs = []

    # Water budget
    water_row = np.zeros(n_vars)
    for f in range(nf):
        ha = instance.fields[f].hectares
        for ci, crop in enumerate(instance.crops):
            water_row[var_idx(f, ci)] = instance.water_per_ha[crop] * ha
    ub_rows.append(water_row)
    ub_rhs.append(float(instance.water_budget))

    # Labor budget
    labor_row = np.zeros(n_vars)
    for f in range(nf):
        ha = instance.fields[f].hectares
        for ci, crop in enumerate(instance.crops):
            labor_row[var_idx(f, ci)] = instance.labor_per_ha[crop] * ha
    ub_rows.append(labor_row)
    ub_rhs.append(float(instance.labor_budget))

    # Crop diversity
    max_area = instance.max_crop_fraction * instance.total_hectares
    for ci, crop in enumerate(instance.crops):
        row = np.zeros(n_vars)
        for f in range(nf):
            row[var_idx(f, ci)] = instance.fields[f].hectares
        ub_rows.append(row)
        ub_rhs.append(max_area)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    # Equality: each field gets exactly one crop
    eq_rows = []
    eq_rhs = []
    for f in range(nf):
        row = np.zeros(n_vars)
        for ci in range(nc):
            row[var_idx(f, ci)] = 1.0
        eq_rows.append(row)
        eq_rhs.append(1.0)
    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    # Bounds
    bounds = []
    for f in range(nf):
        for ci, crop in enumerate(instance.crops):
            forbidden = any(
                ff == f and cc == crop
                for ff, cc in instance.rotation_forbidden
            )
            bounds.append((0.0, 0.0 if forbidden else 1.0))

    return {
        "c_obj": c_obj,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "revenue_matrix": revenue_matrix,
        "nitrogen_matrix": nitrogen_matrix,
        "var_idx": var_idx,
    }


def _decode_solution(
    x: np.ndarray, instance: CropRotationInstance, lp_data: dict
) -> tuple[CropAllocation, float, float, float]:
    """Decode LP solution vector into allocation and resource usage."""
    nf = instance.n_fields
    nc = instance.n_crops
    var_idx = lp_data["var_idx"]

    field_crops = {}
    fractions = {}
    total_water = 0.0
    total_labor = 0.0
    total_nitrogen = 0.0

    for f in range(nf):
        best_crop = -1
        best_val = -1.0
        for ci in range(nc):
            val = x[var_idx(f, ci)]
            if val > best_val:
                best_val = val
                best_crop = ci
        field_crops[f] = instance.crops[best_crop]
        fractions[f] = best_val
        ha = instance.fields[f].hectares
        crop = instance.crops[best_crop]
        total_water += instance.water_per_ha[crop] * ha * best_val
        total_labor += instance.labor_per_ha[crop] * ha * best_val
        total_nitrogen += instance.nitrogen_effect[crop] * ha * best_val

    allocation = CropAllocation(field_crops=field_crops, fractions=fractions)
    return allocation, total_water, total_labor, total_nitrogen


def solve_lp_allocation(
    instance: CropRotationInstance,
) -> CropRotationSolution:
    """Solve crop allocation via revenue-maximizing LP.

    Args:
        instance: CropRotationInstance to solve.

    Returns:
        CropRotationSolution with LP results.
    """
    lp_data = _build_lp_data(instance)

    result = linprog(
        c=lp_data["c_obj"],
        A_ub=lp_data["A_ub"],
        b_ub=lp_data["b_ub"],
        A_eq=lp_data["A_eq"],
        b_eq=lp_data["b_eq"],
        bounds=lp_data["bounds"],
        method="highs",
    )

    if not result.success:
        return CropRotationSolution(
            success=False, total_revenue=0.0,
            allocation=CropAllocation({}, {}),
            total_water=0.0, total_labor=0.0, total_nitrogen=0.0,
            method="LP (HiGHS)",
        )

    allocation, water, labor, nitrogen = _decode_solution(
        result.x, instance, lp_data
    )

    return CropRotationSolution(
        success=True,
        total_revenue=-result.fun,
        allocation=allocation,
        total_water=water,
        total_labor=labor,
        total_nitrogen=nitrogen,
        method="LP (HiGHS)",
    )


def solve_pareto_front(
    instance: CropRotationInstance,
    n_points: int = 15,
) -> ParetoFrontSolution:
    """Compute Pareto front for revenue vs nitrogen balance.

    Uses epsilon-constraint: maximize revenue subject to nitrogen >= eps,
    sweeping eps from min to max achievable nitrogen.

    Args:
        instance: CropRotationInstance to solve.
        n_points: Number of epsilon values to sweep.

    Returns:
        ParetoFrontSolution with Pareto-efficient allocations.
    """
    lp_data = _build_lp_data(instance)
    nf = instance.n_fields
    nc = instance.n_crops
    var_idx = lp_data["var_idx"]
    n_vars = nf * nc

    # Find nitrogen range
    nitrogen_obj = np.zeros(n_vars)
    for f in range(nf):
        ha = instance.fields[f].hectares
        for ci, crop in enumerate(instance.crops):
            nitrogen_obj[var_idx(f, ci)] = -instance.nitrogen_effect[crop] * ha

    res_max_n = linprog(
        c=nitrogen_obj,
        A_ub=lp_data["A_ub"], b_ub=lp_data["b_ub"],
        A_eq=lp_data["A_eq"], b_eq=lp_data["b_eq"],
        bounds=lp_data["bounds"], method="highs",
    )
    max_nitrogen = -res_max_n.fun if res_max_n.success else 0.0

    res_min_n = linprog(
        c=-nitrogen_obj,
        A_ub=lp_data["A_ub"], b_ub=lp_data["b_ub"],
        A_eq=lp_data["A_eq"], b_eq=lp_data["b_eq"],
        bounds=lp_data["bounds"], method="highs",
    )
    min_nitrogen = res_min_n.fun if res_min_n.success else 0.0

    # Sweep epsilon
    epsilons = np.linspace(min_nitrogen, max_nitrogen, n_points)
    pareto_front = []

    for eps in epsilons:
        nitrogen_row = np.zeros(n_vars)
        for f in range(nf):
            ha = instance.fields[f].hectares
            for ci, crop in enumerate(instance.crops):
                nitrogen_row[var_idx(f, ci)] = -instance.nitrogen_effect[crop] * ha

        A_ub_ext = np.vstack([lp_data["A_ub"], nitrogen_row.reshape(1, -1)])
        b_ub_ext = np.append(lp_data["b_ub"], -eps)

        res = linprog(
            c=lp_data["c_obj"],
            A_ub=A_ub_ext, b_ub=b_ub_ext,
            A_eq=lp_data["A_eq"], b_eq=lp_data["b_eq"],
            bounds=lp_data["bounds"], method="highs",
        )

        if res.success:
            revenue = -res.fun
            allocation, _, _, nitrogen = _decode_solution(
                res.x, instance, lp_data
            )

            # Check dominance
            dominated = False
            for prev_rev, prev_nit, _ in pareto_front:
                if prev_rev >= revenue - 1e-6 and prev_nit >= nitrogen - 1e-6:
                    dominated = True
                    break

            if not dominated:
                pareto_front = [
                    (r, n, a) for r, n, a in pareto_front
                    if not (revenue >= r - 1e-6 and nitrogen >= n - 1e-6
                            and (revenue > r + 1e-6 or nitrogen > n + 1e-6))
                ]
                pareto_front.append((revenue, nitrogen, allocation))

    pareto_front.sort(key=lambda x: x[0])

    return ParetoFrontSolution(
        points=[(rev, nit) for rev, nit, _ in pareto_front],
        allocations=[alloc for _, _, alloc in pareto_front],
        n_points=len(pareto_front),
        nitrogen_range=(min_nitrogen, max_nitrogen),
    )


if __name__ == "__main__":
    inst = CropRotationInstance.standard_farm()
    print("=== Crop Rotation & Land Allocation ===\n")

    sol = solve_lp_allocation(inst)
    print(f"LP: revenue=${sol.total_revenue:,.0f}, nitrogen={sol.total_nitrogen:+,.0f} kg N")
    for f, crop in sol.allocation.field_crops.items():
        print(f"  {inst.fields[f].name} -> {crop}")

    pareto = solve_pareto_front(inst)
    print(f"\nPareto front: {pareto.n_points} points")
    for rev, nit in pareto.points:
        print(f"  revenue=${rev:,.0f}, nitrogen={nit:+,.0f} kg N")

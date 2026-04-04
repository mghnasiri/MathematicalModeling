# Land Use & Crop Selection

> **Status:** Live — see existing implementation.

## Sector context
**Sector:** Agriculture
**Phase:** Strategic Planning
**Decision-maker:** Farm manager / agricultural planner
**Decision frequency:** Once per season

## Decision question
Which crops should be planted on which parcels to maximize expected profit under yield and price uncertainty?

## OR problem mapping
**Canonical problem(s):** Robust Portfolio Optimization, LP/MIP
**Implementation:** See [`../../agriculture_crop_selection.py`](../../agriculture_crop_selection.py) and [`../../agriculture_field_assignment.py`](../../agriculture_field_assignment.py)

## Key modeling aspects
- Crop-parcel allocation under yield/price uncertainty maps to robust portfolio optimization with uncertain returns
- Field assignment to crops is a bipartite matching problem minimizing soil-crop mismatch cost
- Multi-year rotation constraints add integer variables, lifting the problem to MIP

## Data requirements
- Historical crop yields per parcel (5+ seasons) and market price distributions
- Soil quality metrics per parcel (nutrient levels, pH, drainage class)
- Per-crop input costs (seed, fertilizer, labor) and expected selling prices
- Parcel areas and any regulatory constraints on crop mix

## Canonical problem
- [Robust Portfolio Optimization](../../../problems/9_uncertainty_modeling/robust_portfolio/README.md) -- crop mix as a portfolio under price/yield uncertainty
- [Linear Programming](../../../problems/continuous/linear_programming/README.md) -- deterministic acreage allocation
- [Assignment Problem](../../../problems/4_assignment_matching/assignment/README.md) -- optimal field-to-crop matching

## Status
This decision point is part of the **Agriculture** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.

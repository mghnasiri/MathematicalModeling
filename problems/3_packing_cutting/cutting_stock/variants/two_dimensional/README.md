# Two-Dimensional Cutting Stock Problem (CSP Variant)

## What Changes

In the **two-dimensional cutting stock** variant (2D-CSP), rectangular items must
be cut from large rectangular stock sheets rather than from one-dimensional rolls.
Each item type has a width w_i, height h_i, and demand d_i. The objective is to
minimize the number of stock sheets (width W, height H) used while satisfying all
demands. Items may optionally be rotated by 90 degrees.

The key structural difference from 1D-CSP is the spatial placement constraint:
items occupy two-dimensional area rather than just length, introducing geometric
feasibility checks (no overlap, boundary respect) alongside the demand satisfaction
requirements. This makes pattern generation significantly harder.

## Mathematical Formulation

The base 1D-CSP formulation extends to 2D placement:

**Stock sheet usage:**
```
min  sum_s  y_s    (minimize sheets used)
```

**Demand satisfaction:**
```
sum_s  n_{is} >= d_i    for all item types i
```
where n_{is} is the number of items of type i cut from sheet s.

**Spatial feasibility per sheet:**
```
x_j + w_j <= W,  y_j + h_j <= H    for all items j on sheet s
no_overlap(j, k) for all item pairs on same sheet
```

**Optional rotation:** Each item may swap (w_i, h_i) to (h_i, w_i).

**Optional guillotine constraint:** Cuts must be edge-to-edge straight lines
(common in glass and wood cutting where the saw traverses the entire sheet).

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| 2D-CSP (general) | Strongly NP-hard | Lodi et al. (2002) |
| With guillotine cuts | Strongly NP-hard | Gilmore & Gomory (1965) |
| 1D-CSP (base) | NP-hard | -- |

Strongly NP-hard. The 2D spatial packing sub-problem for each sheet is itself
NP-hard, making pattern enumeration vastly more expensive than in 1D.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy largest-first (base 1D) | No | 1D packing only, ignores 2D geometry |
| Bottom-Left FFD (variant) | Yes | Sort by height, place bottom-left on each sheet |
| Shelf NFDH (variant) | Yes | Shelf-based packing within each sheet |
| SA (variant metaheuristic) | Yes | Item reordering + rotation toggling |

## Implementations

Python files in this directory:
- `instance.py` -- TwoDCSPInstance, sheet dimensions, item type demands
- `heuristics.py` -- Bottom-Left FFD, Shelf NFDH
- `metaheuristics.py` -- SA with item reordering and rotation toggling
- `tests/test_2dcsp.py` -- 25 tests

## Applications

- Glass cutting (large sheets cut into window panes)
- Sheet metal fabrication (parts cut from steel sheets)
- Plywood and MDF cutting (furniture manufacturing)
- Paper and cardboard cutting (packaging production)

## Key References

- Gilmore, P.C. & Gomory, R.E. (1965). "Multistage cutting stock problems of two and more dimensions." Operations Research 13(1), 94-120.
- Lodi, A., Martello, S. & Monaci, M. (2002). "Two-dimensional packing problems: A survey." EJOR 141(2), 241-252.
- Lodi, A., Martello, S. & Vigo, D. (1999). "Heuristic and metaheuristic approaches for a class of two-dimensional bin packing problems." INFORMS J. Computing 11(4), 345-357.

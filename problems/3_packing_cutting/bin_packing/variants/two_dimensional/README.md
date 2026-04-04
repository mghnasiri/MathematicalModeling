# 2D Bin Packing / Strip Packing (BPP Variant)

## What Changes

In the **two-dimensional bin packing** (strip packing) variant, items are
rectangles with width w_i and height h_i rather than one-dimensional sizes.
Items must be packed into a strip of fixed width W, minimizing the total
height used. No overlap is permitted and items are axis-aligned (no rotation
by default, though rotation can be optionally allowed). This extends the
1D problem from a single capacity dimension to spatial placement with
geometric feasibility constraints.

The key structural difference from 1D is that placement order matters
geometrically -- an item's position depends on the 2D skyline of previously
placed items, not just remaining capacity.

## Mathematical Formulation

The base 1D BPP capacity constraint is replaced by 2D spatial constraints:

```
min  H
s.t. x_i + w_i <= W                             for all i  (width bound)
     y_i + h_i <= H                             for all i  (height bound)
     no_overlap(i, j) for all pairs i != j:
       x_i + w_i <= x_j  OR  x_j + w_j <= x_i
       OR  y_i + h_i <= y_j  OR  y_j + h_j <= y_i
     x_i, y_i >= 0                              for all i
```

**Optional rotation:** If allowed, each item may swap (w_i, h_i) to (h_i, w_i),
introducing a binary rotation variable per item.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Strip packing (minimize H) | NP-hard | Coffman et al. (1980) |
| With rotation | NP-hard | Lodi et al. (2002) |
| Guillotine constraint | NP-hard | Lodi et al. (2002) |

The 2D problem is strictly harder than 1D: even the NFDH shelf heuristic
has asymptotic ratio 2 for strip packing, worse than FFD's 11/9 for 1D.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| FFD (base 1D heuristic) | No | Does not handle 2D placement geometry |
| Bottom-Left Decreasing Height (variant) | Yes | Sort by height, place bottom-left |
| Next-Fit Decreasing Height (variant) | Yes | Shelf-based, O(n log n) |
| Simulated Annealing (variant) | Yes | Permutation encoding, BL decoder |

## Implementations

Python files in this directory:
- `instance.py` -- TwoDB PPInstance, rectangular item representation
- `heuristics.py` -- BLDH (Bottom-Left Decreasing Height), NFDH (shelf-based)
- `metaheuristics.py` -- SA with permutation encoding and BL decoder
- `tests/test_2dbpp.py` -- 32 tests

## Applications

- **VLSI layout**: placing circuit components on a chip
- **Sheet metal cutting**: arranging parts on metal sheets
- **Textile cutting**: pattern layout on fabric rolls
- **Newspaper/web layout**: placing articles and ads on pages

## Key References

- Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980). "Performance bounds for level-oriented two-dimensional packing algorithms." SIAM J. Computing 9(4), 808-826.
- Lodi, A., Martello, S. & Vigo, D. (2002). "Two-dimensional packing problems: A survey." EJOR 141(2), 241-252.
- Jakobs, S. (1996). "On genetic algorithms for the packing of polygons." EJOR 88(1), 165-181.

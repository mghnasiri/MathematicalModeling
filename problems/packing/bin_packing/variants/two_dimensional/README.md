# 2D Bin Packing / Strip Packing (2D-BPP)

## Problem Definition

Pack n rectangular items (width w_i, height h_i) into a strip of fixed width W to minimize total height used. Items may not overlap or exceed strip boundaries.

```
min  H
s.t. x_i + w_i ≤ W                     ∀ i  (width constraint)
     y_i + h_i ≤ H                     ∀ i  (height constraint)
     no overlap between any pair (i, j)
     x_i, y_i ≥ 0                      ∀ i
```

## Complexity

NP-hard (generalizes 1D Bin Packing).

## Applications

- **VLSI layout**: placing circuit components on a chip
- **Sheet metal cutting**: arranging parts on metal sheets
- **Textile cutting**: pattern layout on fabric rolls
- **Newspaper layout**: placing articles and ads on pages

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Bottom-Left Decreasing Height | Heuristic | Coffman et al. (1980) |
| Next-Fit Decreasing Height | Heuristic | Coffman et al. (1980) |
| Simulated Annealing | Metaheuristic | Jakobs (1996) |

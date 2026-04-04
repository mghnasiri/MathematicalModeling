# 2D Strip Packing Problem (2D-SPP)

## 1. Problem Definition

- **Input:** $n$ rectangular items with widths $w_i$ and heights $h_i$; strip of fixed width $W$, unlimited height
- **Decision:** Place items on the strip (x, y coordinates)
- **Objective:** Minimize the total strip height used
- **Constraints:** No overlap; items fit within width $W$; no rotation
- **Classification:** NP-hard (Baker, Coffman & Rivest, 1980)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| NFDH | Heuristic | $O(n \log n)$ | Next-Fit Decreasing Height: one active level at a time |
| FFDH | Heuristic | $O(n^2)$ | First-Fit Decreasing Height: scan all open levels |
| Split-Fit | Heuristic | $O(n \log n)$ | Partition by width threshold, pack wide/narrow separately |
| Steinberg's | Heuristic | $O(n \log n)$ | Area-based sufficient conditions, 2 OPT bound |

### Next-Fit Decreasing Height (NFDH)

Sort items by height descending. Place items left-to-right on the current level. When an item doesn't fit, start a new level. Achieves height $\leq 2 \cdot \text{OPT} + h_{\max}$.

### FFDH (First-Fit Decreasing Height)

Like NFDH but scans all existing levels for the first one where the item fits,
reducing wasted space. Achieves height $\leq (17/10) \cdot \text{OPT} + 1$.

### Shelf Algorithm Pseudocode (NFDH)

```
NFDH(items, W):
    Sort items by height descending
    levels <- []             // list of (base_y, height, remaining_width)
    strip_height <- 0

    for each item (w_i, h_i) in sorted order:
        if levels is empty or levels.last.remaining_width < w_i:
            // Open a new level
            base_y <- strip_height
            strip_height <- strip_height + h_i
            levels.append( (base_y, h_i, W - w_i) )
            Place item at (0, base_y)
        else:
            // Place on current (last) level
            level <- levels.last
            x <- W - level.remaining_width
            Place item at (x, level.base_y)
            level.remaining_width -= w_i

    return strip_height
```

**Approximation guarantee:** NFDH produces a strip of height at most
$2 \cdot \text{OPT} + h_{\max}$ where $h_{\max}$ is the tallest item
(Coffman et al., 1980).

---

## 3. Illustrative Instance

Consider $n = 5$ items on a strip of width $W = 6$:

| Item | Width | Height |
|------|-------|--------|
| A | 4 | 5 |
| B | 3 | 3 |
| C | 2 | 3 |
| D | 3 | 2 |
| E | 2 | 1 |

Sorted by height: A(4x5), B(3x3), C(2x3), D(3x2), E(2x1).

Level 0 (h=5): A at (0,0). B does not fit (4+3=7 > 6).
Level 1 (h=3): B at (0,5). C at (3,5). D does not fit (3+3=6 ok? 3+2+3=8 > 6).
Wait -- remaining is 6-3-2=1 < 3, so D starts new level.
Level 2 (h=2): D at (0,8). E at (3,8). Remaining = 1.
Strip height = 5+3+2 = 10. OPT lower bound = max(5, total_area/6) = max(5, 37/6) = max(5, 6.17) = 6.17, so OPT >= 7.

---

## 4. Implementations in This Repository

```
strip_packing/
├── instance.py                    # StripPackingInstance, StripPackingSolution
├── heuristics/
│   └── level_algorithms.py        # NFDH, FFDH level-based packing
└── tests/
    └── test_strip_packing.py      # Strip packing test suite
```

---

## 5. Key References

- Baker, B.S., Coffman, E.G. & Rivest, R.L. (1980). Orthogonal packings in two dimensions. *SIAM J. Comput.*, 9(4), 846-855.
- Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980). Performance bounds for level-oriented 2D packing. *SIAM J. Comput.*, 9(4), 808-826.
- Steinberg, A. (1997). A strip-packing algorithm with absolute performance bound 2. *SIAM J. Comput.*, 26(2), 401-409.
- Kenyon, C. & Remila, E. (2000). A near-optimal solution to a two-dimensional cutting stock problem. *Math. Oper. Res.*, 25(4), 645-656.

# 2D Bin Packing Problem (2D-BPP)

## 1. Problem Definition

- **Input:** $n$ rectangular items with widths $w_i$ and heights $h_i$; bins of width $W$ and height $H$
- **Decision:** Assign items to bins and determine placement coordinates
- **Objective:** Minimize number of bins used
- **Constraints:** Items do not overlap; fit within bin boundaries; no rotation (oriented variant)
- **Classification:** NP-hard (Lodi, Martello & Vigo, 2002)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Shelf NFS | Heuristic | $O(n \log n)$ | Next-Fit Shelf: open new shelf when item does not fit current |
| Shelf FFS | Heuristic | $O(n^2)$ | First-Fit Shelf: scan all open shelves for first fit |
| Shelf BFS | Heuristic | $O(n^2)$ | Best-Fit Shelf: place on shelf with least remaining width |
| NFDH (strip) | Heuristic | $O(n \log n)$ | Next-Fit Decreasing Height for strip subproblem |
| Guillotine | Heuristic | $O(n^2)$ | Recursive guillotine cuts with best-fit selection |

### Shelf Algorithms

Pack items in horizontal shelves. Each shelf has a fixed height (tallest item placed on it). Items are placed left-to-right on a shelf until the bin width is exceeded, then a new shelf or bin is started.

### NFDH Pseudocode (Next-Fit Decreasing Height)

NFDH is the foundational shelf algorithm. It sorts items by height, then places
them greedily into horizontal levels within each bin, opening new bins as needed.

```
NFDH-2D-BIN-PACKING(items, W, H):
    Sort items by height descending
    bins <- [new empty bin]
    current_bin <- bins[0]
    shelf_x <- 0           // horizontal position on current shelf
    shelf_y <- 0           // vertical base of current shelf in current bin
    shelf_height <- 0      // height of current shelf

    for each item (w_i, h_i) in sorted order:
        if shelf_x + w_i <= W:
            // Fits on current shelf
            Place item at (shelf_x, shelf_y) in current_bin
            shelf_x <- shelf_x + w_i
            shelf_height <- max(shelf_height, h_i)
        else:
            // Start a new shelf
            shelf_y <- shelf_y + shelf_height
            shelf_x <- 0
            shelf_height <- h_i
            if shelf_y + h_i > H:
                // Current bin is full, open new bin
                current_bin <- new empty bin
                bins.append(current_bin)
                shelf_y <- 0
                shelf_height <- h_i
            Place item at (shelf_x, shelf_y) in current_bin
            shelf_x <- w_i

    return bins
```

Worst-case ratio: NFDH uses at most $\lceil 2 \cdot \text{OPT} \rceil$ bins for the
2D case (Coffman et al., 1980).

---

## 3. Illustrative Instance

Consider $n = 5$ items with bins of size $W = 6$, $H = 6$:

| Item | Width | Height |
|------|-------|--------|
| A | 4 | 4 |
| B | 3 | 3 |
| C | 3 | 2 |
| D | 2 | 2 |
| E | 2 | 1 |

After sorting by height: A(4x4), B(3x3), C(3x2), D(2x2), E(2x1).

Bin 1: Shelf 0 (h=4): A at (0,0). Shelf 1 (h=3): B at (0,4) -- but 4+3=7 > 6,
so B goes to Bin 2. Shelf 0 (h=3): B at (0,0), C at (3,0). Shelf 1 (h=2): D at (0,3), E at (2,3).
Result: 2 bins.

---

## 4. Implementations in This Repository

```
bin_packing_2d/
├── instance.py                        # BinPacking2DInstance, BinPacking2DSolution
├── heuristics/
│   └── shelf_algorithms.py            # Next-Fit, First-Fit, Best-Fit shelf
└── tests/
    └── test_bin_packing_2d.py         # 2D bin packing test suite
```

---

## 5. Key References

- Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing problems: A survey. *European J. Oper. Res.*, 141(2), 241-252.
- Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980). Performance bounds for level-oriented 2D packing. *SIAM J. Comput.*, 9(4), 808-826.
- Berkey, J.O. & Wang, P.Y. (1987). Two-dimensional finite bin-packing algorithms. *J. Oper. Res. Soc.*, 38(5), 423-429.

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
| Shelf algorithms | Heuristic | $O(n \log n)$ | Next-Fit Shelf, First-Fit Shelf, Best-Fit Shelf |

### Shelf Algorithms

Pack items in horizontal shelves. Each shelf has a fixed height (tallest item placed on it). Items are placed left-to-right on a shelf until the bin width is exceeded, then a new shelf or bin is started.

---

## 3. Implementations in This Repository

```
bin_packing_2d/
├── instance.py                        # BinPacking2DInstance, BinPacking2DSolution
├── heuristics/
│   └── shelf_algorithms.py            # Next-Fit, First-Fit, Best-Fit shelf
└── tests/
    └── test_bin_packing_2d.py         # 2D bin packing test suite
```

---

## 4. Key References

- Lodi, A., Martello, S. & Vigo, D. (2002). Two-dimensional packing problems: A survey. *European J. Oper. Res.*, 141(2), 241-252.
- Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980). Performance bounds for level-oriented 2D packing. *SIAM J. Comput.*, 9(4), 808-826.

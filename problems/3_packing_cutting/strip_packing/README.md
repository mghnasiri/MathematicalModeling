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
| Level algorithms | Heuristic | $O(n \log n)$ | NFDH, FFDH — pack items in horizontal levels |

### Next-Fit Decreasing Height (NFDH)

Sort items by height descending. Place items left-to-right on the current level. When an item doesn't fit, start a new level. Achieves $\leq 2 \cdot \text{OPT} + 1$ height.

---

## 3. Implementations in This Repository

```
strip_packing/
├── instance.py                    # StripPackingInstance, StripPackingSolution
├── heuristics/
│   └── level_algorithms.py        # NFDH, FFDH level-based packing
└── tests/
    └── test_strip_packing.py      # Strip packing test suite
```

---

## 4. Key References

- Baker, B.S., Coffman, E.G. & Rivest, R.L. (1980). Orthogonal packings in two dimensions. *SIAM J. Comput.*, 9(4), 846-855.
- Coffman, E.G., Garey, M.R., Johnson, D.S. & Tarjan, R.E. (1980). Performance bounds for level-oriented 2D packing. *SIAM J. Comput.*, 9(4), 808-826.

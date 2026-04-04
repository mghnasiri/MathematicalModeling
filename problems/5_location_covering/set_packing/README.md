# Maximum Weight Set Packing Problem

## 1. Problem Definition

- **Input:** Universe $U$, collection of $m$ subsets $S_i$ with weights $w_i$
- **Decision:** Select a subcollection of pairwise disjoint subsets
- **Objective:** Maximize total weight $\sum w_i$
- **Constraints:** No two selected subsets share an element
- **Classification:** NP-hard. Greedy achieves $1/k$-approximation where $k$ is maximum set size.

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy set packing | Heuristic | $O(m \log m)$ | Select heaviest non-conflicting subset greedily |

---

## 3. Implementations in This Repository

```
set_packing/
├── instance.py                    # SetPackingInstance, SetPackingSolution
├── heuristics/
│   └── greedy_sp.py               # Greedy weight-based set packing
└── tests/
    └── test_set_packing.py        # Set packing test suite
```

---

## 4. Key References

- Hurkens, C.A.J. & Schrijver, A. (1989). On the size of systems of sets every $t$ of which have an SDR. *SIAM J. Discrete Math.*, 2(1), 68-72.

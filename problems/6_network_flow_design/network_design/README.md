# Fixed-Charge Network Design Problem (FCNDP)

## 1. Problem Definition

- **Input:** Potential edges with fixed opening costs and per-unit flow costs; nodes with supply/demand
- **Decision:** Which edges to open; flow routing
- **Objective:** Minimize total cost (fixed edge costs + variable flow costs)
- **Constraints:** Flow conservation; only open edges carry flow
- **Classification:** NP-hard

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy open | Heuristic | $O(E \cdot V)$ | Greedily open edges with best cost-to-flow ratio |

---

## 3. Implementations in This Repository

```
network_design/
├── instance.py                    # NetworkDesignInstance, NetworkDesignSolution
├── heuristics/
│   └── greedy_open.py             # Greedy edge opening
└── tests/
    └── test_network_design.py     # Network design test suite
```

---

## 4. Key References

- Magnanti, T.L. & Wong, R.T. (1984). Network design and transportation planning: Models and algorithms. *Transp. Sci.*, 18(1), 1-55.

# Maximum Coverage Problem

## 1. Problem Definition

- **Input:** Universe $U$ of $n$ elements, collection of $m$ subsets $S_1, \ldots, S_m$, budget $k$
- **Decision:** Select at most $k$ subsets
- **Objective:** Maximize number of covered elements $|\bigcup_{i \in \text{selected}} S_i|$
- **Classification:** NP-hard. Greedy achieves $(1 - 1/e) \approx 0.632$ approximation (optimal for polynomial-time algorithms under standard assumptions).

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy coverage | Heuristic | $O(k \cdot m \cdot n)$ | Iteratively select subset covering most uncovered elements |

The greedy algorithm exploits submodularity of the coverage function, guaranteeing the $(1-1/e)$ ratio.

---

## 3. Implementations in This Repository

```
max_coverage/
├── instance.py                    # MaxCoverageInstance, MaxCoverageSolution
├── heuristics/
│   └── greedy_coverage.py         # Greedy (1-1/e)-approximate coverage
└── tests/
    └── test_max_coverage.py       # Max coverage test suite
```

---

## 4. Key References

- Nemhauser, G.L., Wolsey, L.A. & Fisher, M.L. (1978). An analysis of approximations for maximizing submodular set functions. *Math. Program.*, 14(1), 265-294.

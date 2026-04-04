# Chinese Postman Problem (CPP)

## 1. Problem Definition

- **Input:** Undirected weighted graph $G = (V, E, w)$
- **Decision:** Find a minimum-weight closed walk traversing every edge at least once
- **Objective:** Minimize total traversal weight
- **Constraints:** Must return to start; every edge traversed at least once
- **Classification:** **Polynomial** — $O(V^3)$ via minimum-weight perfect matching on odd-degree vertices

If the graph is Eulerian (all vertices have even degree), the optimal tour weight equals the sum of all edge weights. Otherwise, duplicate edges to make all vertices even-degree.

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Matching-based | Exact | $O(V^3)$ | Min-weight perfect matching on odd-degree vertices |

---

## 3. Implementations in This Repository

```
chinese_postman/
├── instance.py                        # CPPInstance, CPPSolution
├── exact/
│   └── chinese_postman_solver.py      # Matching-based exact solver
└── tests/
    ├── test_chinese_postman.py        # CPP test suite
    └── test_cpp.py                    # Additional CPP tests
```

---

## 4. Key References

- Kwan, M.K. (1962). Graphic programming using odd or even points. *Chinese Mathematics*, 1(1), 273-277.
- Edmonds, J. & Johnson, E.L. (1973). Matching, Euler tours and the Chinese postman. *Math. Program.*, 5(1), 88-124. https://doi.org/10.1007/BF01580113

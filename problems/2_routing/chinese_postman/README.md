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
| Euler circuit | Exact | $O(E)$ | Hierholzer's algorithm on the augmented Eulerian graph |

### Chinese Postman Algorithm Pseudocode

```
CHINESE_POSTMAN(graph):
    // Step 1: Identify odd-degree vertices
    odd_vertices = {v in V : degree(v) is odd}

    // Step 2: If graph is already Eulerian, find Euler circuit directly
    if odd_vertices is empty:
        return HIERHOLZER(graph)

    // Step 3: Compute shortest paths between all pairs of odd vertices
    for each pair (u, v) in odd_vertices:
        dist[u][v] = shortest_path_weight(u, v)

    // Step 4: Find minimum-weight perfect matching on odd vertices
    matching = MIN_WEIGHT_PERFECT_MATCHING(odd_vertices, dist)

    // Step 5: Duplicate matched edges (add shortest path edges)
    augmented_graph = copy(graph)
    for each (u, v) in matching:
        add edges along shortest_path(u, v) to augmented_graph

    // Step 6: Find Euler circuit on augmented graph
    return HIERHOLZER(augmented_graph)

HIERHOLZER(graph):
    stack = [start_vertex]
    circuit = []
    while stack not empty:
        v = stack.top()
        if v has unused edges:
            u = neighbor via an unused edge
            mark edge (v, u) as used
            stack.push(u)
        else:
            stack.pop()
            circuit.append(v)
    return reversed(circuit)
```

---

## 3. Illustrative Instance

Graph with 4 vertices and 5 edges:

| Edge | Weight |
|------|--------|
| (A,B) | 1 |
| (B,C) | 2 |
| (C,D) | 3 |
| (D,A) | 4 |
| (A,C) | 5 |

Degrees: A=3 (odd), B=2 (even), C=3 (odd), D=2 (even). Odd vertices = {A, C}. Shortest path A-C: via B costs 3, direct costs 5. Match (A,C) with cost 3. Duplicate edges A-B and B-C. Augmented graph is Eulerian. Optimal tour weight = 1+2+3+4+5+1+2 = 18.

---

## 4. Implementations in This Repository

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

## 5. Key References

- Kwan, M.K. (1962). Graphic programming using odd or even points. *Chinese Mathematics*, 1(1), 273-277.
- Edmonds, J. & Johnson, E.L. (1973). Matching, Euler tours and the Chinese postman. *Math. Program.*, 5(1), 88-124. https://doi.org/10.1007/BF01580113
- Eiselt, H.A., Gendreau, M. & Laporte, G. (1995). Arc routing problems, part I: The Chinese postman problem. *Oper. Res.*, 43(2), 231-242.
- Thimbleby, H. (2003). The directed Chinese postman problem. *Software: Practice and Experience*, 33(12), 1081-1096.

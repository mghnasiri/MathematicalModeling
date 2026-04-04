# Graph Partitioning Problem

## 1. Problem Definition

- **Input:** Undirected weighted graph $G = (V, E, w)$ with $n$ vertices, number of partitions $k$, balance tolerance $\epsilon$
- **Decision:** Assign each vertex to one of $k$ partitions
- **Objective:** Minimize total edge-cut weight (sum of weights of edges crossing partition boundaries)
- **Constraints:** Each partition has size within $(1 \pm \epsilon) \cdot n/k$ (balanced)
- **Classification:** NP-hard even for $k = 2$ (graph bisection)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of vertices |
| $k$ | Number of partitions |
| $w_{ij}$ | Weight of edge $(i, j)$ |
| $\epsilon$ | Balance tolerance |
| $\pi(v)$ | Partition assignment of vertex $v$ |

### Objective

$$\min \sum_{(i,j) \in E} w_{ij} \cdot \mathbb{1}[\pi(i) \neq \pi(j)] \tag{1}$$

$$\left\lfloor \frac{n}{k}(1-\epsilon) \right\rfloor \leq |V_p| \leq \left\lceil \frac{n}{k}(1+\epsilon) \right\rceil \quad \forall p = 1, \ldots, k \tag{2}$$

### Small Illustrative Instance

```
n = 6, k = 2, ε = 0.1
Adjacency (unit weights): (0,1), (0,2), (1,2), (2,3), (3,4), (3,5), (4,5)
Partition A = {0,1,2}, B = {3,4,5} → cut = 1 (edge 2-3 only)
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Kernighan-Lin (KL) | Heuristic | $O(n^2 \log n)$ | Iterative swap-based improvement for bisection |

### Kernighan-Lin Algorithm (1970)

Starting from an initial balanced bisection, iteratively swap vertex pairs between partitions to maximize cut reduction. Each pass locks swapped vertices. Repeat until no improving pass exists. Extended to $k$-way by recursive bisection.

---

## 4. Implementations in This Repository

```
graph_partitioning/
├── instance.py                    # GraphPartitioningInstance, GraphPartitioningSolution
│                                  #   - Fields: n, k, adjacency, balance_tolerance
├── heuristics/
│   └── greedy_kl.py               # Kernighan-Lin style partitioning
└── tests/
    └── test_graph_partitioning.py # Graph partitioning test suite
```

---

## 5. Key References

- Kernighan, B.W. & Lin, S. (1970). An efficient heuristic procedure for partitioning graphs. *Bell System Technical Journal*, 49(2), 291-307.
- Karypis, G. & Kumar, V. (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. *SIAM J. Sci. Comput.*, 20(1), 359-392.

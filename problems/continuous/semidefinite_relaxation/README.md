# SDP Relaxation for MAX-CUT

## 1. Problem Definition

- **Input:** Undirected weighted graph $G = (V, E, w)$ with $n$ vertices
- **Decision:** Partition vertices into two sets $S$ and $\bar{S}$
- **Objective:** Maximize total weight of edges crossing the partition: $\sum_{(i,j) \in E: i \in S, j \in \bar{S}} w_{ij}$
- **Classification:** NP-hard (Karp, 1972). The Goemans-Williamson SDP relaxation achieves an expected approximation ratio $\alpha_{\text{GW}} \geq 0.878$.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of vertices |
| $w_{ij}$ | Weight of edge $(i, j)$ |
| $x_i \in \{-1, +1\}$ | Partition indicator for vertex $i$ |

### Integer Formulation

$$\max \frac{1}{4} \sum_{(i,j) \in E} w_{ij} (1 - x_i x_j) \tag{1}$$

$$x_i \in \{-1, +1\} \tag{2}$$

### SDP Relaxation (Goemans-Williamson)

Relax $x_i \in \{-1,+1\}$ to unit vectors $v_i \in \mathbb{R}^n$:

$$\max \frac{1}{4} \sum_{(i,j)} w_{ij} (1 - v_i \cdot v_j) \tag{3}$$

$$v_i \cdot v_i = 1 \quad \forall i \tag{4}$$

Equivalently, optimize over $X = V^T V \succeq 0$ with $X_{ii} = 1$. Round by choosing a random hyperplane: $x_i = \text{sign}(r \cdot v_i)$.

### Small Illustrative Instance

```
n = 4, complete graph K₄ with unit weights
Max cut = 4 (partition {0,1} vs {2,3})
GW relaxation gives ≥ 0.878 × 4 = 3.51 expected
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| GW Rounding | Heuristic | $O(n^3)$ | Eigendecomposition + random hyperplane rounding |

### Goemans-Williamson Rounding

1. Solve the SDP relaxation to get PSD matrix $X$
2. Compute Cholesky/eigendecomposition: $X = V^T V$
3. Generate random vector $r \sim \mathcal{N}(0, I)$
4. Set $x_i = \text{sign}(v_i \cdot r)$

The approximation uses the eigendecomposition of the adjacency matrix as a proxy for the SDP solution (avoids requiring a full SDP solver).

---

## 4. Implementations in This Repository

```
semidefinite_relaxation/
├── instance.py                    # MaxCutInstance, MaxCutSolution
│                                  #   - Fields: n, adjacency (weight matrix)
│                                  #   - random() factory
├── heuristics/
│   └── goemans_williamson.py      # GW-style eigendecomposition + rounding
└── tests/
    └── test_maxcut.py             # MAX-CUT SDP test suite
```

---

## 5. Key References

- Goemans, M.X. & Williamson, D.P. (1995). Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming. *JACM*, 42(6), 1115-1145. https://doi.org/10.1145/227683.227684
- Karp, R.M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations* (pp. 85-103). Plenum.

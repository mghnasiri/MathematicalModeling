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
| Greedy set packing | Heuristic | $O(m \log m + m \cdot |U|)$ | Select heaviest non-conflicting subset greedily |
| ILP formulation | Exact | Exponential worst | Binary IP with conflict constraints |
| Local search | Metaheuristic | Problem-dependent | Swap/add/drop moves on selected subsets |

### Mathematical Formulation

$$\max \sum_{i=1}^{m} w_i x_i$$

$$x_i + x_j \leq 1 \quad \forall i, j \text{ where } S_i \cap S_j \neq \emptyset$$

$$x_i \in \{0, 1\}$$

This is equivalent to the maximum weight independent set problem on the conflict
graph where each subset is a node and edges connect intersecting subsets.

### Greedy Set Packing Pseudocode

```
GREEDY-SET-PACKING(S[1..m], w[1..m]):
    used_elements <- {}
    selected <- {}
    total_weight <- 0

    Sort subsets by w[i] descending

    for each subset S[i] in sorted order:
        if S[i] intersect used_elements = empty:
            selected <- selected + {i}
            used_elements <- used_elements + S[i]
            total_weight <- total_weight + w[i]

    return selected, total_weight
```

**Approximation:** For sets of maximum size $k$, the greedy algorithm achieves
a $1/k$-approximation. That is, if no set has more than $k$ elements, the greedy
solution is at least $1/k$ of optimal (Hurkens & Schrijver, 1989).

---

## 3. Illustrative Instance

Universe $U = \{1, 2, 3, 4, 5\}$, $m = 4$ subsets:

| Subset | Elements | Weight |
|--------|----------|--------|
| $S_1$ | {1, 2} | 8 |
| $S_2$ | {2, 3} | 7 |
| $S_3$ | {4, 5} | 6 |
| $S_4$ | {1, 4} | 5 |

Sorted by weight: $S_1$(8), $S_2$(7), $S_3$(6), $S_4$(5).

Greedy: Select $S_1$ (used: {1,2}). $S_2$ conflicts (element 2). Select $S_3$ (used: {1,2,4,5}).
$S_4$ conflicts (elements 1 and 4). Result: {$S_1$, $S_3$}, weight = 14.

Optimal: $S_1 + S_3$ = 14, or $S_2 + S_4$ = 12. Greedy is optimal here.

---

## 4. Implementations in This Repository

```
set_packing/
├── instance.py                    # SetPackingInstance, SetPackingSolution
├── heuristics/
│   └── greedy_sp.py               # Greedy weight-based set packing
└── tests/
    └── test_set_packing.py        # Set packing test suite
```

---

## 5. Key References

- Hurkens, C.A.J. & Schrijver, A. (1989). On the size of systems of sets every $t$ of which have an SDR. *SIAM J. Discrete Math.*, 2(1), 68-72.
- Hazan, E., Safra, S. & Schwartz, O. (2006). On the complexity of approximating $k$-set packing. *Comput. Complexity*, 15(1), 20-39.
- Chandra, B. & Halldorsson, M.M. (1999). Greedy local improvement and weighted set packing approximation. *J. Algorithms*, 39(2), 223-240.

# Set Covering Problem (SCP)

## 1. Problem Definition

- **Input:** Universe $U = \{1, \ldots, m\}$, collection of $n$ subsets $S_j$ with costs $c_j$
- **Decision:** Select a sub-collection of subsets
- **Objective:** Minimize total cost $\sum c_j x_j$
- **Constraints:** Every element in $U$ is covered by at least one selected subset
- **Classification:** NP-hard. Greedy achieves $\ln(m) + 1$ approximation (best possible unless P = NP).

---

## 2. Mathematical Formulation

$$\min \sum_{j=1}^{n} c_j x_j \tag{1}$$

$$\sum_{j: i \in S_j} x_j \geq 1 \quad \forall i \in U \tag{2}$$

$$x_j \in \{0, 1\} \tag{3}$$

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| ILP (HiGHS) | Exact | Exponential worst | MILP via `scipy.optimize.milp` |
| Greedy SCP | Heuristic | $O(m \cdot n)$ | Select cheapest cost-per-new-element subset; $\ln(m)+1$ approx |
| LP Relaxation + Rounding | Heuristic | $O(\text{LP})$ | Solve LP, round variables above $1/f$ where $f$ = max set size |
| Lagrangian Relaxation | Heuristic | $O(n \cdot T)$ | Subgradient optimization on covering constraints |

### Greedy Set Cover Pseudocode

The greedy algorithm repeatedly selects the subset with the lowest cost per newly
covered element. This achieves the best possible polynomial-time approximation ratio
of $H(m) = \ln(m) + 1$ (unless P = NP).

```
GREEDY-SET-COVER(U, S[1..n], c[1..n]):
    uncovered <- U
    selected <- {}
    total_cost <- 0

    while uncovered is not empty:
        // Find subset with minimum cost per new coverage
        best_j <- None
        best_ratio <- infinity

        for j = 1 to n:
            if j not in selected:
                new_coverage <- |S[j] intersect uncovered|
                if new_coverage > 0:
                    ratio <- c[j] / new_coverage
                    if ratio < best_ratio:
                        best_ratio <- ratio
                        best_j <- j

        selected <- selected + {best_j}
        total_cost <- total_cost + c[best_j]
        uncovered <- uncovered \ S[best_j]

    return selected, total_cost
```

The cost-effectiveness ratio $c_j / |S_j \cap \text{uncovered}|$ is re-evaluated at
each step because the marginal coverage changes as elements become covered.

---

## 4. Illustrative Instance

Universe $U = \{1, 2, 3, 4, 5\}$ with $n = 4$ subsets:

| Subset | Elements | Cost | Initial cost/coverage |
|--------|----------|------|-----------------------|
| $S_1$ | {1, 2, 3} | 6 | 6/3 = 2.00 |
| $S_2$ | {2, 4} | 3 | 3/2 = 1.50 |
| $S_3$ | {3, 4, 5} | 5 | 5/3 = 1.67 |
| $S_4$ | {1, 5} | 4 | 4/2 = 2.00 |

Greedy step 1: Best ratio is $S_2$ (1.50). Select $S_2$, covered = {2,4}, cost = 3.
Step 2: $S_1$ covers {1,3} at 6/2=3.0, $S_3$ covers {3,5} at 5/2=2.5, $S_4$ covers {1,5} at 4/2=2.0.
Select $S_4$, covered = {1,2,4,5}, cost = 7.
Step 3: Only element 3 remains. $S_1$ covers {3} at 6/1=6.0, $S_3$ covers {3} at 5/1=5.0.
Select $S_3$, total cost = 12.

Optimal: $S_1 + S_3$ covers all elements at cost 6+5 = 11.

---

## 5. Implementations in This Repository

```
set_covering/
├── instance.py                    # SetCoveringInstance, SetCoveringSolution
├── exact/
│   └── ilp_scp.py                 # ILP formulation
├── heuristics/
│   └── greedy_scp.py              # Greedy set covering
└── tests/
    └── test_set_covering.py       # SCP test suite
```

---

## 6. Key References

- Chvatal, V. (1979). A greedy heuristic for the set-covering problem. *Math. Oper. Res.*, 4(3), 233-235.
- Caprara, A., Fischetti, M. & Toth, P. (1999). A heuristic method for the set covering problem. *Oper. Res.*, 47(5), 730-743.
- Beasley, J.E. (1990). A Lagrangian heuristic for set-covering problems. *Naval Research Logistics*, 37(1), 151-164.
- Balas, E. & Carrera, M.C. (1996). A dynamic subgradient-based branch-and-bound procedure for set covering. *Oper. Res.*, 44(6), 875-890.

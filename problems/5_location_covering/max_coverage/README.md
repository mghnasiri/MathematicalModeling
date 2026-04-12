# Maximum Coverage Problem

## 1. Problem Definition

- **Input:** Universe $U$ of $n$ elements, collection of $m$ subsets $S_1, \ldots, S_m$, budget $k$
- **Decision:** Select at most $k$ subsets
- **Objective:** Maximize number of covered elements $|\bigcup_{i \in \text{selected}} S_i|$
- **Classification:** NP-hard. Greedy achieves $(1 - 1/e) \approx 0.632$ approximation (optimal for polynomial-time algorithms under standard assumptions).

Maximum coverage arises in facility placement (cover the most customers with $k$
facilities), sensor deployment (monitor the most area), advertising (reach the most
users with $k$ ads), and influence maximization in social networks.

### Relationship to Set Cover

Maximum coverage is the dual perspective of set cover: instead of minimizing cost
to cover everything, we maximize coverage under a budget constraint. While set cover
asks "how cheaply can I cover all elements?", max coverage asks "how much can I cover
with a fixed budget?".

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy coverage | Heuristic | $O(k \cdot m \cdot n)$ | Iteratively select subset covering most uncovered elements |
| LP Relaxation + Rounding | Heuristic | $O(\text{LP})$ | Solve LP relaxation, round variables exceeding threshold |
| Local Search | Metaheuristic | $O(k \cdot m \cdot n)$ per swap | Swap selected/unselected subsets to improve coverage |

The greedy algorithm exploits submodularity of the coverage function, guaranteeing the $(1-1/e)$ ratio.

### Greedy (1 - 1/e)-Approximation Pseudocode

```
GREEDY-MAX-COVERAGE(U, S[1..m], k):
    covered <- {}
    selected <- {}

    for round = 1 to k:
        best_set <- None
        best_gain <- -1

        for j = 1 to m:
            if j not in selected:
                gain <- |S[j] \ covered|   // newly covered elements
                if gain > best_gain:
                    best_gain <- gain
                    best_set <- j

        selected <- selected + {best_set}
        covered <- covered + S[best_set]

    return selected, |covered|
```

**Approximation ratio:** The greedy algorithm achieves coverage of at least
$(1 - 1/e) \cdot \text{OPT} \approx 0.632 \cdot \text{OPT}$. This bound is
tight: no polynomial-time algorithm can do better unless P = NP
(Feige, 1998).

### Submodularity

The coverage function $f(T) = |\bigcup_{i \in T} S_i|$ is monotone and submodular,
meaning adding a set to a smaller collection never yields less marginal gain than
adding it to a larger collection. This property is what makes the greedy bound hold.

---

## 3. Illustrative Instance

Universe $U = \{1, 2, 3, 4, 5, 6\}$, budget $k = 2$:

| Subset | Elements |
|--------|----------|
| $S_1$ | {1, 2, 3} |
| $S_2$ | {2, 4, 5} |
| $S_3$ | {3, 5, 6} |
| $S_4$ | {1, 6} |

Round 1: $S_1$ covers 3 new, $S_2$ covers 3, $S_3$ covers 3, $S_4$ covers 2.
Tie-break selects $S_1$. Covered = {1,2,3}.
Round 2: $S_2$ covers {4,5} = 2 new, $S_3$ covers {5,6} = 2, $S_4$ covers {6} = 1.
Tie-break selects $S_2$. Covered = {1,2,3,4,5}. Total coverage = 5 out of 6.

Optimal: $S_1 + S_3$ covers {1,2,3,5,6} = 5, or $S_2 + S_3$ covers {2,3,4,5,6} = 5.
Maximum possible with $k=2$ is 5, so greedy is optimal here.

---

## 4. Implementations in This Repository

```
max_coverage/
├── instance.py                    # MaxCoverageInstance, MaxCoverageSolution
├── heuristics/
│   └── greedy_coverage.py         # Greedy (1-1/e)-approximate coverage
└── tests/
    └── test_max_coverage.py       # Max coverage test suite
```

---

## 5. Key References

- Nemhauser, G.L., Wolsey, L.A. & Fisher, M.L. (1978). An analysis of approximations for maximizing submodular set functions. *Math. Program.*, 14(1), 265-294.
- Feige, U. (1998). A threshold of $\ln n$ for approximating set cover. *J. ACM*, 45(4), 634-652.
- Khuller, S., Moss, A. & Naor, J. (1999). The budgeted maximum coverage problem. *Inform. Process. Lett.*, 70(1), 39-45.

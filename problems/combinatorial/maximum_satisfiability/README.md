# Maximum Satisfiability Problem (MAX-SAT)

## 1. Problem Definition

- **Input:** Boolean formula in CNF with $n$ variables, $m$ weighted clauses $C_j$ with weights $w_j$
- **Decision:** Truth assignment $x \in \{0, 1\}^n$
- **Objective:** Maximize total weight of satisfied clauses $\sum_{j: C_j \text{ satisfied}} w_j$
- **Classification:** NP-hard. MAX-2SAT is APX-complete. Random assignment gives 1/2-approximation; SDP relaxation gives 0.878 for MAX-2SAT (Goemans & Williamson, 1994).

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of Boolean variables |
| $m$ | Number of clauses |
| $C_j$ | Clause $j$ (disjunction of literals) |
| $w_j$ | Weight of clause $j$ |
| $x_i$ | Truth value of variable $i$ ($0$ or $1$) |

### ILP Formulation

$$\max \sum_{j=1}^{m} w_j z_j \tag{1}$$

$$\sum_{i \in C_j^+} x_i + \sum_{i \in C_j^-} (1 - x_i) \geq z_j \quad \forall j \tag{2}$$

$$x_i \in \{0,1\}, \quad z_j \in \{0,1\} \tag{3}$$

where $C_j^+$ and $C_j^-$ are the sets of positive and negative literals in clause $j$.

### Small Illustrative Instance

```
n = 3 variables, m = 4 clauses
C₁ = (x₁ ∨ x₂),      w₁ = 3
C₂ = (¬x₁ ∨ x₃),     w₂ = 2
C₃ = (x₂ ∨ ¬x₃),     w₃ = 1
C₄ = (¬x₁ ∨ ¬x₂),    w₄ = 4

Assignment x = (0, 1, 1): satisfies C₁, C₂, C₃, C₄ → total = 10 (all)
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy MAX-SAT | Heuristic | $O(n \cdot m)$ | Set each variable to maximize satisfied clause weight |

### Greedy MAX-SAT

For each variable (in some order), compute the total weight of clauses satisfied by setting it to true vs. false. Assign the value that maximizes weight. The Johnson-Lovász greedy achieves a 1/2-approximation.

#### Greedy MAX-SAT Pseudocode

```
GREEDY_MAXSAT(variables x_1..x_n, clauses C_1..C_m with weights w_j):
    for i = 1 to n:
        W_true = sum of w_j for clauses newly satisfied by x_i = TRUE
        W_false = sum of w_j for clauses newly satisfied by x_i = FALSE
        if W_true >= W_false:
            x_i = TRUE
        else:
            x_i = FALSE
        mark all clauses satisfied by this assignment
    return assignment x, total weight of satisfied clauses
```

**Guarantee:** The greedy approach satisfies clauses with total weight at least $W^*/2$, where $W^*$ is the optimum. For MAX-$k$-SAT (each clause has at most $k$ literals), a random assignment achieves a $(1 - 2^{-k})$-approximation in expectation. The derandomized version by Johnson (1974) matches this guarantee deterministically.

### Relationship to SAT

- **SAT** (decision): Is there an assignment satisfying all clauses? NP-complete (Cook, 1971).
- **MAX-SAT** (optimization): Maximize satisfied clause weight. NP-hard.
- **MAX-2SAT**: APX-complete; best polynomial ratio is 0.9401 (Lewin, Livnat & Zwick, 2007).
- **MAX-3SAT**: $(7/8)$-approximable but not $(7/8 + \epsilon)$ unless P = NP.

### Applications

- **VLSI design** (satisfying as many design constraints as possible)
- **AI planning** (maximizing achieved goals under conflicting constraints)
- **Software verification** (maximizing satisfied assertions during testing)

---

## 4. Implementations in This Repository

```
maximum_satisfiability/
├── instance.py                    # MaxSATInstance, MaxSATSolution
│                                  #   - Fields: n_vars, clauses (signed literals), weights
├── heuristics/
│   └── greedy_maxsat.py           # Greedy variable-setting heuristic
└── tests/
    └── test_maxsat.py             # MAX-SAT test suite
```

---

## 5. Key References

- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability*. W.H. Freeman.
- Goemans, M.X. & Williamson, D.P. (1995). Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming. *JACM*, 42(6), 1115-1145. https://doi.org/10.1145/227683.227684
- Johnson, D.S. (1974). Approximation algorithms for combinatorial problems. *JCSS*, 9(3), 256-278.
- Hastad, J. (2001). Some optimal inapproximability results. *JACM*, 48(4), 798-859. https://doi.org/10.1145/502090.502098
- Williamson, D.P. & Shmoys, D.B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.

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

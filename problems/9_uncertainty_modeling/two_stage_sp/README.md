# Two-Stage Stochastic Programming (2SSP)

## 1. Problem Definition

- **Input:** First-stage cost $c$, constraint matrix $A$, RHS $b$; for each scenario $s$: recourse cost $q(s)$, technology matrix $T(s)$, recourse matrix $W(s)$, RHS $h(s)$; probabilities $p_s$
- **Decision:** First-stage $x$ (before uncertainty); second-stage $y(s)$ (after scenario $s$ is revealed)
- **Objective:** Minimize total expected cost $c^T x + \sum_s p_s \, q(s)^T y(s)$
- **Constraints:** First-stage feasibility $Ax = b, x \geq 0$; recourse feasibility $T(s)x + W(s)y(s) \leq h(s), y(s) \geq 0$ for all $s$
- **Classification:** NP-hard in general. Deterministic equivalent LP has $O(n_1 + S \cdot n_2)$ variables.

### Key Concepts

| Term | Definition |
|------|------------|
| **Recourse** | Second-stage corrective action after uncertainty is observed |
| **Deterministic Equivalent** | Large LP expanding all scenarios explicitly |
| **Expected Value (EV)** | Solution using mean scenario — lower bound |
| **VSS** | Value of Stochastic Solution — benefit of stochastic over deterministic approach |
| **EVPI** | Expected Value of Perfect Information — upper bound on information value |
| **SAA** | Sample Average Approximation — solve on random subset of scenarios |

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $x \in \mathbb{R}^{n_1}$ | First-stage decision vector |
| $y(s) \in \mathbb{R}^{n_2}$ | Second-stage recourse vector for scenario $s$ |
| $c \in \mathbb{R}^{n_1}$ | First-stage cost vector |
| $q(s) \in \mathbb{R}^{n_2}$ | Second-stage cost vector under scenario $s$ |
| $A \in \mathbb{R}^{m_1 \times n_1}$ | First-stage constraint matrix |
| $T(s) \in \mathbb{R}^{m_2 \times n_1}$ | Technology matrix (links stages) |
| $W(s) \in \mathbb{R}^{m_2 \times n_2}$ | Recourse matrix |
| $S$ | Number of scenarios |
| $p_s$ | Probability of scenario $s$ |

### General Formulation

$$\min_{x} \quad c^T x + \sum_{s \in S} p_s \, q(s)^T y(s) \tag{1}$$

$$Ax = b \tag{2}$$

$$T(s)x + W(s)y(s) \leq h(s) \quad \forall s \in S \tag{3}$$

$$x \geq 0, \quad y(s) \geq 0 \quad \forall s \tag{4}$$

### Deterministic Equivalent (Extensive Form)

Stack all scenarios into one large LP:

$$\min \begin{pmatrix} c^T & p_1 q_1^T & \cdots & p_S q_S^T \end{pmatrix} \begin{pmatrix} x \\ y_1 \\ \vdots \\ y_S \end{pmatrix}$$

Size: $(m_1 + S \cdot m_2)$ constraints, $(n_1 + S \cdot n_2)$ variables. Grows linearly with $S$.

### Small Illustrative Instance (Newsvendor as 2SSP)

```
First stage: x = order quantity, cost c = 10
Scenarios: D ∈ {50, 75, 100}, equal probability
Second stage: y1 = units sold (revenue p=25), y2 = unsold (salvage v=5)

Per scenario s:
  min  10x + p_s * (-25*y1_s - 5*y2_s)
  s.t. y1_s + y2_s = x       (inventory balance)
       y1_s <= D_s             (can't sell more than demand)
       x, y1_s, y2_s >= 0

Optimal: x* = 75 (matches critical fractile of newsvendor)
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Deterministic Equivalent | Exact | $O(\text{LP}(n_1 + S n_2))$ | Expand all scenarios into one LP (HiGHS) |
| Expected Value Solution | Heuristic | $O(\text{LP}(n_1 + n_2))$ | Solve with mean scenario as proxy |
| SAA | Metaheuristic | $O(R \cdot \text{LP}(n_1 + N n_2))$ | Solve $R$ replications of $N$-sample subproblems |

### Deterministic Equivalent

Constructs a single LP by duplicating the second-stage variables and constraints for each scenario. Solved directly via HiGHS. Exact for discrete scenarios but size grows with $S$.

### Expected Value (EV) Solution

Replace all scenarios with their expected value $\bar{\xi} = \sum_s p_s \xi_s$. Solve the resulting deterministic problem. Provides a lower bound; the gap to the stochastic solution is the **VSS**.

### Sample Average Approximation (SAA)

1. Draw $R$ independent samples of $N$ scenarios each
2. Solve the deterministic equivalent for each sample
3. Evaluate candidate solutions on a large validation set
4. Construct confidence intervals on the optimality gap

---

## 4. Implementations in This Repository

```
two_stage_sp/
├── instance.py                        # TwoStageSPInstance, TwoStageSPSolution
│                                      #   - newsvendor_as_2ssp() factory
│                                      #   - capacity_planning() factory
├── heuristics/
│   └── deterministic_equivalent.py    # Extensive form LP (HiGHS), EV solution
├── metaheuristics/
│   └── sample_average.py             # SAA with replications
└── tests/
    └── test_two_stage_sp.py          # 10 tests, 4 test classes
```

---

## 5. Key References

- Dantzig, G.B. (1955). Linear programming under uncertainty. *Management Science*, 1(3-4), 197-206. https://doi.org/10.1287/mnsc.1.3-4.197
- Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed. Springer. https://doi.org/10.1007/978-1-4614-0237-4
- Kleywegt, A.J., Shapiro, A. & Homem-de-Mello, T. (2002). The sample average approximation method for stochastic discrete optimization. *SIAM J. Optim.*, 12(2), 479-502. https://doi.org/10.1137/S1052623499363220
- Van Slyke, R.M. & Wets, R. (1969). L-shaped linear programs with applications to optimal control and stochastic programming. *SIAM J. Appl. Math.*, 17(4), 638-663.

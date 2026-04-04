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

### Structural Properties

**Fixed recourse:** When $W(s) = W$ and $q(s) = q$ are identical across all
scenarios, the problem has *fixed recourse*. Only the technology matrix $T(s)$
and RHS $h(s)$ vary with the scenario. Both factory instances in this
repository (`newsvendor_as_2ssp`, `capacity_planning`) have fixed recourse
matrices, which simplifies analysis because the feasible set of the second
stage has the same structure for every scenario.

**Relatively complete recourse:** For any feasible first-stage decision $x$
and any scenario $s$, the second-stage problem has a feasible solution. This
property guarantees that no feasibility cuts are needed in the L-shaped
method, and only optimality cuts are required for convergence. The newsvendor
formulation satisfies this because the inventory balance constraint
$y_1 + y_2 = x$ always admits a non-negative solution (set $y_1 = \min(x, D_s)$
and $y_2 = x - y_1$).

**Complete recourse:** A stronger condition where $\{y : Wy \leq r\}$ is
non-empty for *every* right-hand side $r$. This holds when $W$ contains an
identity submatrix (slack variables), ensuring feasibility regardless of
the first-stage decision. Complete recourse implies relatively complete
recourse.

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

#### Extensive Form Size Discussion

The deterministic equivalent is practical for moderate scenario counts but becomes
a bottleneck for large-scale problems. Concretely, a problem with $n_1 = 10$
first-stage variables, $n_2 = 50$ recourse variables per scenario, and $S = 10{,}000$
scenarios produces an LP with $10 + 500{,}000 = 500{,}010$ columns. Memory and
solution time both grow as $O(S \cdot (n_2 + m_2))$. This motivates decomposition
methods (L-shaped, progressive hedging) and sampling approaches (SAA) for
problems with continuous or very large discrete scenario sets.

### L-Shaped (Benders) Decomposition

The L-shaped method (Van Slyke & Wets, 1969) decomposes the extensive form into
a **master problem** (first-stage) and $S$ independent **subproblems** (one per
scenario). It avoids constructing the full extensive form matrix, instead passing
information between stages via optimality cuts.

**Master problem** (relaxed first-stage):

$$\min_{x, \theta} \quad c^T x + \theta$$
$$Ax = b, \quad x \geq 0$$

where $\theta$ is a scalar approximating $\mathcal{Q}(x) = \sum_s p_s Q_s(x)$.

**Subproblem for scenario $s$** (given $\hat{x}$ from master):

$$Q_s(\hat{x}) = \min_{y_s} \quad q(s)^T y_s$$
$$W(s) y_s \leq h(s) - T(s)\hat{x}, \quad y_s \geq 0$$

The dual of the subproblem yields multipliers $\pi_s$ that define
optimality cuts added to the master:

$$\theta \geq \sum_{s} p_s \bigl[ \pi_s^T h(s) - \pi_s^T T(s) x \bigr]$$

Each iteration adds one cut, progressively tightening the approximation of
the recourse function. The algorithm terminates when the gap between the
master objective and the best known upper bound falls below tolerance $\varepsilon$.
For LP subproblems with relatively complete recourse, convergence is finite
(Birge & Louveaux, 2011, Ch. 5).

### EVPI and VSS: Measuring the Value of Information

Two fundamental quantities measure how much the stochastic model gains over
simpler approaches.

**Wait-and-See (WS) solution:** Solve each scenario independently assuming
perfect foreknowledge, then take the expectation:

$$WS = \sum_{s} p_s \min_{x, y_s} \bigl\{ c^T x + q(s)^T y_s : Ax = b,\; T(s)x + W(s)y_s \leq h(s) \bigr\}$$

**Recourse Problem (RP):** The true two-stage stochastic optimum (Eq. 1).

**Expected Value of Perfect Information (EVPI):**

$$\text{EVPI} = RP - WS \geq 0$$

This is the maximum amount a decision-maker should pay for a perfect forecast.
When EVPI is small, the uncertainty has limited impact on optimal decisions.

**Expected result of the EV solution (EEV):** Take the EV first-stage
decision $x^{EV}$ (obtained by solving on the mean scenario) and evaluate
its expected recourse cost across all original scenarios:

$$EEV = c^T x^{EV} + \sum_s p_s Q_s(x^{EV})$$

**Value of the Stochastic Solution (VSS):**

$$\text{VSS} = EEV - RP \geq 0$$

This quantifies the cost of ignoring uncertainty. A large VSS justifies
the computational overhead of solving the stochastic program. By definition
$WS \leq RP \leq EEV$, so both EVPI and VSS are non-negative.

### Small Illustrative Instance (Newsvendor as 2SSP)

```
First stage: x = order quantity, cost c = 10
Scenarios: D ∈ {50, 75, 100}, equal probability (p_s = 1/3)
Second stage: y1 = units sold (revenue p=25), y2 = unsold (salvage v=5)

Per scenario s:
  min  10x + p_s * (-25*y1_s - 5*y2_s)
  s.t. y1_s + y2_s = x       (inventory balance)
       y1_s <= D_s             (can't sell more than demand)
       x, y1_s, y2_s >= 0

Optimal: x* = 75 (matches critical fractile of newsvendor)
```

#### Numerical Walkthrough

Parameters: $c = 10$, $p = 25$, $v = 5$, so overage cost $c_o = c - v = 5$,
underage cost $c_u = p - c = 15$. Critical fractile $= c_u / (c_u + c_o) = 15/20 = 0.75$.

**Extensive form** (3 scenarios, 1 first-stage + 6 second-stage variables, 3 constraints):

| Scenario $s$ | $D_s$ | $p_s$ | $y_1^* = \min(x, D_s)$ | $y_2^* = x - y_1^*$ | Recourse cost $-25 y_1 - 5 y_2$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 50 | 1/3 | 50 | 25 | $-25(50) - 5(25) = -1375$ |
| 2 | 75 | 1/3 | 75 | 0 | $-25(75) - 5(0) = -1875$ |
| 3 | 100 | 1/3 | 75 | 0 | $-25(75) - 5(0) = -1875$ |

At $x^* = 75$:
- First-stage cost: $10 \times 75 = 750$
- Expected recourse: $(1/3)(-1375 + -1875 + -1875) = -1708.33$
- **RP** $= 750 - 1708.33 = -958.33$

**Wait-and-See (WS):** Solve per scenario knowing demand:
- $D=50$: order 50, profit $= 50 \times 25 - 50 \times 10 = 750$, cost $= -750$
- $D=75$: order 75, profit $= 75 \times 25 - 75 \times 10 = 1125$, cost $= -1125$
- $D=100$: order 100, profit $= 100 \times 25 - 100 \times 10 = 1500$, cost $= -1500$
- $WS = (1/3)(-750 + -1125 + -1500) = -1125.00$

**EVPI** $= RP - WS = -958.33 - (-1125.00) = 166.67$

**EV solution:** Mean demand $\bar{D} = 75$, so $x^{EV} = 75$. In this case
the EV solution coincides with RP ($EEV = RP$), giving $VSS = 0$. This is a
special case; for asymmetric cost structures (e.g., $c_o \neq c_u$) with more
skewed demand distributions, VSS is typically positive.

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Deterministic Equivalent | Exact | $O(\text{LP}(n_1 + S n_2))$ | Expand all scenarios into one LP (HiGHS) |
| Expected Value Solution | Heuristic | $O(\text{LP}(n_1 + n_2))$ | Solve with mean scenario as proxy |
| SAA | Metaheuristic | $O(R \cdot \text{LP}(n_1 + N n_2))$ | Solve $R$ replications of $N$-sample subproblems |

### Deterministic Equivalent

Constructs a single LP by duplicating the second-stage variables and constraints for each scenario. Solved directly via HiGHS. Exact for discrete scenarios but size grows with $S$.

Implementation (`heuristics/deterministic_equivalent.py`):
1. Build objective vector $[c,\; p_1 q_1,\; \ldots,\; p_S q_S]$ of length $n_1 + S \cdot n_2$
2. Assemble block-diagonal inequality matrix with $T(s)$ linking first-stage to each scenario
3. Call `scipy.optimize.linprog` with HiGHS backend
4. Extract $x^*$ and per-scenario recourse $y_s^*$

### Expected Value (EV) Solution

Replace all scenarios with their expected value $\bar{\xi} = \sum_s p_s \xi_s$. Solve the resulting deterministic problem. The EV first-stage decision $x^{EV}$, when evaluated against all original scenarios, yields the EEV (see Section 2). The gap $VSS = EEV - RP$ measures the cost of ignoring uncertainty.

```
EV SOLUTION PSEUDOCODE
1.  Compute mean scenario: q_bar = Σ p_s q(s), T_bar = Σ p_s T(s),
                            W_bar = Σ p_s W(s), h_bar = Σ p_s h(s)
2.  Construct single-scenario instance with (c, A, b, q_bar, T_bar, W_bar, h_bar)
3.  Solve via deterministic equivalent → x_EV
4.  FOR each original scenario s = 1..S:
      Solve recourse LP: min q(s)^T y  s.t. W(s)y ≤ h(s) - T(s)x_EV, y ≥ 0
5.  EEV = c^T x_EV + Σ p_s Q_s(x_EV)
6.  VSS = EEV - RP
```

### Sample Average Approximation (SAA)

SAA replaces the full scenario set with a small random sample, making the
deterministic equivalent tractable for problems with very large or continuous
scenario spaces.

```
SAA PSEUDOCODE
Input: instance with S scenarios, sample size N, replications R, seed
1.  FOR r = 1 to R:
      a. Draw N scenarios uniformly with replacement from {1..S}
      b. Assign equal probability 1/N to each sampled scenario
      c. Build DEP with N scenarios; solve via linprog (HiGHS)
      d. Extract candidate first-stage decision x_r
      e. EVALUATE x_r on ALL S original scenarios:
           FOR s = 1 to S:
             Solve recourse LP: min q(s)^T y  s.t. W(s)y ≤ h(s) - T(s)x_r
           cost_r = c^T x_r + Σ p_s Q_s(x_r)
      f. Record cost_r
2.  best = argmin_r cost_r
3.  Compute statistics: mean, std, confidence interval across R replications
4.  RETURN x_best, mean ± z_{α/2} * std / √R
```

#### SAA Convergence Properties

By the **law of large numbers**, the SAA objective $\hat{v}_N = \min_x \{ c^T x + (1/N) \sum_{i=1}^{N} Q(x, \xi_i) \}$ converges almost surely to the true optimal value $v^*$ as $N \to \infty$ (Kleywegt, Shapiro & Homem-de-Mello, 2002). Key results:

- **Consistency:** $\hat{v}_N \to v^*$ a.s. as $N \to \infty$, and every cluster point of SAA optimal solutions is an optimal solution of the true problem.
- **Rate:** Under regularity conditions, $\sqrt{N}(\hat{v}_N - v^*)$ is asymptotically normal, so confidence intervals shrink as $O(1/\sqrt{N})$.
- **Bias:** $E[\hat{v}_N] \leq v^*$ (optimistic bias from optimizing over the sample), but the bias vanishes as $N$ grows. The evaluation step (step 1e above) on all $S$ scenarios provides an unbiased upper bound.
- **Practical guidance:** Shapiro et al. (2009) recommend $R = 20$--$40$ replications with $N$ large enough that the confidence interval half-width is within 1--5% of the objective value.

### L-Shaped Method (Benders Decomposition)

Not implemented in this repository, but the algorithmic outline is included
for reference as it is the classical decomposition approach for 2SSP.

```
L-SHAPED METHOD PSEUDOCODE
Input: instance, tolerance ε
1.  Initialize: UB = +∞, LB = -∞, cut_set = ∅
2.  REPEAT:
      a. Solve master problem:
           min c^T x + θ
           s.t. Ax = b, x ≥ 0
                optimality cuts from cut_set
         → x*, θ*
      b. LB = c^T x* + θ*
      c. FOR each scenario s = 1..S:
           Solve subproblem (recourse LP with fixed x*):
             min q(s)^T y_s  s.t. W(s)y_s ≤ h(s) - T(s)x*
           → optimal value Q_s(x*), dual multipliers π_s
      d. UB = min(UB, c^T x* + Σ p_s Q_s(x*))
      e. IF UB - LB < ε: RETURN x*, UB   (converged)
      f. Add optimality cut to cut_set:
           θ ≥ Σ_s p_s [π_s^T h(s) - π_s^T T(s) x]
    UNTIL convergence
```

The method converges finitely for LP subproblems with relatively complete
recourse. Each iteration adds at most one cut to the master, and the number
of distinct extreme points of the dual polyhedron is finite (Van Slyke &
Wets, 1969). In practice, multi-cut variants add one cut per scenario,
improving convergence at the cost of a larger master problem.

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

### Data Structures

**`TwoStageSPInstance`** stores the first-stage cost vector `c`, constraint
matrix `A`, RHS `b`, a list of scenario dictionaries (each with keys `q`, `T`,
`W`, `h`), and scenario probabilities (defaulting to uniform). Properties
`n1`, `n2`, `m1`, `m2`, `n_scenarios` expose dimensions. Two `@classmethod`
factories generate standard instances:

- `newsvendor_as_2ssp(unit_cost, selling_price, salvage_value, demand_scenarios)`:
  1 first-stage variable (order quantity), 2 recourse variables per scenario
  (units sold, units unsold), 1 constraint per scenario (inventory balance).
- `capacity_planning(n_facilities, n_scenarios, seed)`: `n_facilities` first-stage
  capacity variables, `n_facilities * n_customers` recourse allocation variables,
  `n_facilities + n_customers` constraints per scenario (capacity + demand).

**`TwoStageSPSolution`** stores the first-stage decision `x`, cost breakdown
(`first_stage_cost`, `expected_recourse_cost`, `total_cost`), and an optional
dictionary of per-scenario recourse vectors.

**`SAASolution`** (in `sample_average.py`) wraps the best solution across
replications along with per-replication objective values, mean, standard
deviation, and replication count for confidence interval construction.

---

## 5. Key References

1. Dantzig, G.B. (1955). Linear programming under uncertainty. *Management Science*, 1(3-4), 197-206. https://doi.org/10.1287/mnsc.1.3-4.197
2. Van Slyke, R.M. & Wets, R.J.-B. (1969). L-shaped linear programs with applications to optimal control and stochastic programming. *SIAM J. Appl. Math.*, 17(4), 638-663. https://doi.org/10.1137/0117061
3. Wets, R.J.-B. (1974). Stochastic programs with fixed recourse: The equivalent deterministic program. *SIAM Review*, 16(3), 309-339. https://doi.org/10.1137/1016053
4. Kall, P. & Wallace, S.W. (1994). *Stochastic Programming*. Wiley. ISBN 978-0471951582.
5. Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed. Springer. https://doi.org/10.1007/978-1-4614-0237-4
6. Shapiro, A., Dentcheva, D. & Ruszczynski, A. (2009). *Lectures on Stochastic Programming: Modeling and Theory*. SIAM. https://doi.org/10.1137/1.9780898718751
7. Kleywegt, A.J., Shapiro, A. & Homem-de-Mello, T. (2002). The sample average approximation method for stochastic discrete optimization. *SIAM J. Optim.*, 12(2), 479-502. https://doi.org/10.1137/S1052623499363220
8. Linderoth, J., Shapiro, A. & Wright, S. (2006). The empirical behavior of sampling methods for stochastic programming. *Annals of Operations Research*, 142(1), 215-241. https://doi.org/10.1007/s10479-006-6169-8
9. Birge, J.R. (1985). Decomposition and partitioning methods for multistage stochastic linear programs. *Operations Research*, 33(5), 989-1007. https://doi.org/10.1287/opre.33.5.989
10. Higle, J.L. & Sen, S. (1991). Stochastic decomposition: An algorithm for two-stage linear programs with recourse. *Mathematics of Operations Research*, 16(3), 650-669. https://doi.org/10.1287/moor.16.3.650
11. Ruszczynski, A. (1986). A regularized decomposition method for minimizing a sum of polyhedral functions. *Mathematical Programming*, 35(3), 309-333. https://doi.org/10.1007/BF01580883
12. Madansky, A. (1960). Inequalities for stochastic linear programming problems. *Management Science*, 6(2), 197-204. https://doi.org/10.1287/mnsc.6.2.197

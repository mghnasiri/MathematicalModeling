# Newsvendor Problem (Single-Period Stochastic Inventory)

## 1. Problem Definition

- **Input:** Unit cost $c$, selling price $p > c$, salvage value $v < c$, random demand $D$
- **Decision:** Order quantity $Q$ before demand is realized
- **Objective:** Minimize expected cost $E[c_o \cdot (Q-D)^+ + c_u \cdot (D-Q)^+]$
  - Overage cost: $c_o = c - v$ (unsold units)
  - Underage cost: $c_u = p - c$ (unmet demand)
- **Classification:** Stochastic optimization. $O(1)$ for continuous distributions; $O(S \log S)$ for discrete scenarios.

### Optimal Solution (Critical Fractile)

$$Q^* = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right) = F^{-1}\!\left(\frac{p - c}{p - v}\right)$$

The critical ratio $\frac{c_u}{c_u + c_o}$ is the optimal service level (also called the
optimal cycle-service level or Type-I service level). Intuitively, if the cost of being
one unit short far exceeds the cost of having one unit left over, the retailer should
stock aggressively; if overage is expensive relative to underage, the retailer should
be conservative.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $c$ | Unit purchase cost |
| $p$ | Selling price ($p > c$) |
| $v$ | Salvage value ($v < c$) |
| $D$ | Random demand (continuous CDF $F$, PDF $f$) |
| $Q$ | Order quantity (decision variable) |
| $c_u = p - c$ | Underage (shortage) cost per unit |
| $c_o = c - v$ | Overage (excess) cost per unit |
| $S$ | Number of discrete demand scenarios |
| $d_k, \pi_k$ | Demand value and probability of scenario $k$ |

### Expected Cost Formulation

$$\min_Q \quad E\bigl[C(Q)\bigr] = \min_Q \quad E\bigl[c_o \cdot (Q - D)^+ + c_u \cdot (D - Q)^+\bigr]$$

where $(x)^+ = \max(0, x)$. Expanding the expectation for a continuous demand distribution
with PDF $f$ and CDF $F$:

$$E[C(Q)] = c_o \int_0^Q (Q - x)\,f(x)\,dx \;+\; c_u \int_Q^\infty (x - Q)\,f(x)\,dx$$

The first integral captures expected overage (units ordered but not sold), while the
second integral captures expected underage (demand that goes unfilled). This is a convex
function of $Q$ because the second derivative is $(c_o + c_u) f(Q) > 0$.

### First-Order Condition and the Critical Fractile

Differentiating $E[C(Q)]$ with respect to $Q$ and setting the derivative to zero:

$$\frac{dE[C(Q)]}{dQ} = c_o \cdot F(Q) - c_u \cdot [1 - F(Q)] = 0$$

Solving for $F(Q)$:

$$F(Q^*) = \frac{c_u}{c_u + c_o} = \frac{p - c}{p - v}$$

This is the **critical fractile** result. The optimal $Q^*$ is the quantile of the
demand distribution at level $c_u / (c_u + c_o)$.

**Service level interpretation:** $Q^*$ is the $(c_u/(c_u+c_o))$-quantile of the demand
distribution. The probability of not stocking out equals the critical ratio. This means
$P(D \le Q^*) = c_u/(c_u + c_o)$, so a higher underage-to-total-cost ratio drives a
higher fill probability.

### Normal Demand Special Case

When demand follows $D \sim N(\mu, \sigma^2)$, the critical fractile yields:

$$Q^* = \mu + z \cdot \sigma, \quad \text{where } z = \Phi^{-1}\!\left(\frac{c_u}{c_u + c_o}\right)$$

Here $\Phi^{-1}$ is the standard normal quantile function. For example, if
$c_u/(c_u + c_o) = 0.75$, then $z \approx 0.674$ and $Q^* = \mu + 0.674\sigma$.
The safety stock is $z\sigma$, and the expected cost at the optimum is
$E[C(Q^*)] = (c_u + c_o) \cdot \sigma \cdot \phi(z)$, where $\phi$ is the standard
normal PDF (the standard loss function result).

### Discrete Scenario Formulation

For $S$ scenarios with demands $d_1, \ldots, d_S$ and probabilities $\pi_1, \ldots, \pi_S$:

$$E[C(Q)] = \sum_{k=1}^{S} \pi_k \bigl[ c_o \cdot (Q - d_k)^+ + c_u \cdot (d_k - Q)^+ \bigr]$$

This piecewise-linear convex function is minimized by sorting scenarios by demand value
and scanning cumulative probability until reaching the critical fractile.

### Multi-Product with Budget Constraint

When $n$ products share a total procurement budget $B$:

$$\max_{Q_1, \ldots, Q_n} \quad \sum_{j=1}^{n} E[\text{profit}_j(Q_j)] \quad \text{s.t.} \quad \sum_{j=1}^{n} c_j \, Q_j \le B, \quad Q_j \ge 0$$

The single-product critical fractile no longer directly applies because of the coupling
constraint. Lagrangian relaxation of the budget constraint yields modified critical
fractiles that depend on the dual variable (shadow price of the budget), but the dual
problem is not closed-form in general. Practical approaches include marginal allocation
and independent-then-scale heuristics (see Section 3).

### Small Illustrative Instance

```
c = 10, p = 25, v = 5
c_u = 15, c_o = 5
Critical ratio = 15/20 = 0.75
If D ~ U[0, 100]: Q* = 0.75 * 100 = 75
If D ~ N(60, 15^2): Q* = 60 + 0.674 * 15 = 70.1
```

---

## 3. Solution Methods

| Method | Type | Complexity | Scope | Description |
|--------|------|-----------|-------|-------------|
| Critical Fractile | Exact | $O(S \log S)$ | Single product | Sort scenarios, scan CDF to critical ratio |
| Grid Search | Exact | $O(S \cdot G)$ | Single product | Brute-force over demand range |
| Marginal Allocation | Heuristic | $O(n \cdot B / \delta)$ | Multi-product | Greedy unit-by-unit allocation under budget |
| Independent + Scale | Heuristic | $O(n \cdot S \log S)$ | Multi-product | Solve independently, scale to fit budget |

### 3.1 Critical Fractile (Discrete Scenarios)

The exact optimal solution for a single-product instance with discrete scenarios.
Implementation: `exact/critical_fractile.py` function `critical_fractile()`.

**Pseudocode:**

```
FUNCTION critical_fractile(demands, probs, c_u, c_o):
    target <- c_u / (c_u + c_o)
    indices <- ARGSORT(demands)             // sort by demand value
    sorted_demands <- demands[indices]
    sorted_probs   <- probs[indices]
    cumulative_prob <- 0
    FOR k = 0, 1, ..., S-1:
        cumulative_prob <- cumulative_prob + sorted_probs[k]
        IF cumulative_prob >= target:
            RETURN sorted_demands[k]
    RETURN sorted_demands[S-1]              // fallback: maximum demand
```

**Key properties:**
- Returns the smallest demand value whose cumulative probability meets or exceeds the
  critical ratio, which is optimal for the piecewise-linear cost function.
- Complexity is dominated by the sort: $O(S \log S)$.
- With weighted (non-uniform) scenario probabilities, the algorithm is identical; only
  the probability accumulation changes.

### 3.2 Grid Search

A brute-force alternative that evaluates $E[C(Q)]$ at $G$ evenly spaced points across
the demand range $[\min(d_k), \max(d_k)]$. Implementation: `exact/critical_fractile.py`
function `grid_search()`.

**Pseudocode:**

```
FUNCTION grid_search(instance, G):
    candidates <- LINSPACE(min_demand, max_demand, G)
    best_Q    <- candidates[0]
    best_cost <- expected_cost(candidates[0])
    FOR each Q in candidates[1:]:
        cost <- expected_cost(Q)
        IF cost < best_cost:
            best_cost <- cost
            best_Q    <- Q
    RETURN best_Q
```

**Key properties:**
- Each cost evaluation is $O(S)$ (dot product over scenarios), so total is $O(S \cdot G)$.
- Useful when the cost function is modified beyond the standard newsvendor (e.g.,
  piecewise costs, fixed ordering costs) and the critical fractile does not apply.
- Accuracy depends on grid resolution; the implementation defaults to $G = 1000$.

### 3.3 Marginal Allocation (Multi-Product)

For budget-constrained multi-product newsvendor. Iteratively allocates one step of
inventory to the product with the highest marginal expected profit per dollar spent.
Implementation: `heuristics/multi_product.py` function `marginal_allocation()`.

**Pseudocode:**

```
FUNCTION marginal_allocation(products, budget, step):
    Q[1..n] <- 0                            // initial order quantities
    budget_remaining <- budget
    WHILE budget_remaining > 0:
        best_product <- NONE
        best_marginal <- -INF
        FOR j = 1, ..., n:
            cost_step <- products[j].unit_cost * step
            IF cost_step > budget_remaining:
                CONTINUE
            new_profit <- expected_profit_j(Q[j] + step)
            old_profit <- expected_profit_j(Q[j])
            marginal   <- (new_profit - old_profit) / cost_step
            IF marginal > best_marginal:
                best_marginal <- marginal
                best_product  <- j
        IF best_product = NONE OR best_marginal <= 0:
            BREAK
        Q[best_product] <- Q[best_product] + step
        budget_remaining <- budget_remaining - products[best_product].unit_cost * step
    RETURN Q[1..n]
```

**Key properties:**
- Greedy approach that approximates the Lagrangian dual of the budget constraint.
- Step size $\delta$ controls the granularity-speed tradeoff; smaller steps yield
  better solutions but more iterations ($O(B / \delta)$ iterations total).
- Each iteration evaluates profit for $n$ products, each costing $O(S)$, so overall
  complexity is $O(n \cdot S \cdot B / \delta)$.
- Stops when no product has a positive marginal return, preventing waste.

### 3.4 Independent then Scale (Multi-Product)

A fast two-phase heuristic: solve each product independently using the critical fractile,
then proportionally scale all quantities down if the total cost exceeds the budget.
Implementation: `heuristics/multi_product.py` function `independent_then_scale()`.

**Pseudocode:**

```
FUNCTION independent_then_scale(products, budget):
    FOR j = 1, ..., n:
        Q[j] <- critical_fractile(products[j])     // unconstrained optimal
    total_cost <- SUM(products[j].unit_cost * Q[j])
    IF total_cost > budget:
        scale <- budget / total_cost
        Q[j]  <- Q[j] * scale   FOR ALL j
    RETURN Q[1..n]
```

**Key properties:**
- Very fast: $O(n \cdot S \log S)$ total (one critical fractile per product).
- Proportional scaling preserves the relative allocation ratios but may be suboptimal
  because it does not account for differing marginal returns across products.
- Provides a quick upper bound or starting solution for more refined methods.

### 3.5 Censored Demand and Bias Correction

In practice, retailers often observe only **sales** $\min(D, Q)$ rather than true demand $D$.
When demand exceeds the stock level, the excess is unobserved (censored). Naive estimation
from sales data underestimates the true mean demand, leading to a downward spiral of
understocking.

**Bias correction approaches:**
- **Kaplan-Meier estimator:** Treat each period as a right-censored observation; use
  survival analysis techniques to reconstruct the demand distribution.
- **EM algorithm:** Alternate between estimating censored demand values (E-step) and
  re-fitting the distribution parameters (M-step). Converges to the MLE under standard
  regularity conditions.
- **Bayesian updating:** Place a prior on the demand distribution parameters, update
  with observed sales (accounting for censoring in the likelihood), and use the posterior
  predictive distribution in the newsvendor formula.

This censored-demand problem is not currently implemented in the repository but is a
natural extension for future work.

---

## 4. Implementations in This Repository

```
newsvendor/
├── instance.py                    # NewsvendorInstance, critical fractile
├── exact/
│   └── critical_fractile.py       # Critical fractile + grid search
├── heuristics/
│   └── multi_product.py           # Marginal allocation, independent+scale
└── tests/
    └── test_newsvendor.py         # 13 tests
```

---

## 5. Extensions and Variants

### 5.1 Multi-Period Newsvendor (Dynamic Inventory)

The single-period model assumes perishable goods with no carryover. When leftover
inventory can be carried to the next period, the problem becomes a **dynamic inventory
control** problem. The optimal policy has a base-stock (order-up-to) structure: at the
start of each period, order up to a target level $S_t$ that depends on remaining
inventory and the demand distribution over the remaining horizon. The single-period
newsvendor is the special case with horizon $T = 1$. For stationary demand and infinite
horizon, the optimal base-stock level converges to a constant that can be computed via
dynamic programming or the critical fractile on the lead-time demand distribution.

Key references: Scarf (1960) established the optimality of $(s, S)$ policies under
fixed ordering costs; Clark and Scarf (1960) extended the analysis to multi-echelon
systems with serial structure.

### 5.2 Risk-Averse Newsvendor (CVaR Objective)

The classical newsvendor minimizes expected cost, which is risk-neutral. A risk-averse
decision-maker may instead minimize a risk measure such as the **Conditional Value at
Risk (CVaR)**:

$$\min_Q \quad \text{CVaR}_\alpha[C(Q)]$$

where $\text{CVaR}_\alpha$ is the expected cost in the worst $\alpha$-fraction of outcomes.
This leads to a modified critical fractile where the effective underage-to-overage ratio
is adjusted by the risk parameter. The CVaR newsvendor orders less than the risk-neutral
optimum because it penalizes the tail of the overage distribution more heavily.

### 5.3 Data-Driven Newsvendor

Traditional approaches assume a known demand distribution, but in practice the
distribution must be estimated from data. The **data-driven newsvendor** directly maps
feature information (e.g., weather, day-of-week, promotions) to order quantities without
explicitly estimating the demand distribution.

Bertsimas and Kallus (2020) proposed a framework where the newsvendor cost is minimized
over a weighted empirical distribution, with weights derived from machine learning
predictions (k-nearest neighbors, kernel methods, random forests, etc.). This approach
integrates prediction and optimization, avoiding the two-step estimate-then-optimize
pipeline that can propagate estimation errors into suboptimal decisions.

### 5.4 Newsvendor with Pricing (Joint Quantity-Price Optimization)

When the retailer can also choose the selling price $p$, demand becomes a function
of price: $D = D(p)$. The joint problem optimizes both price and order quantity:

$$\max_{p, Q} \quad p \cdot E[\min(Q, D(p))] + v \cdot E[(Q - D(p))^+] - c \cdot Q$$

For common demand models (e.g., linear $D(p) = a - bp + \epsilon$ or isoelastic
$D(p) = a \cdot p^{-b} \cdot \epsilon$), the optimal price and quantity can be
characterized through a system of first-order conditions. The key insight is that
the optimal price depends on the order quantity and vice versa, requiring iterative
or joint solution methods.

### 5.5 Minimax (Distributionally Robust) Newsvendor

When the demand distribution is unknown but known to lie in an ambiguity set $\mathcal{P}$,
the minimax newsvendor solves:

$$\min_Q \quad \max_{P \in \mathcal{P}} \quad E_P[C(Q)]$$

Scarf (1958) showed that when only the mean $\mu$ and variance $\sigma^2$ are known
(moment-based ambiguity), the minimax optimal order quantity is:

$$Q^* = \mu + \frac{\sigma}{2}\left(\sqrt{\frac{c_u}{c_o}} - \sqrt{\frac{c_o}{c_u}}\right)$$

This elegant closed-form result does not require knowledge of the full distribution
shape and provides a robust hedge against distributional misspecification.

---

## 6. Computational Notes

### Relationship to Other Problems in This Repository

- **Stochastic Knapsack** (`../stochastic_knapsack/`): Multi-item selection under weight
  uncertainty generalizes the budget-constrained multi-product newsvendor when items have
  stochastic sizes.
- **Two-Stage Stochastic Programming** (`../two_stage_sp/`): The newsvendor is the
  simplest two-stage stochastic program: $Q$ is the first-stage decision, and the
  sales/salvage outcome is the second-stage recourse.
- **Distributionally Robust Optimization** (`../dro/`): Scarf's minimax newsvendor is
  a canonical DRO problem with moment-based ambiguity sets.
- **Safety Stock** (`../../7_inventory_lotsizing/safety_stock/`): The newsvendor critical
  fractile directly determines safety stock levels in continuous-review inventory systems.

### Implementation Details

- **Instance representation** (`instance.py`): The `NewsvendorInstance` dataclass stores
  cost parameters and discrete scenarios. Properties compute `overage_cost`, `underage_cost`,
  and `critical_fractile` on the fly. The `expected_cost()` and `expected_profit()` methods
  use vectorized NumPy operations over all scenarios.
- **Critical fractile solver** (`exact/critical_fractile.py`): Uses `np.argsort` for
  scenario sorting and `np.searchsorted` on cumulative probabilities for efficient
  quantile lookup.
- **Multi-product heuristics** (`heuristics/multi_product.py`): The `MultiProductInstance`
  dataclass wraps a list of `NewsvendorInstance` objects with a shared budget. Marginal
  allocation uses a configurable step size (default 1.0) for the greedy increments.
- **Test suite** (`tests/test_newsvendor.py`): 13 tests across 3 test classes covering
  instance properties, critical fractile correctness (symmetric demand, high/low cost
  ratios, agreement with grid search), and multi-product budget feasibility.

---

## 7. Key References

- Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250-272. https://doi.org/10.2307/1906813
- Scarf, H. (1958). A min-max solution of an inventory problem. In K.J. Arrow, S. Karlin, & H. Scarf (Eds.), *Studies in the Mathematical Theory of Inventory and Production* (pp. 201-209). Stanford University Press.
- Clark, A.J. & Scarf, H. (1960). Optimal policies for a multi-echelon inventory problem. *Management Science*, 6(4), 475-490. https://doi.org/10.1287/mnsc.6.4.475
- Hadley, G. & Whitin, T.M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.
- Porteus, E.L. (2002). *Foundations of Stochastic Inventory Theory*. Stanford University Press.
- Lau, H.-S. & Lau, A.H.-L. (1996). The newsstand problem: A capacitated multiple-product single-period inventory problem. *European Journal of Operational Research*, 94(1), 29-42. https://doi.org/10.1016/0377-2217(95)00192-1
- Petruzzi, N.C. & Dada, M. (1999). Pricing and the newsvendor problem: A review with extensions. *Operations Research*, 47(2), 183-194. https://doi.org/10.1287/opre.47.2.183
- Khouja, M. (1999). The single-period (news-vendor) problem: Literature review and suggestions for future research. *Omega*, 27(5), 537-553. https://doi.org/10.1016/S0305-0483(99)00017-1
- Qin, Y., Wang, R., Vakharia, A.J., Chen, Y. & Seref, M.M.H. (2011). The newsvendor problem: Review and directions for future research. *European Journal of Operational Research*, 213(2), 361-374. https://doi.org/10.1016/j.ejor.2010.11.024
- Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). *Inventory and Production Management in Supply Chains* (4th ed.). CRC Press.
- Bertsimas, D. & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025-1044. https://doi.org/10.1287/mnsc.2018.3253

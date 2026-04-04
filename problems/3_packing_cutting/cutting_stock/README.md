# 1D Cutting Stock Problem (CSP)

## 1. Problem Definition

- **Input:**
  - Stock material of length $L$
  - $m$ item types, each with length $l_i$ and demand $d_i$
- **Decision:** Determine cutting patterns (how to cut each stock roll)
- **Objective:** Minimize the number of stock rolls used
- **Constraints:** Each roll is cut into items totaling $\leq L$. All demands satisfied.
- **Classification:** NP-hard (generalizes bin packing where identical items are interchangeable)

### Relationship to Bin Packing

CSP is a generalization of BPP: when all demands $d_i = 1$, CSP reduces to BPP. The key difference is that CSP has item *types* with demands, enabling pattern-based formulations that don't exist for BPP.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $L$ | Stock roll length | $\mathbb{R}_{>0}$ |
| $m$ | Number of item types | $\mathbb{Z}^+$ |
| $l_i$ | Length of item type $i$ | $(0, L]$ |
| $d_i$ | Demand for item type $i$ | $\mathbb{Z}_{>0}$ |
| $a_{ip}$ | Number of times item $i$ appears in pattern $p$ | $\mathbb{Z}_{\geq 0}$ |
| $x_p$ | Number of times pattern $p$ is used | $\mathbb{Z}_{\geq 0}$ |

### Gilmore-Gomory Formulation (Pattern-Based)

$$\min \sum_{p \in P} x_p \tag{1}$$

$$\sum_{p \in P} a_{ip} \cdot x_p \geq d_i \quad \forall i \quad \text{(demand satisfaction)} \tag{2}$$

$$\sum_{i=1}^{m} l_i \cdot a_{ip} \leq L \quad \forall p \quad \text{(pattern feasibility)} \tag{3}$$

$$x_p \in \mathbb{Z}_{\geq 0} \tag{4}$$

**Column generation:** The number of feasible patterns is exponential. Solve the LP relaxation with column generation: the pricing subproblem is a knapsack problem (find pattern with negative reduced cost).

### Gilmore-Gomory Column Generation Detail

The pattern-based formulation above has an exponentially large set of feasible patterns $P$. Gilmore and Gomory (1961, 1963) introduced column generation to solve the LP relaxation without enumerating all patterns.

**Master Problem (Restricted LP):** At any iteration the master LP uses only a subset $\bar{P} \subseteq P$ of known patterns:

$$\min \sum_{p \in \bar{P}} x_p$$

$$\sum_{p \in \bar{P}} a_{ip} \cdot x_p \geq d_i \quad \forall i = 1, \ldots, m$$

$$x_p \geq 0 \quad \forall p \in \bar{P}$$

Solving this LP yields optimal dual prices $\pi_i \geq 0$ associated with the demand constraints.

**Pricing Subproblem (Bounded Knapsack):** To check whether any pattern outside $\bar{P}$ can improve the LP objective, solve the pricing problem:

$$\max \sum_{i=1}^{m} \pi_i \cdot a_i$$

$$\text{s.t.} \quad \sum_{i=1}^{m} l_i \cdot a_i \leq L$$

$$0 \leq a_i \leq d_i, \quad a_i \in \mathbb{Z}_{\geq 0} \quad \forall i$$

This is a bounded knapsack problem where item values are the current dual prices $\pi_i$, item weights are the lengths $l_i$, and the knapsack capacity is $L$. The upper bound $a_i \leq d_i$ reflects that no pattern needs more copies of item $i$ than its total demand.

- If the optimal pricing objective exceeds 1 (i.e., reduced cost $1 - \sum \pi_i a_i^* < 0$), the new pattern $a^*$ enters the master and we re-optimize.
- If the optimal pricing objective is $\leq 1$, no improving column exists and the current LP solution is optimal over all of $P$.

### LP Relaxation Quality

The LP relaxation of the Gilmore-Gomory formulation is remarkably tight in practice:

- **Integer Round-Up Property (IRUP):** An instance satisfies IRUP if $z^* = \lceil z_{LP}^* \rceil$, i.e., the integer optimum equals the rounded-up LP optimum. A large majority of practical CSP instances satisfy IRUP.
- **Modified Integer Round-Up Property (MIRUP) conjecture:** Scheithauer and Terno (1995) conjectured that for all 1D CSP instances, $z^* \leq \lceil z_{LP}^* \rceil + 1$. This conjecture remains open but has been verified computationally for all known instances. No counterexample has been found.
- **Practical implication:** Column generation solves the LP relaxation to optimality, and simple rounding (e.g., rounding up pattern frequencies and adjusting residual demands) almost always produces a solution within one roll of optimal.

### Continuous Lower Bound

The simplest lower bound ignores the integrality of patterns:

$$LB_1 = \left\lceil \frac{\sum_{i=1}^{m} l_i \cdot d_i}{L} \right\rceil$$

This is the total material demand divided by the stock length, rounded up. It is fast to compute ($O(m)$) and often coincides with the LP optimum for well-structured instances. This is the bound implemented in `instance.py` via `lower_bound()`.

---

## 3. Variants and Related Problems

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| 2D Cutting Stock | `variants/two_dimensional/` | Rectangular items from rectangular sheets |

### Relationship to Other Repository Problems

| Related Problem | Directory | Connection |
|----------------|-----------|------------|
| 1D Bin Packing | `3_packing_cutting/bin_packing/` | CSP with all $d_i = 1$; no pattern aggregation |
| 0-1 Knapsack | `3_packing_cutting/knapsack/` | Pricing subproblem in column generation is a bounded knapsack |

The pricing subproblem in column generation is structurally identical to the bounded knapsack problem (maximize dual-weighted item counts subject to length capacity). This means that improvements to the knapsack solver in `3_packing_cutting/knapsack/` directly benefit CSP column generation performance. Conversely, the FFD-based CSP heuristic reuses bin packing logic from `3_packing_cutting/bin_packing/`.

---

## 4. Complexity and Approximation

### Computational Complexity

The 1D Cutting Stock Problem is **NP-hard** in the strong sense, since it generalizes 1D Bin Packing (set all $d_i = 1$) which is itself strongly NP-hard. No pseudo-polynomial algorithm exists for CSP unless P = NP.

However, the LP relaxation is solvable in polynomial time via column generation (the pricing knapsack is only weakly NP-hard with pseudo-polynomial DP). In practice, column generation converges quickly and the gap to the integer optimum is at most 1 for almost all known instances (MIRUP conjecture).

### Approximation Hierarchy

| Method | Guarantee | Complexity | Notes |
|--------|-----------|------------|-------|
| Continuous LB ($LB_1$) | Lower bound only | $O(m)$ | $\lceil \sum l_i d_i / L \rceil$ |
| LP relaxation (CG) | $z_{LP}^* \leq z^* \leq \lceil z_{LP}^* \rceil + 1$ (MIRUP) | Poly per iter | Tightest practical bound |
| FFD-based | $\frac{11}{9} \cdot \text{OPT} + \frac{6}{9}$ | $O(N^2)$, $N = \sum d_i$ | From bin packing analysis |
| Greedy largest-first | No known tight ratio | $O(R \cdot m)$ | Fast but can be far from optimal |

### Special Cases

- **Divisible item lengths:** When all item lengths divide $L$ (e.g., $l_i \in \{L/2, L/4, L/8\}$), the problem admits a polynomial-time algorithm via matching.
- **Two item types ($m = 2$):** Solvable in $O(1)$ by exhaustive pattern enumeration (at most $O(L/l_{\min})$ feasible patterns).
- **Unit demands ($d_i = 1$):** Reduces to 1D Bin Packing; no pattern aggregation benefit.

---

## 5. Benchmark Instances

- **CUTGEN** generator (Gau & Wascher, 1995): parameterized by $m$, demand range, item length range. Standard parameter settings produce instances with $m \in \{10, 20, 40, 80, 160\}$ item types.
- **Scholl instances:** Derived from bin packing instances by aggregating identical items into type-demand pairs.
- **Waescher instances:** Practical cutting stock instances from the textile and paper industries.

### Implemented Benchmark Instances

This repository provides two built-in instances in `instance.py`:

| Instance | $m$ | $L$ | Lengths | Demands | $LB_1$ | Notes |
|----------|-----|-----|---------|---------|---------|-------|
| `simple3` | 3 | 100 | [45, 36, 31] | [3, 2, 2] | 3 | Greedy uses 4 rolls; optimal is 3 |
| `classic4` | 4 | 10 | [4, 3, 2.5, 2] | [5, 8, 10, 6] | 8 | Classic textbook instance |

Additionally, `CuttingStockInstance.random()` generates random instances parameterized by $m$, stock length, item length range, and demand range.

### Small Illustrative Instance

```
Stock length L = 100
Items: (length, demand) = [(45, 3), (36, 2), (31, 2), (14, 1)]
Greedy: 4 rolls. Optimal: 3 rolls (e.g., [45,45], [45,36,14], [36,31,31])
```

### Instance Difficulty Factors

The difficulty of a CSP instance depends on several structural factors:
- **Number of item types ($m$):** More types increase pricing knapsack difficulty and the number of feasible patterns.
- **Demand magnitude ($d_i$):** Higher demands increase the FFD expansion size $N = \sum d_i$ but may make LP rounding easier (more fractional variables to absorb).
- **Item length distribution ($l_i / L$ ratios):** Instances where items nearly fill a roll (large $l_i / L$) tend to be easier; instances with many small items (small $l_i / L$) create more combinatorial freedom and harder packing subproblems.
- **Waste ratio:** $(R \cdot L - \sum l_i d_i) / (R \cdot L)$ measures how much material is wasted. Low-waste instances (tight packing) are typically harder.

---

## 6. Solution Methods

### 6.1 Constructive Heuristics

#### Greedy Largest-First

For each stock roll, greedily pack items starting with the largest item type that still has remaining demand. Repeat until all demands are satisfied. Each roll is filled independently in a single pass over item types sorted by decreasing length.

```
FUNCTION greedy_largest_first(instance):
    remaining[i] = d_i  for all i
    order = sort item types by l_i descending
    patterns = []

    WHILE any remaining[i] > 0:
        pattern = [0, ..., 0]  (length m)
        space = L

        FOR i in order:
            IF remaining[i] > 0 AND l_i <= space:
                count = min(remaining[i], floor(space / l_i))
                pattern[i] = count
                space -= count * l_i

        patterns.append(pattern, frequency=1)
        remaining -= pattern

    RETURN patterns
```

**Complexity:** $O(R \cdot m)$ where $R$ is the number of rolls used ($R \leq \sum d_i$).

#### FFD-Based Approach

Expand item type demands into individual items, apply First Fit Decreasing bin packing, then aggregate identical bins back into patterns with multiplicities.

```
FUNCTION ffd_based(instance):
    items = []
    FOR i = 1 to m:
        append i to items  d_i  times

    sort items by l[item_type] descending

    bins = []
    FOR each item in items:
        FOR each bin b in bins:
            IF remaining_space(b) >= l[item_type]:
                add item to bin b; BREAK
        IF not placed:
            open new bin with item

    aggregate identical bins into (pattern, frequency) pairs
    RETURN aggregated patterns
```

**Complexity:** $O(N^2)$ where $N = \sum d_i$ is the total number of individual items. The aggregation step merges identical bins in $O(N \log N)$ via sorting or hashing. FFD provides an $\frac{11}{9} \cdot \text{OPT} + \frac{6}{9}$ approximation guarantee inherited from bin packing.

### 6.2 Exact Methods

#### Column Generation (Gilmore-Gomory)

The column generation procedure (Gilmore & Gomory, 1961, 1963) solves the LP relaxation to optimality without enumerating all feasible patterns. The procedure alternates between solving a restricted master LP and a pricing subproblem.

```
FUNCTION column_generation(instance):
    # Initialize with trivial patterns (one item type per pattern)
    P_bar = {e_i : l_i <= L}  (unit vectors)

    WHILE true:
        Solve restricted master LP on P_bar
            -> optimal x*, dual prices pi*

        Solve pricing knapsack:
            max  SUM pi_i * a_i
            s.t. SUM l_i * a_i <= L
                 0 <= a_i <= d_i,  a_i integer

        IF pricing_objective <= 1:
            BREAK  (LP optimal, no improving column)

        new_pattern = a*  (from knapsack solution)
        Add new_pattern to P_bar

    RETURN x*  (LP-optimal fractional solution)
```

**LP-to-Integer Rounding:** The LP solution $x^*$ is typically fractional. Common strategies:
- **Simple rounding:** Round each $x_p^*$ down. Solve a residual CSP (demands minus already-covered items) greedily. By MIRUP, this usually adds at most 1 extra roll.
- **Dive-and-fix:** Iteratively fix the fractional variable closest to an integer, re-solve.

**Complexity per iteration:** Solving the master LP is polynomial (simplex or interior point). The pricing knapsack is NP-hard in general but pseudo-polynomial via DP in $O(m \cdot L)$ when $L$ is integral. In practice the number of column generation iterations is small (often $O(m)$ to $O(m^2)$).

#### Branch-and-Price

Branch-and-price embeds column generation inside a branch-and-bound framework to obtain provably optimal integer solutions. The key challenge is defining branching decisions compatible with column generation.

**Standard branching** on individual $x_p$ (use pattern $p$ or not) is inefficient because it creates an exponential branching tree. Instead, Vanderbeck (1999) and Vance et al. (1994) proposed:

- **Arc-flow branching:** Branch on the aggregate flow of items, i.e., $\sum_{p: a_{ip} \geq k} x_p \leq n$ or $\geq n+1$ for item type $i$ at multiplicity $k$. This partitions solutions without destroying the pricing subproblem structure.
- **Ryan-Foster branching** (for set-partitioning models): Branch on pairs of items that must be in the same or different patterns.

At each branch-and-bound node, column generation re-solves the LP relaxation with the added branching constraints. New columns generated at child nodes respect the branching decisions.

**Performance:** Branch-and-price can solve instances with hundreds of item types to proven optimality. The LP bound from column generation is typically very tight (MIRUP property), so the branch-and-bound tree is usually small.

### 6.3 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Pattern swap/merge neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | Pattern modification moves |
| 3 | Tabu Search (TS) | Trajectory | Pattern-level tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Destroy patterns + reconstruct |
| 5 | Genetic Algorithm (GA) | Population | Pattern-set encoding |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Swap → merge → split patterns |

---

## 7. Implementations in This Repository

```
cutting_stock/
├── instance.py                    # CuttingStockInstance, pattern validation
├── heuristics/
│   └── greedy_csp.py              # Greedy largest-first, FFD-based
├── metaheuristics/
│   ├── local_search.py            # Pattern swap/merge
│   ├── simulated_annealing.py     # SA
│   ├── tabu_search.py             # TS
│   ├── iterated_greedy.py         # IG
│   ├── genetic_algorithm.py       # GA: pattern-set encoding
│   └── vns.py                     # VNS
├── variants/
│   └── two_dimensional/           # 2D CSP
└── tests/                         # 7 test files
    ├── test_cutting_stock.py      # Core algorithms
    ├── test_csp_ga.py             # GA
    ├── test_csp_sa.py             # SA
    ├── test_csp_ts.py             # TS
    ├── test_csp_ig.py             # IG
    ├── test_csp_ls.py             # LS
    └── test_csp_vns.py            # VNS
```

**Total:** 2 heuristics (1 file), 6 metaheuristics/LS, 1 variant, 7 test files.

### Implementation Notes

**Data structures (`instance.py`):**
- `CuttingStockInstance` holds $m$ (number of item types), `stock_length` ($L$), `lengths` (numpy array of $l_i$), and `demands` (numpy array of $d_i$). Validation in `__post_init__` checks positivity, dimensionality, and that no item exceeds the stock length.
- `CuttingPattern` stores a count vector (how many of each item type appears in one pattern).
- `CuttingStockSolution` stores a list of `(pattern_counts, frequency)` tuples and the total roll count. This pattern-frequency representation is compact when many rolls share the same cutting pattern.
- `validate_solution()` checks pattern feasibility (total length per pattern $\leq L$), non-negative counts, and demand satisfaction ($\sum_p a_{ip} \cdot \text{freq}_p \geq d_i$).
- `lower_bound()` computes the continuous lower bound $\lceil \sum l_i d_i / L \rceil$.
- `CuttingStockInstance.random()` generates random instances with configurable $m$, stock length, item length range, and demand range; accepts a `seed` parameter for reproducibility.

**Heuristic algorithms (`heuristics/greedy_csp.py`):**
- `greedy_largest_first()` sorts item types by length (descending) once, then iteratively fills rolls by greedily packing the largest remaining items. Each roll produces one pattern with frequency 1.
- `ffd_based()` expands type-demand pairs into $N = \sum d_i$ individual items, applies FFD bin packing, then aggregates identical bins back into pattern-frequency pairs via a dictionary keyed on tuple-converted pattern arrays.
- Both functions use `importlib.util` for explicit file-path imports to avoid name collisions with parent `instance.py` modules (standard pattern in this repository).

**Testing (`tests/test_cutting_stock.py`):**
- 21 tests across 6 test classes covering: instance creation and validation, lower bound correctness, greedy and FFD heuristic quality, solution validation, edge cases (single item type, single demand), and benchmark instance consistency.

---

## 8. Key References

### Foundational Papers

- Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach to the cutting stock problem. *Operations Research*, 9(6), 849-859. https://doi.org/10.1287/opre.9.6.849 -- Introduced column generation for CSP; the master LP plus knapsack pricing subproblem framework.
- Gilmore, P.C. & Gomory, R.E. (1963). A linear programming approach to the cutting stock problem -- Part II. *Operations Research*, 11(6), 863-888. https://doi.org/10.1287/opre.11.6.863 -- Extended to multi-dimensional and multi-stage cutting problems.

### LP Relaxation and MIRUP

- Scheithauer, G. & Terno, J. (1995). The modified integer round-up property of the one-dimensional cutting stock problem. *European Journal of Operational Research*, 84(3), 562-571. https://doi.org/10.1016/0377-2217(95)00022-I -- Conjectured that $z^* \leq \lceil z_{LP}^* \rceil + 1$ for all 1D CSP instances (MIRUP).
- Belov, G. & Scheithauer, G. (2006). A branch-and-cut-and-price algorithm for one-dimensional stock cutting and two-dimensional two-stage cutting. *European Journal of Operational Research*, 171(1), 85-106. https://doi.org/10.1016/j.ejor.2004.08.036 -- State-of-the-art exact method combining column generation, cutting planes, and branching.

### Branch-and-Price and Decomposition

- Vance, P.H., Barnhart, C., Johnson, E.L. & Nemhauser, G.L. (1994). Solving binary cutting stock problems by column generation and branch-and-bound. *Computational Optimization and Applications*, 3(2), 111-130. https://doi.org/10.1007/BF01300970 -- Branch-and-price with Ryan-Foster branching for binary CSP variants.
- Vanderbeck, F. (1999). Computational study of a column generation algorithm for bin packing and cutting stock problems. *Mathematical Programming*, 86(3), 565-594. https://doi.org/10.1007/s101070050105 -- Comprehensive computational study of column generation strategies and branching rules.
- Vanderbeck, F. (2000). On Dantzig-Wolfe decomposition in integer programming and ways to perform branching in a branch-and-price algorithm. *Operations Research*, 48(1), 111-128. https://doi.org/10.1287/opre.48.1.111.12453 -- General framework for branching in decomposition-based integer programming.

### Surveys and Classification

- Wascher, G., Haussner, H. & Schumann, H. (2007). An improved typology of cutting and packing problems. *European Journal of Operational Research*, 183(3), 1109-1130. https://doi.org/10.1016/j.ejor.2005.12.047 -- Comprehensive taxonomy of cutting and packing problem variants.
- Delorme, M., Iori, M. & Martello, S. (2016). Bin packing and cutting stock problems: Mathematical models and exact algorithms. *European Journal of Operational Research*, 255(1), 1-20. https://doi.org/10.1016/j.ejor.2016.04.030 -- Survey of exact methods for BPP and CSP including arc-flow and reflect formulations.

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

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| 2D Cutting Stock | `variants/two_dimensional/` | Rectangular items from rectangular sheets |

---

## 4. Benchmark Instances

- **CUTGEN** generator (Gau & Wascher, 1995): parameterized by $m$, demand range, item length range
- **Scholl instances:** Derived from bin packing instances

### Small Illustrative Instance

```
Stock length L = 100
Items: (length, demand) = [(45, 3), (36, 2), (31, 2), (14, 1)]
Greedy: 4 rolls. Optimal: 3 rolls (e.g., [45,45], [45,36,14], [36,31,31])
```

---

## 5. Solution Methods

### 5.1 Constructive Heuristics

- **Greedy largest-first:** Fill each roll starting with the largest remaining item
- **FFD-based:** Expand demands to individual items, apply FFD, aggregate into patterns

### 5.2 Exact Methods

**Column generation** (Gilmore & Gomory, 1961): Solve the LP relaxation iteratively. At each iteration, solve a knapsack pricing problem to find the most promising new pattern. Round the LP solution for an integer solution.

### 5.3 Metaheuristics

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

## 6. Implementations in This Repository

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

---

## 7. Key References

- Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach to the cutting stock problem. *Operations Research*, 9(6), 849-859.
- Gilmore, P.C. & Gomory, R.E. (1963). A linear programming approach to the cutting stock problem — Part II. *Operations Research*, 11(6), 863-888.
- Vance, P.H., Barnhart, C., Johnson, E.L. & Nemhauser, G.L. (1994). Solving binary cutting stock problems by column generation and branch-and-bound. *Computational Optimization and Applications*, 3(2), 111-130.
- Wascher, G., Haussner, H. & Schumann, H. (2007). An improved typology of cutting and packing problems. *European Journal of Operational Research*, 183(3), 1109-1130.

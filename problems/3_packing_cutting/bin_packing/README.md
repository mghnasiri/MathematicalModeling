# 1D Bin Packing Problem (BPP)

## 1. Problem Definition

- **Input:** $n$ items with sizes $s_i \in (0, C]$ and bins of capacity $C$
- **Decision:** Assign each item to a bin
- **Objective:** Minimize the number of bins used
- **Constraints:** Sum of item sizes in each bin $\leq C$
- **Classification:** Strongly NP-hard

### Complexity

- NP-hard in the strong sense (no pseudo-polynomial DP unless P = NP)
- No asymptotic PTAS exists unless P = NP
- FFD achieves $\frac{11}{9} \text{OPT} + \frac{6}{9}$ (Johnson, 1973; Dosa, 2007 tight bound)
- Lower bounds: $L_1 = \lceil \sum s_i / C \rceil$, $L_2$ considers pairs of large items

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of items | $\mathbb{Z}^+$ |
| $C$ | Bin capacity | $\mathbb{R}_{>0}$ |
| $s_i$ | Size of item $i$ | $(0, C]$ |
| $x_{ij}$ | 1 if item $i$ in bin $j$ | $\{0, 1\}$ |
| $y_j$ | 1 if bin $j$ is used | $\{0, 1\}$ |

### MILP Formulation

$$\min \sum_{j=1}^{n} y_j \tag{1}$$

$$\sum_{j=1}^{n} x_{ij} = 1 \quad \forall i \quad \text{(each item in one bin)} \tag{2}$$

$$\sum_{i=1}^{n} s_i \cdot x_{ij} \leq C \cdot y_j \quad \forall j \quad \text{(capacity)} \tag{3}$$

$$x_{ij} \in \{0,1\},\; y_j \in \{0,1\} \tag{4}$$

Symmetry: bins are interchangeable. Add $y_1 \geq y_2 \geq \ldots$ to break symmetry.

### Set-Cover / Column-Generation Reformulation

An alternative formulation views BPP through the lens of set covering. Let
$\mathcal{P}$ denote the set of all feasible packing patterns, where each
pattern $p \in \mathcal{P}$ is a subset of items whose total size does not
exceed $C$. Define binary variable $z_p = 1$ if pattern $p$ is selected:

$$\min \sum_{p \in \mathcal{P}} z_p \tag{5}$$

$$\sum_{p \ni i} z_p \geq 1 \quad \forall i \in \{1, \ldots, n\} \quad \text{(each item covered)} \tag{6}$$

$$z_p \in \{0, 1\} \quad \forall p \in \mathcal{P} \tag{7}$$

The number of feasible patterns $|\mathcal{P}|$ is exponential, so the LP
relaxation of (5)--(7) is solved via **column generation** (Gilmore &
Gomory, 1961). At each iteration the pricing sub-problem is a 0-1 knapsack:
find a pattern with negative reduced cost. This approach connects BPP
directly to the **Cutting Stock Problem** (CSP), where identical items are
grouped by type and demands replace individual item constraints.

The LP relaxation value provides a strong lower bound, often tighter than
$L_2$, and the integrality gap is conjectured to be at most 1 for all BPP
instances (the Modified Integer Round-Up Property, or MIRUP conjecture).

### Lower Bounds

#### L1: Continuous Lower Bound

The simplest lower bound divides total item volume by bin capacity:

$$L_1 = \left\lceil \frac{\sum_{i=1}^{n} s_i}{C} \right\rceil$$

**Implemented in:** `instance.py` as `BinPackingInstance.lower_bound_l1()`.

This bound is tight when items pack perfectly but can be arbitrarily weak
when many items are slightly larger than $C/2$ (forcing one item per bin).

#### L2: Martello--Toth Lower Bound

A stronger bound that accounts for items that cannot share a bin (Martello
& Toth, 1990). For a threshold $\alpha \in (0, C/2]$, define:

- $N_1(\alpha) = \{i : s_i > C - \alpha\}$ -- items too large to pair with any item of size $> \alpha$
- $N_2(\alpha) = \{i : C - \alpha \geq s_i > C/2\}$ -- large items that may pair only with small items
- $N_3(\alpha) = \{i : \alpha \geq s_i \geq C/2 \text{ is false, i.e., } s_i \leq \alpha\}$ -- small items candidate for residual packing

Each item in $N_1$ needs its own bin. Items in $N_2$ each occupy a bin but
leave residual space. The L2 bound counts whether the small items ($N_3$)
exceed this residual:

$$L_2(\alpha) = |N_1| + |N_2| + \max\left(0,\; \left\lceil \frac{\sum_{i \in N_3} s_i - (|N_2| \cdot C - \sum_{i \in N_2} s_i)}{C} \right\rceil \right)$$

The overall bound is $L_2 = \max_\alpha L_2(\alpha) \geq L_1$.

**Implemented in:** `instance.py` as `BinPackingInstance.lower_bound_l2()`, using the
simplified single-threshold version with $\alpha = C/2$.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Online BPP | `variants/online/` | Items arrive one by one; irrevocable |
| 2D Bin Packing | `variants/two_dimensional/` | Rectangular items, 2D bins |
| Variable-Size BPP | `variants/variable_size/` | Bins have different sizes and costs |

---

## 4. Benchmark Instances

- **Scholl & Klein instances:** Triplet, uniform, hard instances (Scholl et al., 1997)
- **Falkenauer instances:** Uniform and triplet (Falkenauer, 1996)

### Small Illustrative Instance

```
6 items, capacity C = 10
Sizes: [7, 5, 4, 3, 3, 2]
FFD packing: Bin1=[7,3], Bin2=[5,4], Bin3=[3,2] → 3 bins
Optimal: 3 bins (L1 = ⌈24/10⌉ = 3)
```

---

## 5. Solution Methods

### 5.1 Constructive Heuristics

| Method | Complexity | Approximation |
|--------|-----------|--------------|
| Next Fit (NF) | $O(n)$ | $\leq 2 \cdot \text{OPT}$ |
| First Fit (FF) | $O(n^2)$ | $\leq 1.7 \cdot \text{OPT} + 1$ |
| First Fit Decreasing (FFD) | $O(n \log n)$ | $\frac{11}{9} \text{OPT} + \frac{6}{9}$ |
| Best Fit Decreasing (BFD) | $O(n \log n)$ | Same as FFD asymptotically |

#### Competitive Ratios and Online/Offline Classification

| Algorithm | Asymptotic Ratio | Absolute Bound | Online? | Sorting Required? |
|-----------|-----------------|----------------|---------|-------------------|
| NF | 2 | $2 \cdot \text{OPT}$ | Yes | No |
| FF | 1.7 | $\lfloor 1.7 \cdot \text{OPT} \rfloor + 1$ | Yes | No |
| FFD | 11/9 $\approx$ 1.222 | $\frac{11}{9} \text{OPT} + \frac{6}{9}$ | No | Yes (decreasing) |
| BFD | 11/9 $\approx$ 1.222 | $\frac{11}{9} \text{OPT} + \frac{6}{9}$ | No | Yes (decreasing) |

Online algorithms must irrevocably assign each item as it arrives. Offline
algorithms (FFD, BFD) see all items upfront and sort them first.

#### Next Fit (NF)

The simplest bin packing heuristic. Maintain a single "current bin." If the
next item fits, place it; otherwise, close the current bin and open a new one.
Items are never placed in a previously closed bin.

```
ALGORITHM NF(sizes, C)
  current_bin ← new bin
  FOR each item i (in arrival order):
    IF remaining_capacity(current_bin) >= s_i:
      Place i in current_bin
    ELSE:
      Close current_bin
      current_bin ← new bin
      Place i in current_bin
  RETURN all bins
```

**Complexity:** $O(n)$ -- single pass, constant work per item.

**Guarantee:** $\text{NF}(I) \leq 2 \cdot \text{OPT}(I)$. Proof sketch:
consider any two consecutive bins $B_k, B_{k+1}$. Their combined load
exceeds $C$ (otherwise item in $B_{k+1}$ would have fit in $B_k$), so the
average load per bin exceeds $C/2$. Hence NF uses at most $2 \cdot \lceil \sum s_i / C \rceil$ bins.
The ratio of 2 is tight: an adversarial sequence alternating items of size
$C/2 + \epsilon$ and $C/2 - \epsilon$ forces NF to use twice the optimal.

#### First Fit Decreasing (FFD) -- Pseudocode and Guarantee

```
ALGORITHM FFD(sizes, C)
  Sort items by decreasing size: s_{π(1)} ≥ s_{π(2)} ≥ ... ≥ s_{π(n)}
  bins ← []
  FOR each item π(i) in sorted order:
    placed ← FALSE
    FOR each bin b in bins:
      IF remaining_capacity(b) ≥ s_{π(i)}:
        Place π(i) in b; placed ← TRUE; BREAK
    IF not placed:
      Open new bin, place π(i)
  RETURN bins
```

**Complexity:** $O(n \log n)$ for sorting plus $O(n \cdot B)$ for placement,
where $B$ is the number of bins opened. In the worst case $B = O(n)$,
giving $O(n^2)$ total.

**Guarantee:** $\text{FFD}(I) \leq \frac{11}{9} \text{OPT}(I) + \frac{6}{9}$
(Johnson, 1973; tight bound by Dosa, 2007).

**Proof sketch (Johnson, 1973):** Partition items into "large" ($s_i > C/3$)
and "small" ($s_i \leq C/3$) classes. After sorting, FFD first places all
large items. Since each large item exceeds $C/3$, at most 2 large items fit
per bin. The key insight is that bins with two large items are nearly full
(combined size $> 2C/3$), leaving at most $C/3$ residual. When small items
are placed, they fill residual capacity efficiently. A careful counting
argument across six item-size classes ($> C/2$, $(C/3, C/2]$, etc.) yields
the 11/9 ratio. The additive constant 6/9 arises from boundary cases with
fewer than 6 items.

**Implemented in:** `heuristics/first_fit.py` as `first_fit_decreasing()`.

#### Best Fit Decreasing (BFD) -- Pseudocode

```
ALGORITHM BFD(sizes, C)
  Sort items by decreasing size: s_{π(1)} ≥ s_{π(2)} ≥ ... ≥ s_{π(n)}
  bins ← []
  FOR each item π(i) in sorted order:
    best_bin ← NULL
    best_remaining ← ∞
    FOR each bin b in bins:
      IF remaining_capacity(b) ≥ s_{π(i)} AND remaining_capacity(b) < best_remaining:
        best_bin ← b
        best_remaining ← remaining_capacity(b)
    IF best_bin ≠ NULL:
      Place π(i) in best_bin
    ELSE:
      Open new bin, place π(i)
  RETURN bins
```

**Complexity:** Same as FFD: $O(n \log n + n \cdot B)$.

BFD differs from FFD only in bin selection: it picks the bin with the
**least** remaining capacity that still accommodates the item (tightest fit),
whereas FFD picks the **first** feasible bin. Both achieve the same
worst-case asymptotic ratio of 11/9, but BFD tends to leave fewer partially
filled bins in practice, sometimes producing slightly better solutions.

**Implemented in:** `heuristics/first_fit.py` as `best_fit_decreasing()`.

### 5.2 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Item swap/move between bins |
| 2 | Simulated Annealing (SA) | Trajectory | Swap/move with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Item-bin pair tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Empty a bin + repack via FFD |
| 5 | Genetic Algorithm (GA) | Population | Permutation encoding + FF decoder |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Move, swap, merge neighborhoods |

#### Genetic Algorithm -- Encoding and Parameters

The GA in this repository follows the **permutation-based indirect encoding**
approach. Rather than encoding bin assignments directly (which leads to
massive redundancy from bin interchangeability), each chromosome is a
permutation of item indices. A deterministic **First Fit decoder** converts
the permutation into a bin assignment: items are placed in permutation order
using the FF rule.

This encoding eliminates symmetry issues inherent in direct bin-assignment
representations and ensures every chromosome decodes to a feasible solution.

| Component | Implementation Detail |
|-----------|----------------------|
| **Encoding** | Permutation of $\{0, 1, \ldots, n-1\}$ (item indices) |
| **Decoder** | First Fit: place items in permutation order into first feasible bin |
| **Initialization** | 1 FFD-seeded individual (sorted-decreasing) + random permutations |
| **Crossover** | Order Crossover (OX): preserves relative order from both parents |
| **Mutation** | Swap: exchange two random positions in the permutation |
| **Selection** | Tournament selection (minimize bin count) |
| **Elitism** | Best individual always survives to next generation |
| **Fitness** | Number of bins in the decoded solution (minimize) |

**Default parameter values** (from `metaheuristics/genetic_algorithm.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 50 | Population size |
| `generations` | 200 | Number of generations |
| `mutation_rate` | 0.15 | Probability of swap mutation per offspring |
| `tournament_size` | 3 | Individuals per tournament round |
| `seed` | `None` | Random seed for reproducibility |

The FFD-seeded individual provides a strong initial solution (within 11/9 of
optimal), ensuring the GA starts from a competitive baseline. Random
individuals provide diversity for exploration.

**Implemented in:** `metaheuristics/genetic_algorithm.py`.

---

## 6. Implementations in This Repository

```
bin_packing/
├── instance.py                    # BinPackingInstance, L1/L2 lower bounds
├── heuristics/
│   └── first_fit.py               # FF, FFD (11/9 approx), BFD
├── metaheuristics/
│   ├── local_search.py            # Item swap/move neighborhoods
│   ├── simulated_annealing.py     # SA
│   ├── tabu_search.py             # TS
│   ├── iterated_greedy.py         # IG: empty bin + FFD repack
│   ├── genetic_algorithm.py       # GA: permutation + FF decoder
│   └── vns.py                     # VNS: move → swap → merge
├── variants/
│   ├── online/                    # Online BPP
│   ├── two_dimensional/           # 2D BPP
│   └── variable_size/             # Variable-size bins
└── tests/                         # 6 test files
    ├── test_bin_packing.py        # Core algorithms
    ├── test_bpp_sa.py             # SA
    ├── test_bpp_ts.py             # TS
    ├── test_bpp_ig.py             # IG
    ├── test_bpp_ls.py             # LS
    └── test_bpp_vns.py            # VNS
```

**Total:** 3 heuristics (1 file), 6 metaheuristics/LS, 3 variants, 6 test files.

---

## 7. Key References

### Foundational Works

- Johnson, D.S. (1973). *Near-optimal bin packing algorithms*. PhD thesis, MIT, Cambridge, MA. -- Established the 11/9 asymptotic ratio for FFD and introduced the item-class partitioning proof technique.
- Johnson, D.S., Demers, A., Ullman, J.D., Garey, M.R. & Graham, R.L. (1974). Worst-case performance bounds for simple one-dimensional packing algorithms. *SIAM Journal on Computing*, 3(4), 299--325. [doi:10.1137/0203025](https://doi.org/10.1137/0203025) -- Proved FF uses at most $\lfloor 1.7 \cdot \text{OPT} \rfloor + 1$ bins and analyzed NF, BF, FFD, BFD.
- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman, New York. -- Established strong NP-hardness of BPP (no pseudo-polynomial algorithm unless P = NP).

### Lower Bounds and Exact Methods

- Martello, S. & Toth, P. (1990). Lower bounds and reduction procedures for the bin packing problem. *Discrete Applied Mathematics*, 28(1), 59--70. [doi:10.1016/0166-218X(90)90094-S](https://doi.org/10.1016/0166-218X(90)90094-S) -- Introduced the L2 lower bound and dominance-based reduction procedures.
- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley, Chichester. -- Comprehensive reference for BPP, covering exact branch-and-bound with L2 bounds, FFD, and reduction procedures.
- Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach to the cutting-stock problem. *Operations Research*, 9(6), 849--859. [doi:10.1287/opre.9.6.849](https://doi.org/10.1287/opre.9.6.849) -- Introduced column generation for the cutting stock / bin packing LP relaxation.

### Approximation and Tight Bounds

- Dosa, G. (2007). The tight bound of first fit decreasing bin-packing algorithm is FFD(I) $\leq$ 11/9 OPT(I) + 6/9. *ESCAPE*, LNCS 4614, 1--11. [doi:10.1007/978-3-540-74450-4_1](https://doi.org/10.1007/978-3-540-74450-4_1) -- Proved the additive constant 6/9 is tight for FFD.

### Surveys

- Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation algorithms for bin packing: A survey. In D.S. Hochbaum (ed.), *Approximation Algorithms for NP-Hard Problems*, PWS Publishing, 46--93. -- Comprehensive survey covering online/offline heuristics, lower bounds, and average-case analysis.
- Coffman, E.G., Csirik, J., Galambos, G., Martello, S. & Vigo, D. (2013). Bin packing approximation algorithms: Survey and classification. In P.M. Pardalos, D.-Z. Du & R.L. Graham (eds.), *Handbook of Combinatorial Optimization*, Springer, 455--531. [doi:10.1007/978-1-4419-7997-1_35](https://doi.org/10.1007/978-1-4419-7997-1_35) -- Updated survey with post-2000 results on online algorithms and stochastic variants.

### Metaheuristics for BPP

- Falkenauer, E. (1996). A hybrid grouping genetic algorithm for bin packing. *Journal of Heuristics*, 2(1), 5--30. [doi:10.1007/BF00226291](https://doi.org/10.1007/BF00226291) -- Introduced grouping GA encoding that operates on bins rather than items.
- Quiroz-Castellanos, M., Cruz-Reyes, L., Torres-Jimenez, J., Gomez-Santillan, C., Fraire-Huacuja, H.J. & Alvim, A.C.F. (2015). A grouping genetic algorithm with controlled gene transmission for the bin packing problem. *Computers & Operations Research*, 55, 52--64. [doi:10.1016/j.cor.2014.10.010](https://doi.org/10.1016/j.cor.2014.10.010) -- Advanced grouping GA with controlled inheritance mechanisms.

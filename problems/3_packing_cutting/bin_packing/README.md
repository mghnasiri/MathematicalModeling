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
| First Fit (FF) | $O(n^2)$ | $\leq 1.7 \text{OPT} + 1$ |
| First Fit Decreasing (FFD) | $O(n \log n)$ | $\frac{11}{9} \text{OPT} + \frac{6}{9}$ |
| Best Fit Decreasing (BFD) | $O(n \log n)$ | Same as FFD asymptotically |

```
ALGORITHM FFD(sizes, C)
  Sort items by decreasing size
  bins ← []
  FOR each item i:
    placed ← FALSE
    FOR each bin b in bins:
      IF remaining_capacity(b) ≥ s_i:
        Place i in b; placed ← TRUE; BREAK
    IF not placed:
      Open new bin, place i
  RETURN bins
```

### 5.2 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Item swap/move between bins |
| 2 | Simulated Annealing (SA) | Trajectory | Swap/move with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Item-bin pair tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Empty a bin + repack via FFD |
| 5 | Genetic Algorithm (GA) | Population | Permutation encoding + FF decoder |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Move → swap → merge bins |

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

- Johnson, D.S. (1973). *Near-optimal bin packing algorithms*. PhD thesis, MIT.
- Coffman, E.G., Garey, M.R. & Johnson, D.S. (1996). Approximation algorithms for bin packing. In *Approximation Algorithms for NP-Hard Problems*, PWS.
- Dosa, G. (2007). The tight bound of first fit decreasing bin-packing algorithm is FFD(I) ≤ 11/9 OPT(I) + 6/9. *ESCAPE*, LNCS 4614, 1-11.
- Martello, S. & Toth, P. (1990). *Knapsack Problems*. Wiley.

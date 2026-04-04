# Uncapacitated Facility Location Problem (UFLP)

## 1. Problem Definition

- **Input:**
  - $m$ potential facility sites with opening costs $f_i$
  - $n$ customers with assignment costs $c_{ij}$ (cost to serve customer $j$ from facility $i$)
- **Decision:** Select facilities to open ($y_i \in \{0,1\}$) and assign each customer to an open facility ($x_{ij} \in \{0,1\}$)
- **Objective:** Minimize total fixed + assignment cost: $\sum_i f_i y_i + \sum_{i,j} c_{ij} x_{ij}$
- **Constraints:** Each customer assigned to exactly one open facility
- **Classification:** NP-hard. Best known approximation: 1.488 (Li, 2013)

### Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| UFLP | NP-hard | Cornuejols et al. (1977) |
| Capacitated FL | Strongly NP-hard | — |
| Best approximation | 1.488 | Li (2013) |
| Greedy | 1.61 | Jain et al. (2003) |

---

## 2. Mathematical Formulation

$$\min \sum_{i=1}^{m} f_i y_i + \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j \quad \text{(each customer assigned)} \tag{2}$$

$$x_{ij} \leq y_i \quad \forall i, j \quad \text{(assign only to open facilities)} \tag{3}$$

$$y_i \in \{0,1\},\; x_{ij} \in \{0,1\} \tag{4}$$

LP relaxation: replace (4) with $0 \leq x, y \leq 1$. The LP relaxation of UFLP has half-integrality (Nemhauser & Wolsey), and rounding-based approximation algorithms exploit this.

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| Capacitated FL | `variants/capacitated/` | Each facility has a capacity limit |

---

## 4. Solution Methods

### 4.1 Constructive Heuristics

- **Greedy Add:** Start with no facilities open. Iteratively open the facility giving the largest cost reduction. $O(m \cdot n)$ per iteration.
- **Greedy Drop:** Start with all facilities open. Iteratively close the facility whose removal increases cost the least. $O(m \cdot n)$ per iteration.

### 4.2 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Open/close/swap facility moves |
| 2 | Simulated Annealing (SA) | Trajectory | Toggle/swap with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Facility-level tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Close facilities + greedy reopen |
| 5 | Genetic Algorithm (GA) | Population | Binary encoding (open/closed) |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Toggle → swap → multi-swap |

---

## 5. Implementations in This Repository

```
facility_location/
├── instance.py                    # FacilityLocationInstance, validation
├── heuristics/
│   └── greedy_facility.py         # Greedy add, greedy drop
├── metaheuristics/
│   ├── local_search.py            # Open/close/swap
│   ├── simulated_annealing.py     # SA: toggle/swap
│   ├── tabu_search.py             # TS: facility tabu
│   ├── iterated_greedy.py         # IG
│   ├── genetic_algorithm.py       # GA: binary encoding
│   └── vns.py                     # VNS
├── variants/
│   └── capacitated/               # CFLP
└── tests/                         # 6 test files
    ├── test_facility_location.py
    ├── test_fl_sa.py, test_fl_ts.py, test_fl_ig.py
    ├── test_fl_ls.py, test_fl_vns.py
```

---

## 6. Key References

- Cornuejols, G., Fisher, M.L. & Nemhauser, G.L. (1977). Location of bank accounts to optimize float. *Management Science*, 23(8), 789-810.
- Jain, K., Mahdian, M. & Saberi, A. (2003). A new greedy approach for facility location problems. *STOC*, 731-740.
- Li, S. (2013). A 1.488 approximation algorithm for the uncapacitated facility location problem. *Information and Computation*, 222, 45-58.

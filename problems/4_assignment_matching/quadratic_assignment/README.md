# Quadratic Assignment Problem (QAP)

## 1. Problem Definition

- **Input:** $n$ facilities, $n$ locations, flow matrix $F$ ($n \times n$), distance matrix $D$ ($n \times n$)
- **Decision:** Permutation $\pi$ assigning facilities to locations
- **Objective:** Minimize $\sum_{i,j} F_{ij} \cdot D_{\pi(i), \pi(j)}$
- **Classification:** NP-hard. One of the hardest combinatorial optimization problems — instances with $n > 30$ are extremely challenging.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of facilities/locations |
| $F_{ij}$ | Flow between facilities $i$ and $j$ |
| $D_{kl}$ | Distance between locations $k$ and $l$ |
| $\pi$ | Assignment permutation |

### Koopmans-Beckmann Formulation

$$\min_{\pi \in S_n} \sum_{i=1}^{n} \sum_{j=1}^{n} F_{ij} \cdot D_{\pi(i), \pi(j)} \tag{1}$$

Reduces to LAP when $F$ or $D$ is diagonal.

---

### Small Illustrative Instance

```
n = 3 facilities, 3 locations
Flow matrix F:       Distance matrix D:
  [[0, 5, 2],          [[0, 1, 2],
   [5, 0, 3],           [1, 0, 3],
   [2, 3, 0]]           [2, 3, 0]]

Assignment π = (0, 1, 2): cost = 5*1 + 2*2 + 5*1 + 3*3 + 2*2 + 3*3 = 32
Assignment π = (0, 2, 1): cost = 5*2 + 2*1 + 5*2 + 3*3 + 2*1 + 3*3 = 32
Assignment π = (1, 0, 2): cost = 5*1 + 2*3 + 5*1 + 3*2 + 2*3 + 3*2 = 34
Optimal: π = (0, 1, 2) or (0, 2, 1), cost = 32
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy QAP | Heuristic | $O(n^2)$ | Assign by largest flow to shortest distance |
| Simulated Annealing | Metaheuristic | $O(I \cdot n)$ | Pairwise swap with Boltzmann acceptance |
| Branch and Bound | Exact | Exponential | Gilmore-Lawler bound, practical for n <= 20 |

### Greedy QAP Pseudocode

```
GREEDY_QAP(F, D, n):
    # Rank facility pairs by flow (descending)
    flow_pairs = sort all (i, j) by F[i][j] descending
    # Rank location pairs by distance (ascending)
    dist_pairs = sort all (k, l) by D[k][l] ascending

    assignment = {}
    for each (i, j) in flow_pairs:
        if i not assigned and j not assigned:
            for each (k, l) in dist_pairs:
                if k not assigned and l not assigned:
                    assignment[i] = k
                    assignment[j] = l
                    break
    # Assign remaining facilities to remaining locations
    fill in unassigned pairs arbitrarily
    return assignment
```

**Complexity:** $O(n^2 \log n)$ for sorting, $O(n^2)$ for greedy assignment. No approximation guarantee exists for general QAP.

### Applications

- **Hospital layout** (minimizing patient transport between departments)
- **Campus planning** (placing departments to minimize inter-department travel)
- **Keyboard layout optimization** (minimizing finger travel distance for typing)
- **Backboard wiring** (VLSI placement of electronic components)

---

## 4. Implementations in This Repository

```
quadratic_assignment/
├── instance.py                    # QAPInstance, QAPSolution
├── heuristics/
│   └── greedy_qap.py             # Greedy assignment
├── metaheuristics/
│   └── simulated_annealing.py    # SA with pairwise swap
└── tests/
    └── test_qap.py               # QAP test suite
```

---

## 5. Key References

- Koopmans, T.C. & Beckmann, M. (1957). Assignment problems and the location of economic activities. *Econometrica*, 25(1), 53-76.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2012). *Assignment Problems*, Revised Reprint. SIAM.
- QAPLIB: http://www.opt.math.tugraz.at/qaplib/
- Sahni, S. & Gonzalez, T. (1976). P-complete approximation problems. *JACM*, 23(3), 555-565.
- Taillard, E. (1991). Robust taboo search for the quadratic assignment problem. *Parallel Computing*, 17(4-5), 443-455.

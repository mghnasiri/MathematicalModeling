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

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy QAP | Heuristic | $O(n^2)$ | Assign by largest flow to shortest distance |
| Simulated Annealing | Metaheuristic | $O(I \cdot n)$ | Pairwise swap with Boltzmann acceptance |

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

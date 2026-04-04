# Robust Single Machine Scheduling

## 1. Problem Definition

- **Input:** $n$ jobs with weights $w_j$, uncertain processing times $p_j(s)$ across $S$ scenarios, scenario probabilities $p_s$
- **Decision:** Job processing order (permutation $\pi$)
- **Objective:** Minimize worst-case regret of total weighted completion time: $\min_\pi \max_s [\sum w_j C_j(\pi, s) - \sum w_j C_j(\pi^*_s, s)]$
- **Constraints:** All jobs processed without preemption on a single machine
- **Classification:** NP-hard for general interval data (Lebedev & Averbakh, 2006). 2-approximable.

### Scheduling Notation

$1 \mid \text{uncertain } p_j \mid \min \max\text{-regret } \sum w_j C_j$

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of jobs |
| $p_j(s)$ | Processing time of job $j$ under scenario $s$ |
| $w_j$ | Weight (priority) of job $j$ |
| $C_j(\pi, s)$ | Completion time of job $j$ in schedule $\pi$ under scenario $s$ |
| $\pi^*_s$ | Optimal schedule (WSPT) under scenario $s$ |
| $S$ | Number of scenarios |

### Robustness Criteria

| Criterion | Formulation |
|-----------|-------------|
| Min-Max Cost | $\min_\pi \max_{s} \sum w_j C_j(\pi, s)$ |
| Min-Max Regret | $\min_\pi \max_{s} \bigl[\sum w_j C_j(\pi, s) - \sum w_j C_j(\pi^*_s, s)\bigr]$ |
| Expected Cost | $\min_\pi \sum_s p_s \sum w_j C_j(\pi, s)$ |

### WSPT Optimality (Per Scenario)

Under known processing times, the WSPT rule (sort by $p_j / w_j$ ascending) minimizes $\sum w_j C_j$. This gives $\pi^*_s$ for each scenario $s$.

### Small Illustrative Instance

```
n = 3, S = 2
Weights: w = [3, 1, 2]
Scenario 1: p = [2, 5, 3]  → WSPT order: [0, 2, 1], ΣwjCj = 3·2 + 2·5 + 1·10 = 26
Scenario 2: p = [6, 2, 4]  → WSPT order: [1, 2, 0], ΣwjCj = 1·2 + 2·6 + 3·12 = 50

Schedule π = [0, 2, 1]:
  s1: ΣwjCj = 26, regret = 26 - 26 = 0
  s2: ΣwjCj = 3·6 + 2·10 + 1·12 = 50, regret = 50 - 50 = 0
  max_regret = 0 → optimal for this instance
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Midpoint WSPT | Heuristic | $O(n \log n)$ | WSPT on mean processing times $\bar{p}_j = \sum_s p_s p_j(s)$ |
| Scenario Enumeration | Heuristic | $O(S \cdot n \log n + S^2 \cdot n)$ | WSPT per scenario, cross-evaluate regret |
| Worst-Case WSPT | Heuristic | $O(n \log n)$ | WSPT on maximum processing times |
| Simulated Annealing | Metaheuristic | $O(I \cdot S \cdot n)$ | Swap/insertion to minimize max regret |

### Midpoint WSPT

Compute mean processing time $\bar{p}_j$ across scenarios, apply WSPT rule. Fast but may miss worst-case structure. Provides a 2-approximation for interval uncertainty (Kasperski & Zielinski, 2008).

### Scenario Enumeration

1. Compute WSPT schedule $\pi^*_s$ for each scenario
2. Evaluate each candidate against all scenarios: $\text{regret}(\pi^*_s, s') = \sum w_j C_j(\pi^*_s, s') - \sum w_j C_j(\pi^*_{s'}, s')$
3. Select the candidate with minimum max regret

---

## 4. Implementations in This Repository

```
robust_scheduling/
├── instance.py                            # RobustSchedulingInstance, RobustSchedulingSolution
│                                          #   - total_weighted_completion(), max_regret_twc()
│                                          #   - mean_processing, random() factory
├── heuristics/
│   └── minmax_regret_heuristics.py        # Midpoint WSPT, scenario enumeration, worst-case WSPT
├── metaheuristics/
│   └── simulated_annealing.py             # Swap/insertion SA for min max-regret ΣwjCj
└── tests/
    └── test_robust_scheduling.py          # 13 tests, 3 test classes
```

---

## 5. Key References

- Kouvelis, P. & Yu, G. (1997). *Robust Discrete Optimization and Its Applications*. Springer. https://doi.org/10.1007/978-1-4757-2620-6
- Lebedev, V. & Averbakh, I. (2006). Complexity of minimizing the total flow time with interval data and minmax regret criterion. *Discrete Appl. Math.*, 154(15), 2167-2177. https://doi.org/10.1016/j.dam.2005.04.015
- Kasperski, A. & Zielinski, P. (2008). A 2-approximation algorithm for interval data minmax regret sequencing problems with the total flow time criterion. *Oper. Res. Lett.*, 36(5), 561-564. https://doi.org/10.1016/j.orl.2008.07.004
- Daniels, R.L. & Kouvelis, P. (1995). Robust scheduling to hedge against processing time uncertainty. *Management Science*, 41(2), 363-376.

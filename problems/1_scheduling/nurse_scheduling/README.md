# Nurse Scheduling Problem (NSP)

## 1. Problem Definition

- **Input:**
  - A set $I = \{1, 2, \ldots, n\}$ of nurses
  - A planning horizon of $d$ days: $D = \{1, 2, \ldots, d\}$
  - A set $S = \{1, 2, \ldots, s\}$ of shift types per day (e.g., morning, evening, night)
  - Demand matrix $R_{ds}$: required nurses for shift $s$ on day $d$
  - Nurse constraints: max total shifts, max consecutive working days, prohibited shift sequences
- **Decision:** Assign nurses to shifts — binary variable $x_{ids} = 1$ if nurse $i$ works shift $s$ on day $d$
- **Objective:** Minimize total under-coverage $\sum_{d,s} \max(0,\; R_{ds} - \sum_i x_{ids})$
- **Classification:** NP-hard in general (Ernst et al., 2004)
- **Also known as:** Nurse Rostering Problem (NRP), Staff Scheduling

### Complexity

| Problem | Complexity | Reference |
|---------|-----------|-----------|
| NSP with arbitrary constraints | NP-hard | Osogami & Imai (2000) |
| NSP with cyclic patterns | NP-hard | Lau (1996) |
| Shift assignment (single day) | Polynomial if no preferences | — |

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of nurses | $\mathbb{Z}^+$ |
| $d$ | Number of days in planning horizon | $\mathbb{Z}^+$ |
| $s$ | Number of shift types per day | $\mathbb{Z}^+$ |
| $x_{ids}$ | 1 if nurse $i$ works shift $s$ on day $d$ | $\{0, 1\}$ |
| $R_{ds}$ | Required nurses for shift $s$ on day $d$ | $\mathbb{Z}_{\geq 0}$ |
| $u_{ds}$ | Under-coverage for shift $s$ on day $d$ | $\mathbb{Z}_{\geq 0}$ |
| $W$ | Maximum total shifts per nurse | $\mathbb{Z}^+$ |
| $K$ | Maximum consecutive working days | $\mathbb{Z}^+$ |

### Formulation A: Coverage Minimization (ILP)

$$\min \quad \sum_{d=1}^{D} \sum_{s=1}^{S} u_{ds} \tag{1}$$

$$\text{s.t.} \quad \sum_{i=1}^{n} x_{ids} + u_{ds} \geq R_{ds} \quad \forall\, d,\, s \quad \text{(demand coverage)} \tag{2}$$

$$\sum_{s=1}^{S} x_{ids} \leq 1 \quad \forall\, i,\, d \quad \text{(at most one shift per day)} \tag{3}$$

$$\sum_{d=1}^{D} \sum_{s=1}^{S} x_{ids} \leq W \quad \forall\, i \quad \text{(max total shifts)} \tag{4}$$

$$\sum_{d'=d}^{d+K} \sum_{s=1}^{S} x_{id's} \leq K \quad \forall\, i,\, d \leq D-K \quad \text{(max consecutive days)} \tag{5}$$

$$x_{i,d,\text{night}} + x_{i,d+1,\text{morning}} \leq 1 \quad \forall\, i,\, d \quad \text{(no night then morning)} \tag{6}$$

$$x_{ids} \in \{0,1\},\quad u_{ds} \geq 0 \tag{7}$$

**Size:** $O(n \cdot d \cdot s)$ binary variables, $O(n \cdot d + d \cdot s)$ constraints. Practical for $n \leq 50$, $d \leq 28$ with modern MILP solvers.

### Formulation B: Weighted Preference Model

In practice, nurse scheduling often includes soft constraints (preferences):

$$\min \quad \alpha \sum_{d,s} u_{ds} + \beta \sum_{i,d,s} c_{ids} x_{ids}$$

where $c_{ids}$ is the cost of assigning nurse $i$ to shift $s$ on day $d$ (capturing preferences, fairness, overtime costs). Hard constraints remain as above; the objective trades coverage against preference satisfaction.

---

## 3. Variants

| Variant | Description |
|---------|-------------|
| Cyclic rostering | Pattern repeats every $T$ weeks; find one cycle |
| Self-scheduling | Nurses submit preferences; optimize satisfaction |
| Multi-skill | Nurses have skill levels; shifts require specific skills |
| INRC (international competition) | Standardized constraints from NSPLib |

---

## 4. Benchmark Instances

### NSPLib (Curtois, 2010)

Standard benchmark set for nurse scheduling with instances ranging from 8 to 120 nurses over 4-week horizons.

**URL:** http://www.schedulingbenchmarks.org/nsp/

### Small Illustrative Instance

4 nurses, 3 days, 2 shifts (Morning/Night), max 2 shifts total, max 2 consecutive days:

```
Demand:
       Morning  Night
Day 1:    2       1
Day 2:    2       1
Day 3:    1       1

Feasible solution (under_coverage = 0):
Nurse 0: Day1-M, Day2-M
Nurse 1: Day1-M, Day3-M
Nurse 2: Day2-M, Day3-N
Nurse 3: Day1-N, Day2-N
```

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Integer Linear Programming

The ILP formulation (Section 2) can be solved directly by MILP solvers (Gurobi, CPLEX, HiGHS). For instances up to $\sim$50 nurses and 28-day horizons, modern solvers find optimal solutions within minutes. The key acceleration techniques:

- **Column generation:** Generate nurse schedules (columns) as needed; the pricing subproblem is a constrained shortest path
- **Branch-and-price:** Column generation inside B&B; state-of-the-art for large instances
- **Constraint propagation:** CP solvers handle the diverse constraint types naturally

### 5.2 Constructive Heuristics

#### Greedy Roster

```
ALGORITHM GreedyRoster(instance)
  schedule ← zero matrix (n × d × s)
  FOR d = 1 TO D:
    FOR s = 1 TO S:
      needed ← R[d][s]
      candidates ← nurses feasible for (d,s), sorted by total shifts ascending
      FOR nurse in candidates (while assigned < needed):
        IF feasible(nurse, d, s, schedule):
          schedule[nurse][d][s] ← 1
          assigned ← assigned + 1
      under_coverage += max(0, needed - assigned)
  RETURN schedule, under_coverage
```

**Complexity:** $O(d \cdot s \cdot n \log n)$ (sorting candidates per shift).

**Quality:** Typically achieves zero under-coverage when demand $\leq n/2$ per shift, but makes no attempt to optimize preferences or fairness.

### 5.3 Improvement Heuristics

| Neighborhood | Move Definition | Size |
|-------------|----------------|------|
| Swap shifts | Exchange shift assignments of two nurses on same day | $O(n^2 \cdot d)$ |
| Reassign | Move one nurse from shift $s_1$ to shift $s_2$ on day $d$ | $O(n \cdot d \cdot s^2)$ |
| Block swap | Exchange a block of consecutive days between two nurses | $O(n^2 \cdot d)$ |

### 5.4 Metaheuristics

Note: the current implementation includes only the greedy heuristic. The following methods are standard for NSP:

| Method | Encoding | Key Feature |
|--------|----------|-------------|
| SA | Direct schedule matrix | Swap/reassign neighborhood, slow cooling |
| Tabu Search | Direct schedule | Tabu on (nurse, day, shift) triples |
| GA | Binary matrix or pattern-based | Row (nurse) crossover, feasibility repair |
| Variable Neighborhood Search | Direct schedule | Multiple neighborhood structures |

**Recommended parameter ranges for SA on NSP:**

| Parameter | Symbol | Typical Range | Notes |
|-----------|--------|---------------|-------|
| Initial temperature | $T_0$ | Accept 30-50% worsening initially | Calibrate on instance |
| Cooling rate | $\alpha$ | 0.995-0.999 | Slow cooling for complex constraints |
| Iterations per temp | $L$ | $n \cdot d$ | Scale with problem size |
| Neighborhood | — | Swap + reassign | Combined moves |

### 5.5 Decomposition Approaches

**Two-phase approach (common in practice):**
1. **Phase 1 — Pattern generation:** Generate feasible shift patterns for each nurse (sequences of shifts over the horizon satisfying all hard constraints)
2. **Phase 2 — Pattern assignment:** Select one pattern per nurse to minimize under-coverage → set-covering/partitioning problem

This decomposition is natural because the hard constraints (consecutive days, night-morning prohibition) are per-nurse, while demand coverage is cross-nurse.

---

## 6. Implementation Guide

### Constraint Handling Tips

- **Consecutive days:** Use a sliding window of size $K+1$ over the day index
- **Night-then-morning prohibition:** Check shift type on day $d$ and day $d+1$
- **Fairness:** Track total shifts per nurse; penalize imbalance in objective
- **Weekends:** Add constraints that both Saturday and Sunday are either both worked or both off

### Common Pitfalls

- Forgetting to re-check feasibility after assigning a nurse to another shift on the same day
- Not handling the boundary between planning horizons (what happened the week before?)
- Ignoring skill levels in general-purpose implementations

---

## 7. Computational Results Summary

| Method | Small ($n \leq 20$) | Medium ($20 < n \leq 50$) | Large ($n > 50$) |
|--------|---------------------|---------------------------|-------------------|
| ILP (exact) | Optimal in seconds | Optimal in minutes | May timeout |
| Greedy | Zero under-coverage (low demand) | 0-5% under-coverage | Fast baseline |
| SA | Near-optimal | Within 1-2% | Best single-method for large |
| Branch-and-price | Optimal | Optimal with effort | State-of-the-art exact |

---

## 8. Implementations in This Repository

```
nurse_scheduling/
├── instance.py                    # NurseSchedulingInstance, NurseSchedulingSolution
├── heuristics/
│   └── greedy_roster.py           # Greedy shift filling — O(d·s·n log n)
└── tests/
    └── test_nurse_scheduling.py   # NSP test suite
```

**Total:** 0 exact methods, 1 constructive heuristic, 0 metaheuristics, 1 test file.

---

## 9. Key References

### Seminal / Survey

- Ernst, A.T., Jiang, H., Krishnamoorthy, M. & Sier, D. (2004). Staff scheduling and rostering: A review of applications, methods and models. *European Journal of Operational Research*, 153(1), 3-27.
- Burke, E.K., De Causmaecker, P., Vanden Berghe, G. & Van Landeghem, H. (2004). The state of the art of nurse rostering. *Journal of Scheduling*, 7(6), 441-499.
- Warner, D.M. (1976). Scheduling nursing personnel according to nursing preference: A mathematical programming approach. *Operations Research*, 24(5), 842-856.

### Methods

- Jaumard, B., Semet, F. & Vovor, T. (1998). A generalized linear programming model for nurse scheduling. *European Journal of Operational Research*, 107(1), 1-18.
- Dowsland, K.A. (1998). Nurse scheduling with tabu search and strategic oscillation. *European Journal of Operational Research*, 106(2-3), 393-407.
- Aickelin, U. & Dowsland, K.A. (2004). An indirect genetic algorithm for a nurse-scheduling problem. *Computers & Operations Research*, 31(5), 761-778.
- Curtois, T. (2010). Employee shift scheduling benchmark data sets. http://www.schedulingbenchmarks.org/nsp/

### Textbook

- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems.* 5th ed. Springer. Chapter 10 covers personnel scheduling.

# Workforce Scheduling Problem

## 1. Problem Definition

- **Input:** Employees with skill sets and availability windows; shifts with skill demands per period
- **Decision:** Assign employees to shifts
- **Objective:** Minimize uncovered demand (unmet skill-shift requirements)
- **Constraints:** Each employee assigned to at most one shift per period; skill match required
- **Classification:** NP-hard (reduces from set cover)

### Notation

$\text{WS} \mid \text{skills, availability} \mid \min \text{uncovered demand}$

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy shift fill | Heuristic | $O(n \cdot d)$ | Greedily assign available skilled employees |
| Set-cover greedy | Heuristic | $O(n \cdot S \cdot d)$ | Cover each shift-skill demand using set cover logic |
| LP relaxation | Exact | polynomial | LP relaxation of assignment ILP, round fractional values |

### Greedy Cover Pseudocode

```
GREEDY_SHIFT_COVER(employees, shifts, demands):
    assignment = {}
    uncovered = total demand across all (shift, skill) pairs
    for each (shift, skill) pair sorted by demand descending:
        candidates = employees available for shift with matching skill
                     and not yet assigned to this period
        sort candidates by number of additional demands they can cover (descending)
        while demand(shift, skill) > 0 and candidates not empty:
            e* = candidates.pop(0)   // most versatile available employee
            assignment[e*] = shift
            demand(shift, skill) -= 1
            uncovered -= 1
            remove e* from candidates for this period
    return assignment, uncovered
```

---

## 3. Illustrative Instance

3 employees, 2 shifts, 2 skills:

| Employee | Skills | Availability |
|----------|--------|-------------|
| E1 | {Cashier} | {Morning, Evening} |
| E2 | {Cashier, Stocking} | {Morning} |
| E3 | {Stocking} | {Evening} |

| Shift | Demand |
|-------|--------|
| Morning | 1 Cashier, 1 Stocking |
| Evening | 1 Cashier |

Greedy: Assign E2 to Morning (covers Cashier or Stocking -- choose Stocking since E2 is the only morning stocker), E1 to Morning-Cashier, E1 to Evening-Cashier (conflict -- E1 already used). Reassign: E2->Morning-Stocking, E1->Morning-Cashier, E1->Evening-Cashier. Uncovered = 0.

---

## 4. Applications

- **Retail staffing:** Assigning cashiers and floor associates to morning/evening shifts based on skill certifications and labor regulations.
- **Healthcare:** Allocating nurses and technicians with specific qualifications (ICU, pediatrics) across hospital wards and shift rotations.
- **Call centers:** Scheduling agents with language skills and product expertise to cover demand peaks across time zones.
- **Manufacturing:** Assigning operators with machine certifications to production lines across multiple shifts.

---

## 5. Implementations in This Repository

```
workforce_scheduling/
├── instance.py                    # WorkforceInstance, WorkforceSolution
├── heuristics/
│   └── greedy_shift_fill.py       # Greedy skill-aware shift filling
└── tests/
    └── test_workforce.py          # Workforce scheduling test suite
```

---

## 6. Key References

- Ernst, A.T., Jiang, H., Krishnamoorthy, M. & Sier, D. (2004). Staff scheduling and rostering: A review. *European J. Oper. Res.*, 153(1), 3-27.
- Van den Bergh, J., Belien, J., De Bruecker, P., Demeulemeester, E. & De Boeck, L. (2013). Personnel scheduling: A literature review. *European J. Oper. Res.*, 226(3), 367-385.
- Brucker, P., Qu, R. & Burke, E.K. (2011). Personnel scheduling: Models and complexity. *European J. Oper. Res.*, 210(3), 467-473.

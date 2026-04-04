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

---

## 3. Implementations in This Repository

```
workforce_scheduling/
├── instance.py                    # WorkforceInstance, WorkforceSolution
├── heuristics/
│   └── greedy_shift_fill.py       # Greedy skill-aware shift filling
└── tests/
    └── test_workforce.py          # Workforce scheduling test suite
```

---

## 4. Key References

- Ernst, A.T., Jiang, H., Krishnamoorthy, M. & Sier, D. (2004). Staff scheduling and rostering: A review. *European J. Oper. Res.*, 153(1), 3-27.

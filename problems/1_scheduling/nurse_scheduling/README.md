# Nurse Scheduling Problem (NSP)

## 1. Problem Definition

- **Input:** $n$ nurses, $d$ days, $s$ shift types per day, demand requirements per shift, nurse constraints (max shifts, no consecutive nights, etc.)
- **Decision:** Assign nurses to shifts across the planning horizon
- **Objective:** Minimize total under-coverage (unmet demand)
- **Classification:** NP-hard (Ernst et al., 2004)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy roster | Heuristic | $O(n \cdot d \cdot s)$ | Fill shifts greedily by demand priority |

---

## 3. Implementations in This Repository

```
nurse_scheduling/
├── instance.py                    # NurseSchedulingInstance, NurseSchedulingSolution
├── heuristics/
│   └── greedy_roster.py           # Greedy shift filling
└── tests/
    └── test_nurse_scheduling.py   # NSP test suite
```

---

## 4. Key References

- Ernst, A.T., Jiang, H., Krishnamoorthy, M. & Sier, D. (2004). Staff scheduling and rostering: A review. *European J. Oper. Res.*, 153(1), 3-27.

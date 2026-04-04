# Multi-Project Scheduling Problem (MPSP)

## 1. Problem Definition

- **Input:** $P$ projects, each with activities, precedence constraints, and resource requirements; shared renewable resources; project deadlines
- **Decision:** Start time of each activity across all projects
- **Objective:** Minimize total project delays (sum of makespans or weighted tardiness)
- **Classification:** NP-hard (extends RCPSP to multiple projects with shared resources)

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Priority SGS | Heuristic | $O(P \cdot n)$ | Serial generation scheme with priority rules across projects |

---

## 3. Implementations in This Repository

```
project_scheduling/
├── instance.py                        # MultiProjectInstance, MultiProjectSolution
├── heuristics/
│   └── priority_sgs.py               # Priority-rule serial generation scheme
└── tests/
    └── test_project_scheduling.py     # Multi-project scheduling test suite
```

---

## 4. Key References

- Lova, A., Tormos, P., Cervantes, M. & Barber, F. (2009). An efficient hybrid GA for scheduling projects with resource constraints and multiple execution modes. *Int. J. Prod. Econ.*, 117(2), 302-316.

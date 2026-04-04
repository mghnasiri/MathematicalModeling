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
| Delay-based priority | Heuristic | $O(P \cdot n)$ | Prioritize activities from most-delayed project |
| Hybrid GA | Metaheuristic | problem-dependent | Multi-project GA with activity-list encoding |

### Serial Generation Scheme (SGS) Pseudocode

```
SERIAL_SGS_MULTIPROJECT(projects, resources, priority_rule):
    eligible = set of all activities with all predecessors scheduled
    schedule = {}
    while eligible is not empty:
        j* = select activity from eligible using priority_rule
              (e.g., LFT — latest finish time across all projects)
        t  = earliest time t such that:
              - all predecessors of j* finish by t
              - resource availability r_k(t..t+p_j*) >= req_k(j*)
                for all resource types k
        schedule[j*] = t
        update resource profiles
        update eligible (add successors of j* whose predecessors are all scheduled)
    return schedule
```

---

## 3. Illustrative Instance

Consider 2 projects sharing 1 resource (capacity = 2):

| Project | Activity | Duration | Predecessors | Resource |
|---------|----------|----------|-------------|----------|
| P1 | A1 | 3 | - | 1 |
| P1 | A2 | 2 | A1 | 1 |
| P2 | B1 | 2 | - | 1 |
| P2 | B2 | 4 | B1 | 1 |

With LFT priority: schedule A1 at t=0, B1 at t=0 (both fit in capacity 2), then A2 at t=3, B2 at t=3. Total makespan sum = 5 + 7 = 12.

---

## 4. Applications

- **Construction management:** Coordinating multiple building projects sharing cranes, crews, and specialized equipment across a portfolio.
- **R&D portfolio:** Pharmaceutical companies managing parallel drug development pipelines with shared lab resources and regulatory milestones.
- **IT project management:** Scheduling software development sprints across teams that share testing infrastructure.
- **Defense acquisition:** Multi-system integration programs where subsystems share engineering and testing facilities.

---

## 5. Implementations in This Repository

```
project_scheduling/
├── instance.py                        # MultiProjectInstance, MultiProjectSolution
├── heuristics/
│   └── priority_sgs.py               # Priority-rule serial generation scheme
└── tests/
    └── test_project_scheduling.py     # Multi-project scheduling test suite
```

---

## 6. Key References

- Lova, A., Tormos, P., Cervantes, M. & Barber, F. (2009). An efficient hybrid GA for scheduling projects with resource constraints and multiple execution modes. *Int. J. Prod. Econ.*, 117(2), 302-316.
- Kolisch, R. & Padman, R. (2001). An integrated survey of deterministic project scheduling. *Omega*, 29(3), 249-272.
- Hartmann, S. & Briskorn, D. (2010). A survey of variants and extensions of the RCPSP. *European J. Oper. Res.*, 207(1), 1-14.
- Browning, T.R. & Yassine, A.A. (2010). Resource-constrained multi-project scheduling: Priority rule performance revisited. *Int. J. Prod. Econ.*, 126(2), 212-228.

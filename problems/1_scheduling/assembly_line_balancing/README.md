# Simple Assembly Line Balancing Problem (SALBP-1)

## 1. Problem Definition

- **Input:** $n$ tasks with processing times $t_j$, precedence constraints (DAG), cycle time $C$
- **Decision:** Assign each task to a workstation
- **Objective:** Minimize the number of workstations
- **Constraints:** Precedence respected; total task time per station $\leq C$
- **Classification:** NP-hard (Wee & Magazine, 1982)

---

## 2. Mathematical Formulation

$$\min \sum_{k=1}^{K} y_k \tag{1}$$

$$\sum_{k=1}^{K} x_{jk} = 1 \quad \forall j \tag{2}$$

$$\sum_{j=1}^{n} t_j x_{jk} \leq C \cdot y_k \quad \forall k \tag{3}$$

$$\sum_{k} k \cdot x_{ik} \leq \sum_{k} k \cdot x_{jk} \quad \forall (i,j) \in \text{precedence} \tag{4}$$

**Applications:** Automotive assembly, electronics manufacturing, packaging lines, and any serial production system where work is divided among sequential stations connected by a conveyor.

### Problem Variants

- **SALBP-2:** Given a fixed number of stations, minimize the cycle time $C$.
- **U-shaped lines:** Tasks can be assigned from both ends of the line, relaxing precedence locality.
- **Mixed-model ALB:** Multiple product variants share the same line, requiring balancing across models.

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Ranked Positional Weight | Heuristic | $O(n \log n)$ | Priority = task time + sum of successor times |
| Largest Candidate Rule | Heuristic | $O(n \log n)$ | Assign tasks in decreasing processing time order |
| Branch and Bound | Exact | exponential | Enumerate station assignments with dominance pruning |

### Ranked Positional Weight (RPW) Pseudocode

```
RPW(tasks, times, precedence, cycle_time):
    // Step 1: compute positional weights
    for each task j in reverse topological order:
        RPW[j] = times[j] + sum(RPW[s] for s in successors(j))

    // Step 2: sort tasks by decreasing RPW
    sorted_tasks = sort tasks by RPW descending

    // Step 3: assign to stations
    station = 1
    remaining_time = cycle_time
    assignment = {}
    for each task j in sorted_tasks:
        if all predecessors of j are assigned
           AND times[j] <= remaining_time:
            assignment[j] = station
            remaining_time -= times[j]
        else:
            station += 1
            remaining_time = cycle_time - times[j]
            assignment[j] = station
    return assignment, station   // station count = number of workstations
```

---

## 4. Illustrative Instance

5 tasks, cycle time $C = 6$:

| Task | Time | Predecessors |
|------|------|-------------|
| A | 2 | - |
| B | 3 | A |
| C | 2 | A |
| D | 4 | B |
| E | 1 | C, D |

Positional weights: E=1, D=4+1=5, C=2+1=3, B=3+5=8, A=2+8=10. Sorted: A(10), B(8), D(5), C(3), E(1). Station 1: A(2)+B(3)=5. Station 2: D(4)+C(2)=6. Station 3: E(1). Result: 3 stations.

---

## 5. Implementations in This Repository

```
assembly_line_balancing/
├── instance.py                    # SALBPInstance, SALBPSolution
├── heuristics/
│   └── rpw.py                     # Ranked Positional Weight heuristic
└── tests/
    └── test_salbp.py              # SALBP test suite
```

---

## 6. Key References

- Scholl, A. & Becker, C. (2006). State-of-the-art exact and heuristic solution procedures for SALBP. *European J. Oper. Res.*, 168(3), 666-693.
- Wee, T.S. & Magazine, M.J. (1982). Assembly line balancing as generalized bin packing. *Oper. Res. Lett.*, 1(2), 56-58.
- Helgeson, W.B. & Birnie, D.P. (1961). Assembly line balancing using the ranked positional weight technique. *J. Ind. Eng.*, 12(6), 394-398.
- Boysen, N., Fliedner, M. & Scholl, A. (2007). A classification of assembly line balancing problems. *European J. Oper. Res.*, 183(2), 674-693.

# Unrelated Parallel Machine with Total Tardiness (PM Variant)

## What Changes

In the **unrelated tardiness** variant (Rm || Sigma T_j), the objective shifts
from makespan to total tardiness, and machines are unrelated (processing time
p_{ij} depends on both job j and machine i). Each job j has a due date d_j,
and tardiness T_j = max(0, C_j - d_j) measures how late each job finishes.
This models due-date-driven manufacturing environments with heterogeneous
equipment -- semiconductor fabs with different tool capabilities, printing
shops with different press speeds, and logistics centers with varied
processing capacities.

The key structural difference from the base makespan objective is that
tardiness is non-regular: reordering jobs on a machine can increase some
tardinesses while decreasing others. The EDD and ATC dispatching rules
replace LPT as the primary heuristic strategies.

## Mathematical Formulation

The base PM formulation changes the objective:

**Tardiness per job:**
```
T_j = max(0, C_j - d_j)    for all j
```

**Objective:**
```
min  sum_j  T_j
```

**Unrelated processing times:**
```
C_j = completion of job j on its assigned machine i,
      depends on p_{ij} and the sequence position on machine i
```

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Rm \|\| Sigma T_j | Strongly NP-hard | Pinedo (2016) |
| 1 \|\| Sigma T_j | NP-hard | Du & Leung (1990) |
| Pm \|\| Sigma T_j (identical) | Strongly NP-hard | -- |

Strongly NP-hard even for identical parallel machines. The combination of
unrelated speeds and due dates makes this among the hardest parallel machine
scheduling problems.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| LPT (base heuristic) | No | Designed for makespan, ignores due dates |
| EDD-ECT (variant) | Yes | EDD priority + earliest completion time machine |
| ATC dispatching (variant) | Yes | Apparent Tardiness Cost composite rule |
| SA (variant metaheuristic) | Yes | Reassign and swap moves minimizing Sigma T_j |

## Implementations

Python files in this directory:
- `instance.py` -- RmTardinessInstance, due dates, tardiness evaluation
- `heuristics.py` -- EDD-ECT, ATC-based dispatching
- `metaheuristics.py` -- SA with reassign and swap moves
- `tests/test_rm_tardiness.py` -- 19 tests

## Applications

- Semiconductor fabrication (due-date-driven lot scheduling)
- Job shops with heterogeneous machines (order fulfillment)
- Logistics distribution centers (package sorting with deadlines)
- Hospital operating rooms (surgery scheduling with urgency)

## Key References

- Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems. 5th ed. Springer.
- Weng, M.X., Lu, J. & Ren, H. (2001). "Unrelated parallel machine scheduling with setup consideration and a total weighted completion time objective." IJPE 70(3), 215-226.
- Du, J. & Leung, J.Y.-T. (1990). "Minimizing total tardiness on one machine is NP-hard." Mathematics of Operations Research 15(3), 483-495.

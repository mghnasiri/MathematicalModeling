# Flexible Job Shop with Tardiness (Job Shop Variant)

## What Changes

The standard Job Shop Problem (JSP) minimizes makespan with fixed machine assignments. This variant combines **two generalizations**:

1. **Flexible machine assignment** (from FJSP): each operation can be processed on any machine from a set of eligible machines, with machine-dependent processing times.
2. **Weighted tardiness objective** (from 1||SUM wjTj): each job has a due date d_j and weight w_j; the objective is to minimize total weighted tardiness SUM wjTj.

This creates a joint **routing + sequencing + due-date** optimization problem that is significantly harder than either FJSP or weighted tardiness scheduling alone. The interaction between machine choice and tardiness means that assigning an operation to a faster machine may reduce one job's tardiness but delay other jobs sharing that machine.

**Real-world motivation**: make-to-order manufacturing with flexible CNC machines and customer delivery deadlines, semiconductor wafer fabrication with multi-tool processing stations and lot priorities, flexible manufacturing systems with heterogeneous workstations.

## Mathematical Formulation

Extends JSP with flexible machine assignment and tardiness objective:

```
min  sum_j w_j * max(0, C_j - d_j)

s.t. m_{j,o} in E_{j,o}                              (machine eligibility)
     S_{j,o+1} >= S_{j,o} + p_{j,o,m_{j,o}}          (job precedence)
     no overlap on any machine                         (capacity)
     C_j = S_{j,last} + p_{j,last,m_{j,last}}         (completion time)
```

Where E_{j,o} is the set of eligible machines for operation o of job j, and p_{j,o,m} is the processing time of that operation on machine m. In total FJSP, all machines are eligible for every operation; in partial FJSP, only subsets are eligible.

## Complexity

- **Strongly NP-hard**: generalizes both FJSP (NP-hard for makespan) and 1||SUM wjTj (strongly NP-hard).
- The flexible machine assignment adds a combinatorial routing layer on top of the sequencing problem.
- No known constant-factor approximation for the combined problem.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| EDD + ECT dispatching | Yes | Jobs by due date, operations to earliest-completing machine. |
| WATC dispatching | Yes | Weighted ATC priority combining w_j/p_j with due date urgency. |
| Machine reassignment | Yes | SA move: reassign an operation to a different eligible machine. |
| Operation swap | Yes | SA move: swap priority of two operations on the same machine. |
| Simulated Annealing | Yes | Machine reassignment + operation swap, warm-started with WATC. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `FlexTardJSPInstance` (operations with eligible-machine lists, due_dates, weights), `FlexTardJSPSolution` (machine_assignments, start_times, total_weighted_tardiness), `validate_solution()`, `small_ftjsp_3x3()` benchmark |
| `heuristics.py` | `edd_ect()` (EDD priority + ECT machine selection), `watc_dispatch()` (Weighted ATC priority with ECT) |
| `metaheuristics.py` | `simulated_annealing()` with machine reassignment and operation swap neighborhoods |
| `tests/test_ftjsp.py` | Test suite covering solution validation, machine eligibility checks, tardiness computation |

## Relationship to Base JSP

When each operation has exactly one eligible machine (|E_{j,o}| = 1 for all j, o) and the objective is makespan, this reduces to the standard JSP. Adding machine flexibility and changing the objective to weighted tardiness creates two independent axes of generalization that interact in complex ways.

## Key References

- Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search. *Annals of Operations Research*, 41(3), 157-183.
- Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood functions for the flexible job shop problem. *Journal of Scheduling*, 3(1), 3-20.
- Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035-1047.

# Job Shop with Weighted Tardiness (Job Shop Variant)

## What Changes

The standard JSP minimizes **makespan** (Cmax). This variant changes the objective to **total weighted tardiness** SUM wjTj: each job j has a due date d_j and a priority weight w_j, and the tardiness Tj = max(0, Cj - dj) measures how late each job finishes. The objective penalizes late jobs proportionally to their weight.

- **Objective change**: from makespan (min-max) to weighted tardiness (weighted sum of lateness).
- **Due dates and weights** are added as job parameters, introducing a customer-priority dimension.
- **Critical path reasoning** (central to JSP makespan) is less directly applicable; dispatching rules like ATC and WSPT become more important.

**Real-world motivation**: make-to-order manufacturing with customer deadlines and varying order priorities, semiconductor fab scheduling with wafer lots of different urgency levels, batch chemical processing with delivery commitments and penalty costs.

## Mathematical Formulation

Extends JSP with due dates, weights, and tardiness objective:

```
min  sum_j w_j * max(0, C_j - d_j)

s.t. S_{j,k+1} >= S_{j,k} + p_{j,k}                  (job precedence)
     no two operations on the same machine overlap      (capacity)
     C_j = S_{j,last} + p_{j,last}                     (completion time)
     S_{j,k} >= 0                                       (non-negativity)
```

Where d_j is the due date and w_j is the weight (priority) of job j.

## Complexity

- **Strongly NP-hard**: even the single-machine case 1||SUM wjTj is strongly NP-hard (Lenstra et al., 1977).
- The job shop structure adds machine-conflict resolution on top of the already intractable tardiness objective.
- No known polynomial-time approximation schemes exist.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| ATC dispatching | Yes | Apparent Tardiness Cost: combines w_j/p_j ratio with due date urgency; Giffler-Thompson active schedule. |
| WSPT dispatching | Yes | Weighted Shortest Processing Time priority; simpler but less effective than ATC. |
| Swap moves | Yes | SA move: swap two jobs in the priority ordering. |
| Insert moves | Yes | SA move: relocate a job in the priority sequence. |
| Simulated Annealing | Yes | Priority-permutation encoding decoded via dispatching; warm-started with ATC. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `WTJSPInstance` (operations, due_dates, weights), `WTJSPSolution` (start_times, total_weighted_tardiness), `schedule_from_sequences()`, `validate_solution()`, `small_wtjsp_3x3()` benchmark |
| `heuristics.py` | `atc_dispatch()` (ATC priority with Giffler-Thompson), `wspt_dispatch()` (WSPT priority) |
| `metaheuristics.py` | `simulated_annealing()` with priority-permutation encoding, swap/insert moves, ATC warm-start |
| `tests/test_wtjsp.py` | Test suite covering tardiness computation, dispatching correctness, SA improvement |

## Key References

- Singer, M. & Pinedo, M. (1998). A computational study of branch and bound techniques for minimizing the total weighted tardiness in job shops. *IIE Transactions*, 30(2), 109-118.
- Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035-1047.
- Pinedo, M. (2016). *Scheduling: Theory, Algorithms, and Systems*. 5th ed. Springer.

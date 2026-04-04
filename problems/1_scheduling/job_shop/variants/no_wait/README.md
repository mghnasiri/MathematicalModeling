# No-Wait Job Shop Scheduling (Job Shop Variant)

## What Changes

The standard JSP allows jobs to wait between consecutive operations (in intermediate buffers). The **No-Wait Job Shop (NW-JSP)** adds the constraint that once a job starts processing, it must proceed through **all its operations without any idle time** between them. The start time of a job fully determines all its operation start times.

- **No intermediate buffering**: a job's operation sequence is a contiguous block on the time axis.
- **Start time determines everything**: unlike standard JSP where each operation is scheduled independently, here a single decision variable (job start time) fixes the entire schedule for that job.
- **Machine conflicts** are harder to resolve because shifting one operation shifts the entire job.
- **Fewer scheduling variables**: each job has only one decision variable (its start time), compared to one per operation in standard JSP.

**Real-world motivation**: steel manufacturing (continuous casting requires uninterrupted metal flow), chemical processing (reactions cannot be paused mid-process), food processing (perishable goods require continuous handling), pharmaceutical manufacturing (drug production with strict timing constraints).

## Mathematical Formulation

Extends JSP with the no-wait constraint:

```
min  Cmax

s.t. S_{j,k+1} = S_{j,k} + p_{j,k}                  (no-wait: next op starts immediately)
     no two operations on the same machine overlap     (capacity)
     S_j >= 0                                          (non-negativity)
```

Since S_{j,k} = S_{j,0} + sum_{l=0}^{k-1} p_{j,l}, the entire schedule is determined by the n job start times {S_{j,0}}. This structural simplification means the problem can be viewed as finding n start times that avoid all pairwise machine conflicts.

## Complexity

- **NP-hard** (generalizes JSP; also related to no-wait flow shop which is NP-hard for m >= 3).
- Sahni & Cho (1979) established NP-hardness of no-wait scheduling even for restricted cases.
- The contiguous-block structure makes some neighborhoods more efficient (fewer degrees of freedom) but feasibility checking is more constrained.
- Related to the no-wait flow shop (NWFSP), which reduces to an asymmetric TSP on the delay matrix.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy Insertion | Yes | Insert jobs one at a time at feasible start time minimizing Cmax; O(n^2 * m). |
| SPT order | Yes | Schedule jobs in shortest total processing time order with greedy start assignment. |
| Swap moves | Yes | SA move: swap two jobs in the permutation. |
| Insertion moves | Yes | SA move: relocate a job to a different position. |
| Simulated Annealing | Yes | Permutation-based with swap/insert, decoded via greedy start-time assignment. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `NWJSPInstance` (n, m, operations with job-specific routings), `NWJSPSolution` (job_starts, makespan), `validate_solution()` with no-wait and machine-conflict checks, `small_nwjsp_3x3()` benchmark |
| `heuristics.py` | `greedy_insertion()` (insert jobs at earliest feasible time), `spt_schedule()` (SPT-ordered greedy) |
| `metaheuristics.py` | `simulated_annealing()` with permutation encoding, swap/insert moves, warm-started with greedy insertion |
| `tests/test_nwjsp.py` | Test suite covering no-wait constraint validation, feasibility, makespan correctness |

## Relationship to Base JSP

When jobs are allowed to wait between operations (the standard JSP assumption), the no-wait constraint is relaxed and this reduces to standard JSP. The no-wait constraint eliminates intermediate buffers, linking the NW-JSP more closely to flow-shop variants where continuous processing is required.

## Key References

- Mascis, A. & Pacciarelli, D. (2002). Job-shop scheduling with blocking and no-wait constraints. *European Journal of Operational Research*, 143(3), 498-517.
- Sahni, S. & Cho, Y. (1979). Complexity of scheduling shops with no wait in process. *Mathematics of Operations Research*, 4(4), 448-457.
- Schuster, C.J. & Framinan, J.M. (2003). Approximative procedures for no-wait job shop scheduling. *Operations Research Letters*, 31(4), 308-318. [TODO: verify DOI]

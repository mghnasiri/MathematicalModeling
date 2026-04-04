# Single Machine Batch Scheduling (SM Variant)

## What Changes

In the **batch scheduling** variant (1 | batch, s_j | Sigma w_j C_j), jobs are
grouped into batches rather than scheduled individually. All jobs in a batch
begin processing simultaneously; the batch processing time equals the maximum
processing time among its jobs (parallel batching model). A setup time s is
incurred between consecutive batches. This models situations where a machine
processes multiple items at once -- heat treatment furnaces, chemical autoclaves,
semiconductor diffusion ovens, and burn-in testing where batch size is limited
by capacity.

The key structural difference from the base single machine problem is the
two-level decision: (1) partition jobs into batches, and (2) sequence the
batches. The WSPT rule no longer directly applies because batch completion
time depends on the slowest job in each batch.

## Mathematical Formulation

The base SM formulation gains batch partitioning constraints:

**Batch processing time:**
```
P_B = max_{j in B} p_j    for each batch B
```

**Batch completion time with setup:**
```
C_{B_k} = C_{B_{k-1}} + s + P_{B_k}    for batch k = 1, ..., K
C_{B_0} = 0
```

**Job completion time:** All jobs in batch B_k complete at C_{B_k}:
```
C_j = C_{B_k}    for all j in B_k
```

**Objective:**
```
min  sum_j  w_j * C_j
```

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| 1 \| batch, s \| Sigma w_j C_j | NP-hard | Potts & Kovalyov (2000) |
| 1 \| batch, s \| Sigma C_j | NP-hard | Webster & Baker (1995) |
| Equal processing times | Polynomial | Potts & Kovalyov (2000) |

NP-hard even with equal setup times, because the batch partitioning
sub-problem alone is hard (related to bin packing).

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| WSPT (base heuristic) | Partially | Ignores batching -- treats each job as own batch |
| WSPT Batching (variant) | Yes | Group similar-p_j jobs, order batches by WSPT ratio |
| Single-Job Batches (variant) | Yes | Baseline: each job is its own batch |
| SA (variant metaheuristic) | Yes | Move/merge/split batch neighborhood |

## Implementations

Python files in this directory:
- `instance.py` -- BatchSMInstance, batch completion time evaluation
- `heuristics.py` -- WSPT-batch grouping, single-job baseline
- `metaheuristics.py` -- SA with move/merge/split batch neighborhood
- `tests/test_batch.py` -- 17 tests

## Applications

- Heat treatment furnaces (batch of parts processed together)
- Semiconductor diffusion ovens (wafer batches)
- Chemical autoclaves (batch processing of compounds)
- Burn-in testing (electronic components tested in batches)

## Key References

- Webster, S. & Baker, K.R. (1995). "Scheduling groups of jobs on a single machine." Operations Research 43(4), 692-703.
- Potts, C.N. & Kovalyov, M.Y. (2000). "Scheduling with batching: a review." EJOR 120(2), 228-249.
- Uzsoy, R. (1994). "Scheduling a single batch processing machine with non-identical job sizes." Int. J. Production Research 32(7), 1615-1635. [TODO: verify]

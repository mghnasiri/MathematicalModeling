# Lot Streaming Flow Shop (PFSP Variant)

## What Changes

In the **lot streaming flow shop** (Fm | prmu, lot-streaming | Cmax), each job
is split into S sublots (transfer batches) that can move between machines
independently. This allows overlapping: while the second sublot of job j is
processed on machine 1, the first sublot can already be on machine 2. Lot
streaming reduces makespan compared to the standard PFSP by eliminating
forced idle time between stages. The technique models real-world transfer
batches in high-volume manufacturing, where partially completed batches are
forwarded to the next stage without waiting for the entire lot.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional parameter:** S -- number of equal sublots per job (or sublot sizes
l_1, ..., l_S if unequal splitting is allowed).

**Modified completion time recursion (equal sublots):**
For job pi(k) split into S sublots, each sublot has processing time p[i][pi(k)] / S
on machine i. Sublot s of job pi(k) on machine i completes at:
```
C[i][k][s] = max(C[i-1][k][s], C[i][k][s-1]) + p[i][pi(k)] / S
```
with appropriate boundary conditions across jobs. The key insight is that
sublot transfers allow pipeline parallelism across machines.

When S = 1, the formulation reduces to the standard PFSP.

## Complexity

NP-hard for m >= 3 with discrete (integer) sublots. The continuous relaxation
(fractional sublot sizes) is polynomial for fixed m and S. The problem complexity
depends on whether sublots are equal or variable-sized, and whether S is fixed or
part of the decision.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Does not handle sublot transfers |
| NEH (base heuristic) | Partially | Must evaluate makespan with sublot recursion |
| NEH-LS (variant heuristic) | Yes | NEH adapted with lot-streaming evaluation |
| LPT-LS (variant heuristic) | Yes | Longest processing time with sublot splitting |
| SA (base meta, adapted) | Possible | Neighborhood moves transfer; evaluation changes |
| Continuous relaxation | Possible | LP/closed-form for optimal sublot sizes given sequence |

**No implementation in this directory.** Parent NEH can be adapted by replacing
the makespan evaluation with the sublot-aware recursion. Determining optimal
sublot sizes for a given sequence is a separate subproblem.

## Applications

- High-volume manufacturing with transfer batches
- Printed circuit board assembly
- Semiconductor wafer fabrication (lot splitting)
- Textile production (partial lot transfers between stages)

## Key References

- Trietsch, D. & Baker, K.R. (1993). "Basic Techniques for Lot Streaming" -- [DOI](https://doi.org/10.1287/opre.41.6.1065)
- Potts, C.N. & Baker, K.R. (1989). "Flow Shop Scheduling with Lot Streaming" -- [DOI](https://doi.org/10.1016/0167-6377(89)90041-6)
- Sarin, S.C. & Jaiprakash, P. (2007). "Flow Shop Lot Streaming" [TODO: verify DOI]

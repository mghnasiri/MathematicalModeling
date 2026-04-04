# Distributed Permutation Flow Shop (PFSP Variant)

## What Changes

The **distributed permutation flow shop** (DFm | prmu | Cmax) extends the base
PFSP by introducing F identical factories, each containing the same m-machine
flow shop configuration. Jobs must first be assigned to a factory and then
sequenced within that factory. The overall makespan is the maximum completion
time across all factories. This models multi-plant manufacturing networks
where a central planner must decide both the allocation and sequencing of
production orders across geographically distributed sites.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional parameter:** F -- number of identical factories (each with m machines).

**Additional decision:** Assignment function a: {1,...,n} -> {1,...,F} mapping
each job to a factory.

**Per-factory subproblem:** For each factory f, the jobs assigned to it
(N_f = {j : a(j) = f}) are sequenced as a standard PFSP with completion
time recursion:
```
C_f[i][k] = max(C_f[i-1][k], C_f[i][k-1]) + p[i][pi_f(k)]
```

**Objective:** Minimize the overall makespan:
```
Cmax = max_{f=1..F} C_f[m-1][|N_f|-1]
```

The problem combines a partitioning decision (which factory) with F independent
permutation sequencing decisions (job order within each factory).

## Complexity

NP-hard. The problem generalizes the base PFSP (F = 1 reduces to standard PFSP).
Even balancing load across factories without sequencing is related to the NP-hard
multiprocessor scheduling problem.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| NEH (base heuristic) | Partially | Can sequence each factory, but needs assignment logic |
| NEH-DPFSP (variant heuristic) | Yes | NEH with factory assignment by minimum load |
| Round-Robin + NEH | Yes | Assign jobs cyclically, then sequence per factory |
| SA (base meta, adapted) | Possible | Add inter-factory move to neighborhood |
| IG (base meta, adapted) | Possible | Destroy-repair with reassignment across factories |

**No implementation in this directory.** Parent heuristics (NEH, IG) can be
adapted by wrapping them with a factory-assignment layer.

## Applications

- Multi-plant manufacturing networks
- Distributed assembly lines (automotive, electronics)
- Cloud computing job scheduling across data centers

## Key References

- Naderi, B. & Ruiz, R. (2010). "The Distributed Permutation Flowshop Scheduling Problem" -- [DOI](https://doi.org/10.1016/j.cor.2009.06.005)
- Naderi, B. & Ruiz, R. (2014). "A Scatter Search Algorithm for the Distributed Permutation Flowshop Scheduling Problem" [TODO: verify DOI]
- Fernandez-Viagas, V. & Framinan, J.M. (2015). "A Bounded-Search Iterated Greedy Algorithm for the Distributed Permutation Flowshop Scheduling Problem" [TODO: verify DOI]

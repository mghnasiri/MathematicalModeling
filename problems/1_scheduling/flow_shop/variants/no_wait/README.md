# No-Wait Flow Shop (PFSP Variant)

## What Changes

In the **no-wait permutation flow shop** (Fm | prmu, no-wait | Cmax), each job
must be processed on all m machines in sequence without any idle time between
consecutive operations. Once a job starts on machine 1, it flows through all
machines without interruption. This models production processes where the
material cannot sit idle between stages -- steel continuous casting, chemical
reactions that cannot be paused, and temperature-sensitive food processing.
The objective remains makespan minimization, but the no-wait constraint
transforms the problem structure into an asymmetric TSP on a delay matrix.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**No-wait constraint:** For each job j, processing on machine i+1 starts
immediately after machine i finishes:
```
start(j, i+1) = start(j, i) + p[i][j]    for all i = 0, ..., m-2
```

**Delay-based reformulation:** The constraint reduces the problem to inter-job delays:
```
d(j, k) = max over i of [ cumulative_p(j, 0..i) - cumulative_p(k, 0..i-1) ]
```
where d(j, k) is the minimum start-to-start gap between consecutive jobs j and k.

**Makespan in terms of delays:**
```
Cmax = sum_{i=0}^{n-2} d(pi[i], pi[i+1]) + sum_{i=0}^{m-1} p[i, pi[n-1]]
```

This structure makes the NWFSP equivalent to an **asymmetric TSP** on the
delay matrix D, connecting it to the rich body of TSP algorithms.

## Complexity

| Machines | Complexity | Reference |
|----------|-----------|-----------|
| m = 2 | Polynomial (reducible to TSP on a line) | Gilmore & Gomory (1964) |
| m >= 3 | NP-hard | Roeck (1984) |

The reduction to ATSP means any TSP solver or heuristic can be applied to
the delay matrix. For m = 2, the delay matrix has special structure that
admits a polynomial-time algorithm.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Does not enforce no-wait constraint |
| NEH (base heuristic) | No | Must use delay-based makespan evaluation |
| Nearest Neighbor (variant heuristic) | Yes | Greedy TSP-like on delay matrix |
| NEH-NW (variant heuristic) | Yes | NEH with no-wait makespan evaluation |
| Gangadharan-Rajendran (variant heuristic) | Yes | Slope-index priority with NEH-style insertion |
| IG-NW (variant metaheuristic) | Yes | Iterated Greedy adapted for no-wait |
| TSP metaheuristics (2-opt, SA) | Possible | Apply to delay matrix as ATSP instance |

## Implementations

Python files in this directory:
- `instance.py` -- NoWaitFlowShopInstance, delay matrix computation
- `heuristics.py` -- Nearest Neighbor, NEH-NW, Gangadharan-Rajendran
- `metaheuristics.py` -- Iterated Greedy for no-wait (IG-NW)
- `__init__.py` -- Package init

## Applications

- Steel manufacturing (continuous casting, hot rolling)
- Chemical processing (reactions that cannot be interrupted)
- Food processing (temperature-sensitive products)
- Pharmaceutical production (time-critical chemical synthesis)

## Key References

- Gilmore, P.C. & Gomory, R.E. (1964). "Sequencing a One State-Variable Machine"
- Gangadharan, R. & Rajendran, C. (1993). "Heuristic Algorithms for Scheduling in the No-Wait Flowshop"
- Bertolissi, E. (2000). "Heuristic Algorithm for Scheduling in the No-Wait Flow-Shop"
- Pan, Q.-K. et al. (2008). "A Discrete Differential Evolution Algorithm for the PFSP"

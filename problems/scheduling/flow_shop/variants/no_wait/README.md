# No-Wait Flow Shop Scheduling Problem (NWFSP)

## Problem Definition

In the **no-wait permutation flow shop** (Fm | prmu, no-wait | Cmax), each job
must be processed on all m machines in sequence without any waiting time between
consecutive operations. Once a job starts on the first machine, it proceeds
through all machines without interruption.

### Mathematical Formulation

**Given:**
- n jobs, m machines
- Processing times p[i][j] for job j on machine i

**No-wait constraint:** For each job j, processing on machine i+1 must start
immediately after processing on machine i finishes:
```
start(j, i+1) = start(j, i) + p[i][j]    for all i = 0, ..., m-2
```

**Objective:** Minimize the makespan (completion time of the last job on the
last machine).

### Delay-Based Formulation

The no-wait constraint reduces the problem to finding the optimal permutation
where makespan depends on **inter-job delays**:

```
d(j, k) = max over i of [ cumulative_p(j, 0..i) - cumulative_p(k, 0..i-1) ]
```

where d(j, k) is the minimum time gap between starting job j and starting job k.

**Makespan:**
```
Cmax = sum_{i=0}^{n-2} d(pi[i], pi[i+1]) + sum_{i=0}^{m-1} p[i, pi[n-1]]
```

This structure makes the NWFSP equivalent to an **asymmetric Traveling Salesman
Problem (ATSP)** on the delay matrix, connecting it to a rich body of TSP
literature and algorithms.

## Complexity

| Machines | Complexity | Reference |
|----------|-----------|-----------|
| m = 2 | Polynomial (reducible to TSP on a line) | Gilmore & Gomory (1964) |
| m >= 3 | NP-hard | Roeck (1984) |

## Implemented Algorithms

### Heuristics
- **Nearest Neighbor (NN):** Greedy TSP-like construction using delay matrix
- **NEH-NW:** NEH adapted with no-wait makespan evaluation
- **Gangadharan-Rajendran:** Slope-index priority with NEH-style insertion

### Metaheuristics
- **Iterated Greedy (IG-NW):** Ruiz-Stuetzle IG adapted for no-wait

## Applications

- Steel manufacturing (continuous casting)
- Chemical processing (reactions that cannot be interrupted)
- Food processing (temperature-sensitive products)
- Pharmaceutical production

## Key References

- Gilmore, P.C. & Gomory, R.E. (1964). "Sequencing a One State-Variable Machine"
- Gangadharan, R. & Rajendran, C. (1993). "Heuristic Algorithms for Scheduling in the No-Wait Flowshop"
- Bertolissi, E. (2000). "Heuristic Algorithm for Scheduling in the No-Wait Flow-Shop"
- Pan, Q.-K. et al. (2008). "A Discrete Differential Evolution Algorithm for the PFSP"

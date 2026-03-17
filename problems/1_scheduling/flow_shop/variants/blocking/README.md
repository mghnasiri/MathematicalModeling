# Blocking Flow Shop Scheduling Problem (BFSP)

## Problem Definition

In the **blocking permutation flow shop** (Fm | prmu, blocking | Cmax), there
are no intermediate storage buffers between machines. When a job finishes
processing on machine i, it remains on that machine (blocking it) until
machine i+1 becomes available. This prevents the next job from starting
on machine i.

### Mathematical Formulation

**Given:**
- n jobs, m machines
- Processing times p[i][j] for job j on machine i

**Blocking constraint:** A job must stay on machine i until machine i+1 is
free. We track both completion times C and departure times D:

```
C[i][k] = max(C[i-1][k], D[i][k-1]) + p[i][pi(k)]
D[i][k] = max(C[i][k], D[i+1][k])    for i < m-1
D[m-1][k] = C[m-1][k]                  (no blocking on last machine)
```

**Objective:** Minimize Cmax = C[m-1][n-1]

### Relationship to Standard PFSP

- Standard PFSP: C[i][k] = max(C[i-1][k], **C[i][k-1]**) + p[i][pi(k)]
- Blocking PFSP: C[i][k] = max(C[i-1][k], **D[i][k-1]**) + p[i][pi(k)]

The departure time D[i][k-1] >= C[i][k-1], so blocking makespan is always
greater than or equal to the standard PFSP makespan for the same permutation.

## Complexity

| Machines | Complexity | Reference |
|----------|-----------|-----------|
| m = 2 | Polynomial | Gilmore & Gomory (1964) |
| m >= 3 | NP-hard | Hall & Sriskandarajah (1996) |

## Implemented Algorithms

### Heuristics
- **NEH-B:** NEH adapted with blocking-aware makespan evaluation
- **Profile Fitting (PF-B):** Greedy construction minimizing blocking time

### Metaheuristics
- **Iterated Greedy (IG-B):** Ruiz-Stuetzle IG adapted for blocking

## Applications

- Manufacturing lines with limited buffer space
- Robotic cells (robot transfers between machines)
- Paint shops (parts must move immediately to avoid drying)
- Concrete production (material must be poured immediately)

## Key References

- Hall, N.G. & Sriskandarajah, C. (1996). "A Survey of Machine Scheduling Problems with Blocking and No-Wait in Process"
- Ronconi, D.P. (2004). "A Note on Constructive Heuristics for the Flowshop Problem with Blocking"
- Grabowski, J. & Pempera, J. (2007). "The Permutation Flow Shop Problem with Blocking. A Tabu Search Approach"
- Ribas, I. et al. (2011). "An Iterated Greedy Algorithm for the Flowshop Scheduling Problem with Blocking"

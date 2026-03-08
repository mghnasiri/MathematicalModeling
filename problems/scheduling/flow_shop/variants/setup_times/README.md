# Sequence-Dependent Setup Times Flow Shop (SDST)

## Problem Definition

**Notation**: Fm | prmu, Ssd | Cmax

The SDST flow shop extends the standard permutation flow shop by introducing
setup times that depend on both the current and preceding job on each machine.

### Mathematical Formulation

**Given**:
- n jobs, m machines
- Processing times p[i][j]: time for job j on machine i
- Setup times s[i][j][k]: setup on machine i when switching from job j to job k
- Initial setup s[i][n][k]: setup on machine i for first job k (no predecessor)

**Completion time recursion** (anticipatory setups):

```
C[i][π(k)] = max(C[i-1][π(k)], C[i][π(k-1)] + s[i][π(k-1)][π(k)]) + p[i][π(k)]
```

**Objective**: Minimize Cmax = C[m-1][π(n-1)]

### Complexity

NP-hard for m >= 2 (reduces to asymmetric TSP when m=1).

## Applications

- **Printing industry**: Color changeovers between print jobs
- **Chemical processing**: Cleaning/purging between different products
- **Automotive manufacturing**: Tooling changes between different models
- **Semiconductor fabrication**: Recipe changes between wafer types
- **Food processing**: Sanitization between different food products

## Algorithms

### Constructive Heuristics

| Algorithm | Description | Complexity |
|-----------|-------------|------------|
| NEH-SDST | NEH with setup-aware workload sorting and evaluation | O(n^2 * m) |
| GRASP-SDST | Randomized greedy with local search | O(iterations * n^2 * m) |

### Metaheuristics

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| IG-SDST | Iterated Greedy with SDST-aware evaluation | Ruiz et al. (2005) |

## Key References

- Ruiz, R., Maroto, C. & Alcaraz, J. (2005). "Solving the Flowshop Scheduling
  Problem with Sequence Dependent Setup Times Using Advanced Metaheuristics"
  — [DOI](https://doi.org/10.1016/j.ejor.2004.01.022)

- Allahverdi, A. et al. (2008). "A Survey of Scheduling Problems with Setup
  Times or Costs" — [DOI](https://doi.org/10.1016/j.ejor.2006.06.060)

- Rios-Mercado, R.Z. & Bard, J.F. (1998). "Computational Experience with a
  Branch-and-Cut Algorithm for Flowshop Scheduling with Setups"
  — [DOI](https://doi.org/10.1016/S0305-0548(97)00079-8)

# Sequence-Dependent Setup Times Flow Shop (PFSP Variant)

## What Changes

The **SDST flow shop** (Fm | prmu, Ssd | Cmax) extends the standard permutation
flow shop by introducing setup times that depend on both the current and preceding
job on each machine. In real-world manufacturing, changeover operations -- such as
cleaning chemical reactors, swapping printing plates, or retooling assembly lines --
vary in duration depending on which product was processed previously. The objective
remains makespan minimization, but the sequence-dependent setups make evaluating any
given permutation more expensive and eliminate some structural properties of the base
problem.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional parameters:**
- s[i][j][k]: setup time on machine i when switching from job j to job k
- s[i][0][k]: initial setup on machine i for the first job k (no predecessor)

**Modified completion time recursion (anticipatory setups):**
```
C[i][pi(k)] = max(C[i-1][pi(k)], C[i][pi(k-1)] + s[i][pi(k-1)][pi(k)]) + p[i][pi(k)]
```

The setup s[i][pi(k-1)][pi(k)] is added after the previous job completes on
machine i and before the current job begins processing. In the base PFSP, this
term is absent (equivalently, s = 0 everywhere).

## Complexity

NP-hard for m >= 2. When m = 1, the problem reduces to asymmetric TSP on the
setup-time matrix. The addition of sequence-dependent setups makes the problem
harder than the base PFSP: even the 2-machine case is NP-hard, whereas the base
F2 || Cmax is solved in O(n log n) by Johnson's Rule.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Cannot incorporate setup times |
| NEH (base heuristic) | No | Must use setup-aware evaluation and workload sorting |
| NEH-SDST (variant heuristic) | Yes | NEH with setup-aware workload and evaluation, O(n^2 * m) |
| GRASP-SDST (variant heuristic) | Yes | Randomized greedy with local search |
| IG-SDST (variant metaheuristic) | Yes | Iterated Greedy with SDST-aware evaluation |
| SA (base meta, adapted) | Possible | Swap/insertion moves transfer; evaluation must include setups |
| GA (base meta, adapted) | Possible | Encoding unchanged; fitness must include setup costs |

## Implementations

Python files in this directory:
- `instance.py` -- SDSTFlowShopInstance, setup time matrices, makespan with setups
- `heuristics.py` -- NEH-SDST, GRASP-SDST
- `metaheuristics.py` -- Iterated Greedy for SDST (IG-SDST)
- `__init__.py` -- Package init

## Applications

- **Printing industry:** Color changeovers between print jobs
- **Chemical processing:** Cleaning/purging between different products
- **Automotive manufacturing:** Tooling changes between different models
- **Semiconductor fabrication:** Recipe changes between wafer types
- **Food processing:** Sanitization between different food products

## Key References

- Ruiz, R., Maroto, C. & Alcaraz, J. (2005). "Solving the Flowshop Scheduling Problem with Sequence Dependent Setup Times Using Advanced Metaheuristics" -- [DOI](https://doi.org/10.1016/j.ejor.2004.01.022)
- Allahverdi, A. et al. (2008). "A Survey of Scheduling Problems with Setup Times or Costs" -- [DOI](https://doi.org/10.1016/j.ejor.2006.06.060)
- Rios-Mercado, R.Z. & Bard, J.F. (1998). "Computational Experience with a Branch-and-Cut Algorithm for Flowshop Scheduling with Setups" -- [DOI](https://doi.org/10.1016/S0305-0548(97)00079-8)

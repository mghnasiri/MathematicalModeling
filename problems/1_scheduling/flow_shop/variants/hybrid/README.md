# Hybrid Flow Shop (PFSP Variant)

## What Changes

The **hybrid flow shop** (HFm | prmu | Cmax), also called flexible flow shop,
extends the permutation flow shop by providing multiple identical parallel
machines at one or more stages. Each job visits all stages in sequence (like
standard PFSP), but at each stage it can be processed on any available machine.
This models production environments where bottleneck stages are alleviated by
adding parallel capacity -- semiconductor fabs, food canning lines, and
pharmaceutical packaging. The objective remains makespan minimization, but the
problem now involves both sequencing (job order) and machine assignment at each
stage.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional parameter:** m_i -- number of identical parallel machines at stage i.

**Additional decision:** For each job j at stage i, assign it to one of the
m_i available machines.

**Modified completion time recursion:**
```
C[i][j] = max(C[i-1][j], earliest_available(stage i)) + p[i][j]
```
where `earliest_available(stage i)` is the earliest time any of the m_i machines
at stage i becomes free.

When m_i = 1 for all stages, the problem reduces to the standard PFSP.

## Complexity

NP-hard even for 2 stages where one stage has 1 machine and the other has 2
(Gupta, 1988). The hybrid flow shop generalizes both the PFSP (all m_i = 1)
and the parallel machine problem (single stage with m_i > 1).

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Does not handle parallel machines at stages |
| NEH (base heuristic) | Partially | Sequencing logic transfers; machine assignment needed |
| NEH-HFS (variant heuristic) | Yes | NEH with earliest-machine assignment at each stage |
| LPT-HFS (variant heuristic) | Yes | Longest processing time with stage-level dispatching |
| SPT-HFS (variant heuristic) | Yes | Shortest processing time dispatching |
| SA (base meta, adapted) | Possible | Add machine reassignment to neighborhood |
| GA (base meta, adapted) | Possible | Two-layer encoding: sequence + machine assignment |

**No implementation in this directory.** Parent NEH and metaheuristic frameworks
can be adapted by adding a machine-assignment decoder at each stage.

## Applications

- Semiconductor fabrication (multi-tool stages)
- Food canning and packaging lines
- Pharmaceutical manufacturing
- Textile production (dyeing, weaving with parallel looms)

## Key References

- Gupta, J.N.D. (1988). "Two-Stage Hybrid Flowshop Scheduling Problem" -- [DOI](https://doi.org/10.1057/jors.1988.63)
- Linn, R. & Zhang, W. (1999). "Hybrid Flow Shop Scheduling: A Survey" -- [DOI](https://doi.org/10.1016/S0305-0548(98)00023-6)
- Ruiz, R. & Vazquez-Rodriguez, J.A. (2010). "The Hybrid Flow Shop Scheduling Problem" -- [DOI](https://doi.org/10.1016/j.ejor.2009.09.024)

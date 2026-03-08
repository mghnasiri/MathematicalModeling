# Uncapacitated Facility Location Problem (UFLP)

## Problem Definition

Given $m$ potential facility sites with opening costs $f_i$ and $n$ customers, with assignment costs $c_{ij}$ for serving customer $j$ from facility $i$, select facilities to open and assign customers to minimize total fixed + assignment cost.

$$\min \sum_{i \in S} f_i + \sum_{j=1}^{n} \min_{i \in S} c_{ij}$$

where $S \subseteq \{1, \ldots, m\}$ is the set of open facilities.

## Complexity

NP-hard. Best known approximation: 1.488 (Li, 2013).

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Greedy Add | Heuristic | Iteratively open most cost-reducing facility |
| Greedy Drop | Heuristic | Start all open, drop least impactful |
| Simulated Annealing | Metaheuristic | Toggle/swap moves with Boltzmann acceptance |

## Key References

- Cornuéjols, G. et al. (1977). Location of bank accounts to optimize float. *Management Science*, 23(8), 789-810. https://doi.org/10.1287/mnsc.23.8.789
- Li, S. (2013). A 1.488 approximation for UFLP. *Inf. Comput.*, 222, 45-58. https://doi.org/10.1016/j.ic.2012.01.007
- Krarup, J. & Pruzan, P.M. (1983). The simple plant location problem. *EJOR*, 12(1), 36-81. https://doi.org/10.1016/0377-2217(83)90181-9

# Chance-Constrained Facility Location

## Problem Definition

Open facilities and assign customers to minimize total cost (fixed + assignment), subject to chance constraints on facility capacities under stochastic demand:

$$P\left(\sum_{j \in S_i} D_j \leq C_i\right) \geq 1 - \alpha, \quad \forall i$$

## Solution Approaches

| Method | Type | Description |
|--------|------|-------------|
| Greedy Open | Heuristic | Iteratively open cost-reducing facilities with CC checks |
| Mean-Demand Greedy | Heuristic | Deterministic proxy using expected demands |
| Simulated Annealing | Metaheuristic | Toggle/swap facilities with violation penalty |

## Key References

- Bertsimas, D. & Sim, M. (2004). The price of robustness. *Oper. Res.*, 52(1), 35-53. https://doi.org/10.1287/opre.1030.0065
- Snyder, L.V. (2006). Facility location under uncertainty. *IIE Trans.*, 38(7), 547-564. https://doi.org/10.1080/07408170500216480

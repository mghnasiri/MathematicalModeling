# Stochastic Vehicle Routing Problem (SVRP)

## Problem Definition

Extends CVRP with **stochastic customer demands**. Routes are designed a priori; if a vehicle overflows mid-route, it incurs a recourse cost (return to depot).

## Approaches

| Method | Type | Description |
|--------|------|-------------|
| Chance-Constrained CW | Heuristic | Clarke-Wright savings with P(overflow) <= alpha check |
| Mean-Demand Savings | Heuristic | CW with expected demands as proxy |
| Simulated Annealing | Metaheuristic | Relocate/swap/2-opt with recourse penalty |

## Key References

- Bertsimas, D.J. (1992). A vehicle routing problem with stochastic demand. *Oper. Res.*, 40(3), 574-585.
- Gendreau, M., Laporte, G. & Séguin, R. (1996). Stochastic vehicle routing. *EJOR*, 88(1), 3-12.

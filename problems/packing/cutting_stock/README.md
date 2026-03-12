# 1D Cutting Stock Problem (CSP)

## Problem Definition

Given stock material of length $L$ and $m$ item types, each with length $l_i$ and demand $d_i$, determine how to cut stock rolls to satisfy all demands using the minimum number of rolls.

## Mathematical Formulation (Gilmore-Gomory)

Let $\mathcal{P}$ be the set of all feasible cutting patterns, where pattern $p$ cuts $a_{ip}$ pieces of type $i$.

$$\min \sum_{p \in \mathcal{P}} x_p$$

$$\text{s.t.} \quad \sum_{p \in \mathcal{P}} a_{ip} x_p \geq d_i \quad \forall i, \quad x_p \in \mathbb{Z}_{\geq 0}$$

The LP relaxation is solved via column generation; each pricing subproblem is a bounded knapsack problem.

## Complexity

NP-hard (reduces from Bin Packing). The LP relaxation satisfies the Integer Round-Up Property (IRUP): $OPT_{IP} \leq \lceil OPT_{LP} \rceil + 1$ for most practical instances.

## Solution Approaches

| Method | Description |
|--------|-------------|
| Greedy (largest-first) | For each roll, greedily pack items by decreasing length |
| FFD-based | Expand demands to individual items, apply FFD, aggregate patterns |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [2D Cutting Stock (2D-CSP)](variants/two_dimensional/) | `variants/two_dimensional/` | Cut rectangular items from 2D stock sheets |

## Key References

- Gilmore, P.C. & Gomory, R.E. (1961). A linear programming approach to the cutting-stock problem. *Oper. Res.*, 9(6), 849-859. https://doi.org/10.1287/opre.9.6.849
- Gilmore, P.C. & Gomory, R.E. (1963). A linear programming approach — Part II. *Oper. Res.*, 11(6), 863-888. https://doi.org/10.1287/opre.11.6.863
- Wäscher, G. et al. (2007). An improved typology of cutting and packing problems. *EJOR*, 183(3), 1109-1130. https://doi.org/10.1016/j.ejor.2005.12.047

# p-Median Problem (PMP)

## Problem Definition

Given $n$ demand points and $m$ candidate facility locations, open exactly $p$ facilities and assign each customer to its nearest open facility to minimize total weighted distance.

$$\min \sum_{j=1}^{n} w_j \min_{i \in S} d_{ij}$$

subject to $|S| = p$, $S \subseteq \{1, \ldots, m\}$.

## Complexity

NP-hard for general $p$ (Kariv & Hakimi, 1979).

## Solution Approaches

| Method | Complexity | Description |
|--------|-----------|-------------|
| Greedy | $O(m \cdot p \cdot n)$ | Iteratively add most cost-reducing facility |
| Teitz-Bart Interchange | $O(m \cdot p \cdot n \cdot \text{iter})$ | Swap open/closed facilities until no improvement |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Capacitated p-Median (CPMP)](variants/capacitated/) | `variants/capacitated/` | Facilities have limited capacity; customer demands must not exceed it |

## Key References

- Hakimi, S.L. (1964). Optimum locations of switching centers. *Oper. Res.*, 12(3), 450-459. https://doi.org/10.1287/opre.12.3.450
- Teitz, M.B. & Bart, P. (1968). Heuristic methods for estimating the generalized vertex median. *Oper. Res.*, 16(5), 955-961. https://doi.org/10.1287/opre.16.5.955
- Reese, J. (2006). Solution methods for the p-median problem. *Networks*, 48(3), 125-142. https://doi.org/10.1002/net.20128

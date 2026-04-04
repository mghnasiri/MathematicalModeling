# Bounded Knapsack Problem (Knapsack Variant)

## What Changes

The standard 0-1 Knapsack allows at most one copy of each item. The **Bounded Knapsack Problem (BKP)** allows **multiple copies** of each item type, up to an upper bound b_i. This models scenarios where items are available in limited stock rather than being unique.

- **Decision variable** changes from binary x_i in {0,1} to integer x_i in {0,...,b_i}.
- **Intermediate** between 0-1 Knapsack (b_i = 1 for all i) and Unbounded Knapsack (b_i = infinity).
- **DP state space** remains O(n*W) but transitions must consider all quantities 0..b_i per item.
- **Greedy heuristics** naturally extend: take as many copies as the bound allows before moving to the next item.

**Real-world motivation**: investment allocation with share purchase limits, cargo loading with limited stock of each product, production planning with bounded raw material availability, purchasing decisions with quantity caps.

## Mathematical Formulation

Extends 0-1 Knapsack with integer quantities and bounds:

```
max  sum_i v_i * x_i
s.t. sum_i w_i * x_i <= W                 (capacity)
     0 <= x_i <= b_i                       (bounded copies)
     x_i integer                           (integrality)
```

Where b_i is the maximum number of copies of item type i. When b_i = 1 for all i, this reduces to the standard 0-1 Knapsack. When all b_i are unbounded, it becomes the Unbounded Knapsack Problem.

## Complexity

- **NP-hard** (weakly): reduces to 0-1 Knapsack when all b_i = 1.
- Admits **O(n * W) pseudo-polynomial DP**, same complexity class as 0-1 Knapsack.
- Can be solved by binary expansion (converting each bounded item into O(log b_i) binary items) and applying standard 0-1 Knapsack algorithms, though direct BKP DP is more efficient.
- The LP relaxation provides the same fractional bound as 0-1 Knapsack but with potentially fractional quantities.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy value-density | Yes | Sort by v_i/w_i, take max copies that fit; O(n log n). |
| Dynamic Programming | Yes | Exact O(n * W) with quantity tracking per item type. |
| Binary expansion + 0-1 DP | Yes | Convert to O(n * sum(log b_i)) binary items, apply standard DP. |
| Simulated Annealing | Yes | Integer-vector encoding: increment/decrement random item quantity. |
| LP relaxation + rounding | Possible | Fractional solution rounded down; gap at most one item. Not implemented. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `BKPInstance` (values, weights, bounds, capacity), `BKPSolution` (quantities, total_value, total_weight), `validate_solution()`, `small_bkp_5()` benchmark |
| `heuristics.py` | `greedy_density()` (value-density greedy with bounds), `dynamic_programming()` (exact O(n*W) DP with bound constraints) |
| `metaheuristics.py` | `simulated_annealing()` with integer-vector encoding, increment/decrement neighborhood |
| `tests/test_bkp.py` | Test suite covering bound enforcement, DP optimality, greedy quality |

## Relationship to Base Knapsack

When all bounds b_i = 1, the BKP reduces to the standard 0-1 Knapsack. When all bounds are sufficiently large (b_i >= floor(W/w_i)), it becomes the Unbounded Knapsack. The BKP sits between these two extremes, and its DP formulation naturally extends the 0-1 case by considering multiple copies per item in each state transition.

## Key References

- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.
- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
- Pisinger, D. (2000). A minimal algorithm for the bounded knapsack problem. *INFORMS Journal on Computing*, 12(1), 75-82. [TODO: verify DOI]

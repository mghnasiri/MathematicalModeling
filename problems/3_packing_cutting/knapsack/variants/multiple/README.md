# Multiple Knapsack Problem (Knapsack Variant)

## What Changes

The standard 0-1 Knapsack has **one** knapsack. The **Multiple Knapsack Problem (mKP)** has **k knapsacks**, each with its own capacity. Items must be assigned to at most one knapsack (or left out entirely) to maximize total packed value.

- **Assignment decision** added: each item is assigned to a specific knapsack or skipped, not just included/excluded.
- **k capacity constraints**: one per knapsack, each independently enforced.
- **Decision variable** changes from binary x_i to integer a_i in {-1, 0, ..., k-1}, where -1 means unassigned and 0..k-1 identify the knapsack.
- Generalizes both 0-1 Knapsack (k=1) and Bin Packing (uniform values, minimize bins used).

**Real-world motivation**: cargo loading across multiple vehicles with different weight limits, budget allocation across multiple funding sources, cloud computing task assignment to servers with resource limits, portfolio distribution across investment accounts with different constraints.

## Mathematical Formulation

Extends 0-1 Knapsack with k knapsacks:

```
max  sum_j v_j * y_j
s.t. sum_{j: a_j = i} w_j <= C_i    for all i in {0,...,k-1}   (per-knapsack capacity)
     a_j in {-1, 0, ..., k-1}       for all j                  (assignment or skip)
     y_j = 1 if a_j >= 0, else 0                                (selection indicator)
```

Where C_i is the capacity of knapsack i, and a_j is the assignment of item j.

## Complexity

- **NP-hard**: generalizes 0-1 Knapsack (which is the case k = 1).
- Even with k = 2 and identical capacities, the problem remains NP-hard (reduces from PARTITION).
- Pseudo-polynomial algorithms exist but scale as O(n * product of capacities).

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy value-density | Yes | Sort by v_j/w_j, assign to first knapsack with room; O(n log n + n*k). |
| Greedy best-fit | Yes | Sort by v_j/w_j, assign to knapsack with tightest fit; reduces waste. |
| Separate 0-1 DP per knapsack | Limited | Ignores inter-knapsack competition; serves as upper bound. |
| Genetic Algorithm | Yes | Integer-vector encoding gene[j] in {-1,...,k-1}, repair removes lowest-density items from overloaded knapsacks. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `MultipleKnapsackInstance` (n, k, values, weights, capacities), `MultipleKnapsackSolution` (assignments, total_value), `validate_solution()` with per-knapsack capacity checking |
| `heuristics.py` | `greedy_value_density()` (first-fit by v/w), `greedy_best_fit()` (tightest-fit by v/w) |
| `metaheuristics.py` | `genetic_algorithm()` with integer-vector encoding, uniform crossover, repair operator for capacity violations |
| `tests/test_mkp_multi.py` | Test suite covering assignment validation, capacity enforcement, greedy and GA quality |

## Key References

- Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
- Pisinger, D. (1999). An exact algorithm for large multiple knapsack problems. *European Journal of Operational Research*, 114(3), 528-541.
- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.

# Subset Sum Problem (Knapsack Variant)

## What Changes

The standard 0-1 Knapsack has distinct values and weights per item. The **Subset Sum Problem (SSP)** is a special case where **values equal weights** (or equivalently, there are no separate values): given n positive integers and a target T, find a subset summing to exactly T (decision version) or as close to T as possible without exceeding it (optimization version).

- **No separate value dimension**: the "value" of including an element is its weight itself.
- **Decision version** asks a yes/no question: does a subset summing to exactly T exist?
- **Optimization version** maximizes the subset sum subject to sum <= T.
- Fundamental building block: many NP-completeness reductions go through Subset Sum.

**Real-world motivation**: splitting expenses evenly among groups, cryptographic knapsack schemes, load balancing across processors (equal-weight partition), ballot counting verification, memory allocation to exact size requirements.

## Mathematical Formulation

Special case of 0-1 Knapsack where v_i = w_i:

```
Decision:
  exists? x in {0,1}^n  such that  sum_i a_i * x_i = T

Optimization:
  max  sum_i a_i * x_i
  s.t. sum_i a_i * x_i <= T
       x_i in {0, 1}    for all i
```

Where a_i are the positive integer elements and T is the target sum.

## Complexity

- **NP-complete** (decision version): one of Karp's 21 NP-complete problems (1972).
- **Weakly NP-hard**: admits O(n * T) pseudo-polynomial DP, which is efficient when T is polynomially bounded.
- The PARTITION problem (T = sum/2) is a special case, also NP-complete.
- Approximation schemes (FPTAS) exist with (1-epsilon) guarantee in O(n / epsilon) time.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Greedy largest-first | Yes | Sort descending, take elements that fit; O(n log n). No approximation guarantee. |
| Dynamic Programming | Yes | Exact O(n * T) pseudo-polynomial; practical for moderate T. |
| Meet-in-the-middle | Possible | Split into two halves, enumerate 2^(n/2) each, merge; O(2^(n/2)). Not implemented. |
| Simulated Annealing | Yes | Add/remove/swap neighborhood on binary selection vector. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `SubsetSumInstance` (values as integer array, target), `SubsetSumSolution` (selected indices, achieved_sum, exact_match flag), `validate_solution()` |
| `heuristics.py` | `greedy_largest_first()` (descending greedy), `dynamic_programming()` (exact O(n*T) DP with backtracking) |
| `metaheuristics.py` | `simulated_annealing()` with add/remove/swap neighborhood, penalty for exceeding target |
| `tests/test_ssp.py` | Test suite covering exact-match detection, DP optimality, greedy quality on various instances |

## Relationship to Base Knapsack

When all item values equal their weights (v_i = w_i), the 0-1 Knapsack reduces to the optimization form of Subset Sum. Conversely, Subset Sum can be seen as the simplest non-trivial Knapsack variant, and many NP-completeness reductions use Subset Sum as the source problem.

## Key References

- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability*. W.H. Freeman.
- Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*. Springer.
- Koiliaris, K. & Xu, C. (2019). A faster pseudopolynomial time algorithm for subset sum. *ACM Transactions on Algorithms*, 15(3), 1-20. [TODO: verify DOI]

# Multi-dimensional Knapsack Problem (Knapsack Variant)

## What Changes

The standard 0-1 Knapsack has a **single** capacity constraint. The **Multi-dimensional Knapsack Problem (MKP)** introduces **d capacity constraints** simultaneously: each item consumes resources across d dimensions, and the selection must respect all d capacity limits.

- **d weight vectors** per item: w_ij is the consumption of item i in dimension j.
- **d capacity constraints**: the selection must satisfy sum_i w_ij * x_i <= W_j for all j.
- **Pseudo-polynomial DP does not extend**: the 1D O(n*W) DP becomes impractical for d >= 2 since the state space grows as O(W_1 * W_2 * ... * W_d).
- **LP relaxation** is tighter than for 1D, but the integrality gap can be large.

**Real-world motivation**: capital budgeting (projects consume budget, labor, and time), cargo loading (items have weight, volume, and fragility constraints), cloud computing resource allocation (tasks require CPU, memory, and bandwidth), project selection with multi-department budget constraints.

## Mathematical Formulation

Extends 0-1 Knapsack with d capacity dimensions:

```
max  sum_i v_i * x_i
s.t. sum_i w_ij * x_i <= W_j    for all j in {1,...,d}   (d capacity constraints)
     x_i in {0, 1}              for all i in {1,...,n}    (binary selection)
```

The key structural difference: each item has a d-dimensional weight vector (w_i1, ..., w_id) rather than a single scalar weight. The weights matrix has shape (d, n) where d is the number of resource dimensions.

## Complexity

- **NP-hard** (strongly for d >= 2): the pseudo-polynomial DP for 1D does not generalize efficiently.
- LP relaxation can be solved in polynomial time, providing upper bounds for branch-and-bound.
- The problem becomes significantly harder as d increases; practical exact methods rely on branch-and-bound with LP bounds.
- Standard OR benchmark library: Chu & Beasley (1998) instances with n up to 500 and d up to 30.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Pseudo-utility greedy | Yes | Sort by v_i / (sum of normalized weights across dimensions); O(n log n + n*d). |
| Max-value greedy | Yes | Sort by value alone, take feasible items; simple baseline. |
| LP relaxation bound | Yes | Solve LP relaxation for upper bound; useful in B&B. |
| Genetic Algorithm | Yes | Binary encoding, uniform crossover, repair operator removes infeasible items by worst pseudo-utility ratio. |
| Surrogate relaxation | Possible | Combine d constraints into a single surrogate; reduces to 1D. Not implemented. |
| Branch-and-bound | Possible | LP relaxation bounds at each node; effective for moderate n. Not implemented. |

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `MKPInstance` (n, d, values, weights matrix, capacities), `MKPSolution` (selected items, total_value), `validate_solution()` with multi-constraint checking, `small_mkp_5x2()` benchmark |
| `heuristics.py` | `pseudo_utility_greedy()` (Pirkul-style normalized ratio), `max_value_greedy()` (value-only ordering) |
| `metaheuristics.py` | `genetic_algorithm()` with binary encoding, uniform crossover, bit-flip mutation, pseudo-utility repair operator |
| `tests/test_mkp.py` | Test suite covering multi-constraint feasibility, greedy quality, GA convergence |

## Relationship to Base Knapsack

When d = 1, the MKP reduces to the standard 0-1 Knapsack with its O(n*W) pseudo-polynomial DP. For d >= 2, the DP state space explodes multiplicatively across dimensions, making pseudo-polynomial algorithms impractical and shifting the problem to the strongly NP-hard class.

## Key References

- Chu, P.C. & Beasley, J.E. (1998). A genetic algorithm for the multidimensional knapsack problem. *Journal of Heuristics*, 4(1), 63-86.
- Pirkul, H. (1987). A heuristic solution procedure for the multiconstraint zero-one knapsack problem. *Naval Research Logistics*, 34(2), 161-172.
- Freville, A. (2004). The multidimensional 0-1 knapsack problem: An overview. *European Journal of Operational Research*, 155(1), 1-21.
- Loulou, R. & Michaelides, E. (1979). New greedy-like heuristics for the multidimensional 0-1 knapsack problem. *Operations Research*, 27(6), 1101-1114.

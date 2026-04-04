# Maximum Weight Matching (Assignment Variant)

## What Changes

In the **maximum weight matching** variant, the problem relaxes two constraints
of the base LAP: (1) the two sets can have different sizes (n workers, m tasks
where n != m), and (2) not all entities need to be matched -- only a subset of
edges is selected to maximize total weight. The base LAP requires a perfect
matching in a balanced bipartite graph minimizing cost; max-weight matching
seeks the most valuable partial matching in a possibly unbalanced graph.

This models scenarios where assignments are optional and value-driven rather
than mandatory -- matching reviewers to papers (not all papers need every
reviewer), job applicant shortlisting (more applicants than positions), and
organ donor-recipient matching where compatibility varies.

## Mathematical Formulation

The base LAP formulation is relaxed from equality to inequality constraints:

```
max  sum_i sum_j  w_ij * x_ij
s.t. sum_j  x_ij <= 1        for all i = 1..n  (each worker matched at most once)
     sum_i  x_ij <= 1        for all j = 1..m  (each task matched at most once)
     x_ij in {0,1}
```

where w_ij >= 0 is the weight (value) of matching worker i to task j.
Zero-weight edges represent infeasible or undesirable pairings.

**Reduction to LAP:** Pad the smaller set with dummy nodes (zero-weight edges)
to create a balanced n' x n' matrix, then solve as a maximization LAP.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Max-weight bipartite matching | O(n^3) | Kuhn (1955), via augmented matrix |
| Greedy matching | O(E log E) | -- |
| Base LAP (min cost, balanced) | O(n^3) | Kuhn (1955) |

Polynomial. The Hungarian method applies after padding the weight matrix to
square form and negating weights (converting max to min). The greedy approach
is faster but does not guarantee optimality.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Hungarian (base LAP exact) | Yes | Pad to square, negate weights for maximization |
| Greedy max-weight (variant) | Yes | Sort edges by weight, add if no conflict |

## Implementations

Python files in this directory:
- `instance.py` -- MaxMatchingInstance, non-square weight matrix
- `heuristics.py` -- Greedy max-weight matching, Hungarian via augmentation
- `tests/test_maxmatch.py` -- 16 tests

## Applications

- Paper-reviewer assignment (maximize expertise matching)
- Job applicant shortlisting (select best candidate-position pairs)
- Organ donor-recipient matching (maximize compatibility)
- Ad placement (maximize click-through value across ad slots)

## Key References

- Kuhn, H.W. (1955). "The Hungarian method for the assignment problem." Naval Research Logistics Quarterly 2(1-2), 83-97.
- Munkres, J. (1957). "Algorithms for the assignment and transportation problems." J. SIAM 5(1), 32-38.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). Assignment Problems. SIAM.

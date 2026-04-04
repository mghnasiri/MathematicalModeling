# Variable-Size Bin Packing (BPP Variant)

## What Changes

In the **variable-size bin packing problem** (VS-BPP), bins are no longer
identical. Instead, K bin types are available, each with its own capacity C_k
and cost c_k. The objective shifts from minimizing the number of bins to
minimizing total bin cost. This models scenarios where resource containers
come in different sizes at different price points -- choosing between VM
instance types in cloud computing, selecting container sizes for shipping,
or allocating memory pages of varying granularity.

The key structural difference from standard BPP is the joint optimization of
bin type selection and item assignment. Even if fewer large bins suffice,
using many small bins may be cheaper depending on the cost structure.

## Mathematical Formulation

The base BPP formulation gains bin-type selection variables:

```
min  sum_k  c_k * y_k            (minimize total bin cost)
s.t. sum_k  x_ijk = 1            for all i  (each item in exactly one bin)
     sum_i  s_i * x_ijk <= C_k   for all bins j of type k  (capacity)
     x_ijk in {0,1}, y_k >= 0 integer
```

where y_k counts bins of type k used and x_ijk indicates item i is in the
j-th bin of type k. When all bin types have equal cost, VS-BPP reduces to
standard BPP.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| General VS-BPP | NP-hard | Friesen & Langston (1986) |
| K = 1 (standard BPP) | NP-hard (strongly) | Garey & Johnson (1979) |
| Equal cost per unit capacity | Equivalent to BPP | -- |

The problem is NP-hard since it generalizes standard BPP (K = 1 case).
Approximation results depend on the cost structure of available bin types.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| FFD (base heuristic) | Partially | Single bin type only, ignores cost |
| FFD Best-Type (variant) | Yes | FFD with cheapest-fit bin type selection |
| Cost-Ratio Greedy (variant) | Yes | Prefer bin types with best capacity/cost |
| Simulated Annealing (variant) | Yes | Permutation + type encoding, FFD decoder |

## Implementations

Python files in this directory:
- `instance.py` -- VSBPPInstance, bin type definitions, cost computation
- `heuristics.py` -- FFD Best-Type, Cost-Ratio Greedy
- `metaheuristics.py` -- SA with permutation encoding and best-fit decoder
- `tests/test_vsbpp.py` -- 24 tests

## Applications

- **Cloud computing**: selecting VM instance types (t3.micro vs m5.xlarge)
- **Container shipping**: choosing container sizes (20ft vs 40ft)
- **Memory allocation**: page size selection (4KB, 2MB, 1GB huge pages)
- **Fleet selection**: choosing vehicle types for deliveries

## Key References

- Friesen, D.K. & Langston, M.A. (1986). "Variable sized bin packing." SIAM J. Computing 15(1), 222-230.
- Correia, I., Gouveia, L. & Saldanha-da-Gama, F. (2008). "Solving the variable size bin packing problem with discretized formulations." C&OR 35(6), 2103-2113.
- Haouari, M. & Serairi, M. (2009). "Heuristics for the variable sized bin-packing problem." C&OR 36(10), 2877-2884. [TODO: verify year]

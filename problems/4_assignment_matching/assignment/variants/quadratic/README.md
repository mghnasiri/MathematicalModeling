# Quadratic Assignment Problem (Assignment Variant)

## What Changes

In the **quadratic assignment problem** (QAP), the objective becomes quadratic
in the assignment variables rather than linear. Given n facilities and n
locations, a flow matrix F (flow between pairs of facilities) and a distance
matrix D (distance between pairs of locations), each facility is assigned to
a unique location to minimize the total interaction cost: the sum over all
facility pairs of their flow times the distance between their assigned
locations. This models facility layout problems where co-location matters --
hospital department layout (minimize patient transport), campus building
placement, keyboard key arrangement, and circuit board component placement.

The key structural difference from the base LAP is the quadratic objective:
the cost of assigning facility i depends on where all other facilities are
assigned, creating pairwise dependencies that make the problem dramatically
harder. QAP is one of the hardest problems in combinatorial optimization.

## Mathematical Formulation

The base LAP linear objective is replaced by a quadratic one:

```
min  sum_i sum_j  f_ij * d_{pi(i), pi(j)}
```
where pi is a permutation (assignment of facilities to locations).

**Equivalently with binary variables:**
```
min  sum_i sum_j sum_k sum_l  f_ij * d_kl * x_ik * x_jl
s.t. sum_k  x_ik = 1        for all i  (each facility placed once)
     sum_i  x_ik = 1        for all k  (each location used once)
     x_ik in {0,1}
```

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| QAP (general) | NP-hard | Sahni & Gonzalez (1976) |
| No constant-factor approx | Unless P=NP | Sahni & Gonzalez (1976) |
| LAP (base, linear objective) | Polynomial O(n^3) | Kuhn (1955) |

NP-hard with no known constant-factor approximation algorithm. Exact methods
struggle beyond n = 30. This inapproximability result makes QAP fundamentally
harder than most other assignment variants.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Hungarian (base LAP exact) | No | Cannot handle quadratic objective |
| Greedy construction (variant) | Yes | Assign highest-flow pair to closest locations |
| 2-opt local search (variant) | Yes | Pairwise facility swap, O(n^2) per iteration |
| SA (variant metaheuristic) | Yes | Pairwise swap neighborhood, Boltzmann acceptance |

## Implementations

Python files in this directory:
- `instance.py` -- QAPInstance, flow matrix F, distance matrix D
- `heuristics.py` -- Greedy construction, 2-opt local search
- `metaheuristics.py` -- SA with pairwise swap neighborhood
- `tests/test_qap.py` -- 17 tests

## Applications

- Hospital department layout (minimize patient transport)
- Campus building placement (minimize inter-building traffic)
- Keyboard layout optimization (minimize finger travel)
- Circuit board component placement (minimize wiring length)

## Key References

- Koopmans, T.C. & Beckmann, M. (1957). "Assignment problems and the location of economic activities." Econometrica 25(1), 53-76.
- Sahni, S. & Gonzalez, T. (1976). "P-complete approximation problems." JACM 23(3), 555-565.
- Burkard, R.E., Dell'Amico, M. & Martello, S. (2009). Assignment Problems. SIAM.
- Burkard, R.E. & Rendl, F. (1984). "A thermodynamically motivated simulation procedure for combinatorial optimization problems." EJOR 17(2), 169-174.

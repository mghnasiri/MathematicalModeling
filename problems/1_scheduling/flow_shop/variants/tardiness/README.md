# Flow Shop with Total Weighted Tardiness (PFSP Variant)

## What Changes

The **tardiness flow shop** (Fm | prmu | Sigma wj Tj) replaces the standard
makespan objective with total weighted tardiness. Each job j has a due date d_j
and a weight w_j reflecting its importance. The tardiness of job j is
T_j = max(0, C_j - d_j), and the objective is to minimize the weighted sum.
This models production environments where customer delivery commitments and
order priorities matter more than overall line throughput -- contract
manufacturing, order fulfillment, and make-to-order production with penalty
clauses for late delivery.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional parameters:**
- d_j: due date of job j
- w_j: weight (priority) of job j

**Completion time recursion (unchanged):**
```
C[i][k] = max(C[i-1][k], C[i][k-1]) + p[i][pi(k)]
```

**Modified objective:**
```
min sum_{j=1}^{n} w_j * max(0, C[m-1][pi(j)] - d_{pi(j)})
```

The machine-level constraints and completion time recursion are identical to
the base PFSP. Only the objective function changes from Cmax to Sigma wj Tj.

## Complexity

NP-hard. Even the single-machine case 1 || Sigma wj Tj is strongly NP-hard
(Lawler, 1977). The flow shop version inherits this hardness and adds the
multi-machine sequencing dimension. Due date assignment methods (SLK, TWK, RDK)
are used to generate benchmark instances.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Optimizes Cmax, not tardiness |
| NEH (base heuristic) | Partially | Insertion criterion must use tardiness, not makespan |
| NEH-Tardiness (variant heuristic) | Yes | NEH with weighted tardiness insertion criterion |
| EDD Rule (variant heuristic) | Yes | Sort by due date; no weight consideration |
| WSPT Rule (variant heuristic) | Yes | Sort by p_j/w_j; no due date consideration |
| ATC-based (variant heuristic) | Possible | Adapt ATC from single-machine; composite priority |
| SA (base meta, adapted) | Yes | Swap/insertion with tardiness objective |
| GA (base meta, adapted) | Possible | Fitness = total weighted tardiness |
| IG (base meta, adapted) | Possible | Destroy-repair with tardiness evaluation |

**No implementation in this directory.** Parent metaheuristics (SA, GA, IG)
can be adapted by changing the objective evaluation from makespan to total
weighted tardiness. The completion time computation remains identical; only
the fitness/cost function changes.

## Applications

- Make-to-order manufacturing with delivery deadlines
- Contract manufacturing with late-delivery penalties
- Order fulfillment in e-commerce warehouses
- Semiconductor wafer fabrication with customer priorities

## Key References

- Kim, Y.-D. (1993). "Heuristics for Flowshop Scheduling Problems Minimizing Mean Tardiness" -- [DOI](https://doi.org/10.1057/jors.1993.31)
- Vallada, E. & Ruiz, R. (2010). "Genetic Algorithms with Path Relinking for the Minimum Tardiness Permutation Flowshop Problem" [TODO: verify DOI]
- Armentano, V.A. & Ronconi, D.P. (1999). "Tabu Search for Total Tardiness Minimization in Flowshop Scheduling Problems" -- [DOI](https://doi.org/10.1016/S0305-0548(98)00060-1)

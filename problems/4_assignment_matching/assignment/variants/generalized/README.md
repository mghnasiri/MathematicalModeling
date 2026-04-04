# Generalized Assignment Problem (Assignment Variant)

## What Changes

In the **generalized assignment problem** (GAP), the base linear assignment
problem is extended with agent capacities and variable resource consumption.
In the base LAP, each agent handles exactly one task with uniform cost; in GAP,
m agents each have capacity b_i, and assigning job j to agent i consumes a_{ij}
units of resource and incurs cost c_{ij}. Each job must be assigned to exactly
one agent, and each agent's total resource consumption must not exceed its
capacity. This models scenarios with heterogeneous resources -- assigning tasks
to machines with different capacities, scheduling jobs on servers with memory
limits, and workforce allocation where employees have varying skill levels
and workloads.

The key structural difference is that GAP is NP-hard (even checking feasibility
is NP-complete), whereas the base LAP is solvable in polynomial time via the
Hungarian method.

## Mathematical Formulation

The base LAP formulation gains capacity constraints and variable resource use:

```
min  sum_i sum_j  c_ij * x_ij
s.t. sum_i  x_ij = 1                    for all j  (each job assigned once)
     sum_j  a_ij * x_ij <= b_i          for all i  (agent capacity)
     x_ij in {0,1}
```

Note that the square cost matrix of LAP is replaced by a rectangular (m x n)
cost matrix and a separate (m x n) resource consumption matrix. Multiple jobs
can be assigned to the same agent as long as capacity allows.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| GAP (general) | Strongly NP-hard | Martello & Toth (1990) |
| Feasibility only | NP-complete | Martello & Toth (1990) |
| LAP (base, n=m, b_i=1) | Polynomial O(n^3) | Kuhn (1955) |

Strongly NP-hard. Even determining whether a feasible assignment exists
is NP-complete, making GAP one of the hardest assignment variants.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Hungarian (base LAP exact) | No | Does not handle capacities or multi-assignment |
| Greedy ratio (variant) | Yes | Assign by best cost/resource ratio |
| First-fit decreasing (variant) | Yes | Sort jobs by difficulty, assign greedily |
| SA (variant metaheuristic) | Yes | Reassign and swap moves with capacity checks |

## Implementations

Python files in this directory:
- `instance.py` -- GAPInstance, resource matrix, capacity validation
- `heuristics.py` -- Greedy ratio, first-fit decreasing
- `metaheuristics.py` -- SA with reassign and swap moves
- `tests/test_gap.py` -- 23 tests

## Applications

- Machine loading in flexible manufacturing systems
- Server task assignment with memory/CPU constraints
- Workforce scheduling with skill and workload limits
- Vehicle-to-route assignment in logistics

## Key References

- Ross, G.T. & Soland, R.M. (1975). "A branch and bound algorithm for the generalized assignment problem." Mathematical Programming 8(1), 91-103.
- Martello, S. & Toth, P. (1990). Knapsack Problems: Algorithms and Computer Implementations. Wiley.
- Osman, I.H. (1995). "Heuristics for the generalized assignment problem: simulated annealing and tabu search approaches." OR Spektrum 17(4), 211-225.

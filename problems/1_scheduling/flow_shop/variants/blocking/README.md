# Blocking Flow Shop (PFSP Variant)

## What Changes

In the **blocking permutation flow shop** (Fm | prmu, blocking | Cmax), there
are no intermediate storage buffers between machines. When a job finishes
processing on machine i, it remains on that machine (blocking it) until
machine i+1 becomes available. This prevents the next job from starting
on machine i. Real-world production lines with limited buffer space --
such as robotic cells, paint shops, and concrete production -- motivate
this variant. The objective remains makespan minimization, but blocking
always yields makespan >= the standard PFSP for the same permutation.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Additional variable:** Departure time D[i][k] -- the time job pi(k) leaves machine i.

**Modified completion time recursion:**
```
C[i][k] = max(C[i-1][k], D[i][k-1]) + p[i][pi(k)]
D[i][k] = max(C[i][k], D[i+1][k-1])    for i < m-1
D[m-1][k] = C[m-1][k]                    (no blocking on last machine)
```

In the standard PFSP, D[i][k] = C[i][k] because jobs can wait in an
unlimited buffer. Here, D[i][k] >= C[i][k], so the previous job's
departure time replaces its completion time in the recursion.

## Complexity

| Machines | Complexity | Reference |
|----------|-----------|-----------|
| m = 2 | Polynomial | Gilmore & Gomory (1964) |
| m >= 3 | NP-hard | Hall & Sriskandarajah (1996) |

The 2-machine case reduces to a special TSP solvable in polynomial time.
For m >= 3, the problem is NP-hard, same complexity class as the base PFSP.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Does not account for blocking delays |
| NEH (base heuristic) | No | Must use blocking-aware makespan evaluation |
| NEH-B (variant heuristic) | Yes | NEH adapted with departure-time recursion |
| Profile Fitting (variant heuristic) | Yes | Greedy construction minimizing blocking time |
| IG-B (variant metaheuristic) | Yes | Iterated Greedy with blocking-aware evaluation |
| Tabu Search (base meta, adapted) | Possible | Neighborhood moves transfer; evaluation must change |
| SA (base meta, adapted) | Possible | Swap/insertion moves work; re-evaluate with D[i][k] |

## Implementations

Python files in this directory:
- `instance.py` -- BlockingFlowShopInstance, departure time computation
- `heuristics.py` -- NEH-B, Profile Fitting (PF-B)
- `metaheuristics.py` -- Iterated Greedy for blocking (IG-B)
- `__init__.py` -- Package init

## Applications

- Manufacturing lines with limited buffer space
- Robotic cells (robot transfers between machines)
- Paint shops (parts must move immediately to avoid drying)
- Concrete production (material must be poured immediately)

## Key References

- Hall, N.G. & Sriskandarajah, C. (1996). "A Survey of Machine Scheduling Problems with Blocking and No-Wait in Process"
- Ronconi, D.P. (2004). "A Note on Constructive Heuristics for the Flowshop Problem with Blocking"
- Grabowski, J. & Pempera, J. (2007). "The Permutation Flow Shop Problem with Blocking. A Tabu Search Approach"
- Ribas, I. et al. (2011). "An Iterated Greedy Algorithm for the Flowshop Scheduling Problem with Blocking"

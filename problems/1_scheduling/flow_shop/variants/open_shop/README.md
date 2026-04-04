# Open Shop Scheduling (PFSP Variant)

## What Changes

In the **open shop** (Om || Cmax), each job must be processed on all m machines,
but the order of operations within each job is unrestricted -- any job can visit
machines in any order. This is the key structural difference from the flow shop,
where all jobs follow the same fixed machine sequence. Open shop models service
environments where tasks have no technological ordering: diagnostic testing in
hospitals (blood work, X-ray, MRI in any order), vehicle maintenance with
independent repair stations, and exam scheduling where students take tests in
any order. The objective remains makespan minimization.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Removed constraint:** The fixed machine ordering M1 -> M2 -> ... -> Mm is dropped.

**Decision variables:** For each job j, determine both the machine ordering
sigma_j (a permutation of {1,...,m}) and the start times:
```
start(j, sigma_j(k))  for k = 1, ..., m
```

**Constraints:**
- Each machine processes one job at a time (disjunctive)
- Each job is on one machine at a time (conjunctive within job, but order is free)
- No preemption: once started, an operation runs to completion

**Objective:** Minimize Cmax = max_{j,i} (start(j,i) + p[i][j])

This is substantially more complex than PFSP because the solution space includes
per-job machine orderings in addition to the scheduling decisions.

## Complexity

| Machines | Complexity | Reference |
|----------|-----------|-----------|
| m = 2 | Polynomial, O(n log n) | Gonzalez & Sahni (1976) |
| m >= 3 | NP-hard | Gonzalez & Sahni (1976) |

The 2-machine open shop has an elegant polynomial algorithm. A useful lower
bound is LB = max(max_i sum_j p[i][j], max_j sum_i p[i][j]).

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Johnson's Rule (base exact) | No | Assumes fixed machine ordering |
| NEH (base heuristic) | No | Insertion logic assumes flow shop structure |
| LPT dispatching (variant heuristic) | Yes | Assign longest available operation first |
| Greedy earliest-start (variant heuristic) | Yes | Schedule operations by earliest feasibility |
| SA (variant metaheuristic) | Yes | Neighborhood: swap operation order within a job |
| GA (base meta, adapted) | Possible | Needs operation-level encoding, not permutation |

**No implementation in this directory.** The open shop requires fundamentally
different data structures (per-job machine orderings) than the permutation-based
parent. Parent metaheuristic frameworks (SA, GA) can be adapted with an
operation-level encoding and appropriate neighborhood operators.

## Applications

- Hospital diagnostic testing (tests in any order)
- Vehicle maintenance and repair shops
- Satellite ground station scheduling
- Exam timetabling (exams in any order per student)

## Key References

- Gonzalez, T. & Sahni, S. (1976). "Open Shop Scheduling to Minimize Finish Time" -- [DOI](https://doi.org/10.1145/321958.321970)
- Liaw, C.-F. (2000). "A Hybrid Genetic Algorithm for the Open Shop Scheduling Problem" -- [DOI](https://doi.org/10.1016/S0377-2217(98)00328-4)
- Gueret, C. & Prins, C. (1999). "A New Lower Bound for the Open Shop Problem" [TODO: verify DOI]

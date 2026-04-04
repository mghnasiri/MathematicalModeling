# Parallel Machine with Sequence-Dependent Setup Times (PM Variant)

## What Changes

In the **SDST parallel machine** variant (Rm | S_sd | C_max), setup time between
consecutive jobs depends on the machine, the preceding job, and the next job.
The base parallel machine problem assumes jobs are independent on each machine;
this variant introduces sequence-dependent changeovers s_{ijk} representing the
setup time on machine k when switching from job i to job j. This models
real-world scenarios where changeover costs vary by both the job transition
and the equipment -- semiconductor tool qualification, printing press color
changes, and injection mold swaps.

The key structural difference is that each machine's sub-problem becomes a
sequencing problem (similar to a TSP) rather than a simple load-balancing
problem. The joint assignment + sequencing makes the problem significantly
harder than the base Rm||C_max.

## Mathematical Formulation

The base PM formulation gains setup time terms:

**Completion time on machine k:**
```
C_{j,k} = C_{prev(j,k),k} + s_{prev(j,k),j,k} + p_{j,k}
```
where prev(j,k) is the job processed immediately before j on machine k.

**Makespan definition:**
```
C_max >= C_{last(k),k}    for all k = 1, ..., m
```

**Setup time matrix:** s_{ijk} for each machine k, giving the changeover time
from job i to job j. Typically s_{iik} = 0 (no setup between identical jobs).

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| Rm \| S_sd \| C_max | NP-hard | Allahverdi et al. (2008) |
| Pm \| S_sd \| C_max (identical) | NP-hard | Allahverdi et al. (2008) |
| Single machine (1 \| S_sd \| C_max) | NP-hard | -- |

NP-hard since it generalizes Rm||C_max. The per-machine sequencing
sub-problem is itself an asymmetric TSP with setup-time distances.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| LPT (base heuristic) | Partially | Ignores setup times in assignment |
| Greedy ECT-SDST (variant) | Yes | Assign to machine with earliest completion including setup |
| LPT-SDST (variant) | Yes | Sort by avg processing time, assign with setup-aware ECT |
| SA (variant metaheuristic) | Yes | Move/swap/reorder neighborhoods on bottleneck machine |

## Implementations

Python files in this directory:
- `instance.py` -- PMSDSTInstance, 3D setup time matrix s[i][j][k]
- `heuristics.py` -- Greedy ECT-SDST, LPT-SDST
- `metaheuristics.py` -- SA with move/swap/reorder neighborhoods
- `tests/test_pmsdst.py` -- 31 tests

## Applications

- **Semiconductor fabrication**: machine-specific tool changeovers
- **Printing**: color and plate changeovers vary by press
- **Injection molding**: mold changes depend on previous and next mold
- **Chemical processing**: cleaning between product batches varies by reactor

## Key References

- Allahverdi, A., Ng, C.T., Cheng, T.C.E. & Kovalyov, M.Y. (2008). "A survey of scheduling problems with setup times or costs." EJOR 187(3), 985-1032.
- Rabadi, G., Moraga, R.J. & Al-Salem, A. (2006). "Heuristics for the unrelated parallel machine scheduling problem with setup times." J. Intelligent Manufacturing 17(2), 199-207.
- Weng, M.X., Lu, J. & Ren, H. (2001). "Unrelated parallel machine scheduling with setup consideration and a total weighted completion time objective." IJPE 70(3), 215-226.

# Multi-Mode RCPSP (RCPSP Variant)

## What Changes

In the **multi-mode RCPSP** (MPS | prec | C_max), each activity can be executed
in one of several modes, where each mode specifies a different combination of
duration and resource requirements. This introduces a time-resource tradeoff:
an activity can be completed faster by consuming more resources, or slower
using fewer resources. The decision problem is now two-fold: (1) select a mode
for each activity, and (2) schedule activities respecting precedence and
resource constraints.

This models real-world projects where tasks can be "crashed" (expedited using
more workers or equipment) or performed in an economy mode (fewer resources,
longer duration) -- construction projects with crew sizing options, software
development with staffing alternatives, and R&D projects with equipment
choices.

## Mathematical Formulation

The base RCPSP formulation gains mode selection variables:

**Mode selection:**
```
sum_m  x_{im} = 1    for all activities i  (exactly one mode per activity)
```

**Duration depends on mode:**
```
d_i = sum_m  d_{im} * x_{im}    for all i
```

**Resource consumption depends on mode:**
```
r_{ik} = sum_m  r_{ikm} * x_{im}    for all i, resources k
```

**Resource constraint (per time period):**
```
sum_{i: i active at t}  r_{ik} <= R_k    for all t, k
```

The base precedence and makespan constraints remain unchanged.

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| MRCPSP (general) | Strongly NP-hard | Kolisch & Drexl (1997) |
| RCPSP (single mode) | Strongly NP-hard | Blazewicz et al. (1983) |
| Mode selection alone | NP-hard | Talbot (1982) |

Strongly NP-hard. Even the mode selection sub-problem (ignoring scheduling)
is NP-hard when resource constraints couple the mode choices across activities.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Serial SGS (base RCPSP) | Partially | Needs mode pre-selection |
| Serial SGS + shortest mode (variant) | Yes | Pick fastest mode, schedule with SGS |
| Serial SGS + resource-aware mode (variant) | Yes | Pick mode balancing speed and resource use |
| SA (variant metaheuristic) | Yes | Mode switching + activity swap neighborhood |

## Implementations

Python files in this directory:
- `instance.py` -- MRCPSPInstance, mode definitions, precedence DAG
- `heuristics.py` -- Serial SGS with shortest-mode and resource-aware mode selection
- `metaheuristics.py` -- SA with mode switching and activity swapping
- `tests/test_mrcpsp.py` -- 16 tests

## Applications

- Construction project management (crew sizing decisions)
- Software project planning (staffing level choices)
- R&D project scheduling (equipment and lab allocation)
- Manufacturing process planning (tool and workforce flexibility)

## Key References

- Talbot, F.B. (1982). "Resource-constrained project scheduling with time-resource tradeoffs: the nonpreemptive case." Management Science 28(10), 1197-1210.
- Sprecher, A. & Drexl, A. (1998). "Multi-mode resource-constrained project scheduling by a simple, general and powerful sequencing algorithm." EJOR 107(2), 431-450.
- Hartmann, S. & Briskorn, D. (2010). "A survey of variants and extensions of the resource-constrained project scheduling problem." EJOR 207(1), 1-14.
- Kolisch, R. & Drexl, A. (1997). "Local search for nonpreemptive multi-mode resource-constrained project scheduling." IIE Transactions 29(11), 987-999. [TODO: verify volume/pages]

# Preemptive Single Machine Scheduling (SM Variant)

## What Changes

In the **preemptive** variant (1 | pmtn, r_j | Sigma C_j), jobs have release
dates r_j and the machine may interrupt (preempt) a running job to start a
more urgent one. The interrupted job resumes later from where it left off
with no penalty. This models CPU scheduling in operating systems, preemptive
manufacturing with priority interrupts, and hospital triage where incoming
emergencies displace lower-priority patients.

The key structural difference from the base single machine problem is that
the schedule is no longer a simple permutation. A job may be processed in
multiple non-contiguous intervals, and the decision at each event point
(release or completion) is which job to process next.

## Mathematical Formulation

The base SM formulation is augmented with release dates and preemption:

**Release constraint:**
```
start(j) >= r_j    for all j
```

**Preemption:** A job j with remaining processing time rem_j can be
interrupted at any time. The total processing received must equal p_j:
```
sum of intervals assigned to j = p_j
```

**Event-driven policy:** At each event (job release or job completion),
select the job with shortest remaining processing time (SRPT).

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| 1 \| pmtn, r_j \| Sigma C_j | Polynomial (SRPT optimal) | Schrage (1968) |
| 1 \| pmtn, r_j \| Sigma w_j C_j | NP-hard | Labetoulle et al. (1984) |
| 1 \| pmtn \| L_max | Polynomial (EDD) | Jackson (1955) |

For the unweighted case, SRPT (Shortest Remaining Processing Time) is optimal
and runs in O(n^2) via event-driven simulation. Adding weights makes the
problem NP-hard, requiring heuristic approaches.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| SPT (base heuristic) | No | Ignores release dates, no preemption logic |
| SRPT (variant exact) | Yes | Optimal for 1\|pmtn,r_j\|Sigma C_j, O(n^2) |
| Weighted SRPT (variant heuristic) | Yes | Heuristic extension for Sigma w_j C_j |
| EDD (base heuristic) | Partially | Optimal for L_max with preemption |

## Implementations

Python files in this directory:
- `instance.py` -- PreemptiveSMInstance, release dates, remaining time tracking
- `heuristics.py` -- SRPT (optimal for unweighted), Weighted SRPT (heuristic)
- `tests/test_preemptive.py` -- 16 tests

## Applications

- Operating system CPU scheduling (round-robin, priority preemption)
- Manufacturing with priority interrupts (rush orders)
- Hospital triage (incoming emergencies preempt lower-priority care)
- Real-time embedded systems (interrupt-driven task scheduling)

## Key References

- Schrage, L. (1968). "A proof of the optimality of the shortest remaining processing time discipline." Operations Research 16(3), 687-690.
- Baker, K.R. & Trietsch, D. (2009). Principles of Sequencing and Scheduling. Wiley.
- Labetoulle, J., Lawler, E.L., Lenstra, J.K. & Rinnooy Kan, A.H.G. (1984). "Preemptive scheduling of uniform machines subject to release dates." Progress in Combinatorial Optimization, 245-261. [TODO: verify publisher]

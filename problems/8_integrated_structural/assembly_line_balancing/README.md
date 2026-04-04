# Assembly Line Balancing (ALBP)

> **Status:** Redirect — full implementation lives in `1_scheduling/assembly_line_balancing/`.

## Family 8 · Integrated Structural

Assembly Line Balancing is classified as an **integrated structural** problem because it combines task-to-station **assignment** (Family 4) with precedence-constrained **scheduling** (Family 1) under cycle time constraints. However, in this repository the primary implementation resides in the scheduling family where it shares infrastructure with other scheduling problems.

## Problem Summary

- **SALBP-1:** Given a set of tasks with processing times and a precedence graph, assign tasks to stations to minimize the number of stations while respecting the cycle time $C$
- **SALBP-2:** Minimize the cycle time for a given number of stations
- **Complexity:** NP-hard (Wee & Magazine, 1982)
- **Key heuristic:** Ranked Positional Weight (RPW) — assign tasks by decreasing sum of own + successor processing times

## Implementation

The full implementation with instance dataclass, RPW heuristic, and tests is at:

**[`../../1_scheduling/assembly_line_balancing/`](../../1_scheduling/assembly_line_balancing/)**

```
1_scheduling/assembly_line_balancing/
├── instance.py               # ALBInstance with precedence graph
├── heuristics/
│   └── rpw_heuristic.py      # Ranked Positional Weight
└── tests/
    └── test_alb.py            # Test suite
```

## Key References

- Salveson, M.E. (1955). The assembly line balancing problem. *Journal of Industrial Engineering*, 6(3), 18-25.
- Scholl, A. & Becker, C. (2006). State-of-the-art exact and heuristic solution procedures for simple assembly line balancing. *European Journal of Operational Research*, 168(3), 666-693.
- Helgeson, W.P. & Birnie, D.P. (1961). Assembly line balancing using the ranked positional weight technique. *Journal of Industrial Engineering*, 12(6), 394-398.

## See also
- [`../../1_scheduling/assembly_line_balancing/`](../../1_scheduling/assembly_line_balancing/) — Full implementation with RPW heuristic and tests
- [`../../4_assignment_matching/assignment/`](../../4_assignment_matching/assignment/) — Related assignment problem

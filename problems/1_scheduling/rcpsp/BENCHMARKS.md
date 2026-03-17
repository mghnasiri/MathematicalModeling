# RCPSP — Benchmarks

## Standard Test Instance Libraries

### PSPLIB (Project Scheduling Problem Library) — The Gold Standard
- **URL**: https://www.om-db.wi.tum.de/psplib/
- **Maintained by**: TU Munich (originally Kolisch & Sprecher)
- **Instance sets**:

| Set | Activities | Resources | Instances | Status |
|-----|-----------|-----------|-----------|--------|
| `j30` | 30 | 4 | 480 | All solved to optimality |
| `j60` | 60 | 4 | 480 | Most solved, some open |
| `j90` | 90 | 4 | 480 | Many open |
| `j120` | 120 | 4 | 600 | Most still open |

- **Parameters varied**: Network complexity (NC), resource factor (RF), resource strength (RS)
- **Usage**: THE standard benchmark — virtually all RCPSP papers use PSPLIB

### Multi-Mode RCPSP (MMLIB)
- **URL**: https://www.om-db.wi.tum.de/psplib/
- **Instance sets**: `j10` through `j30` (multi-mode)
- **Modes**: Each activity has 1–3 execution modes
- **Resources**: Renewable + non-renewable

### PSPLIB Extensions
- **RCPSP/max instances**: Generalized precedence relations
- **Multi-skill instances**: Workers with skill sets

### RanGen Instances (Demeulemeester et al.)
- **Generator**: RanGen1 and RanGen2
- **Purpose**: Controlled generation with specific complexity parameters
- **URL**: https://www.projectmanagement.ugent.be/research/data

### Patterson Instances (1984)
- **Instances**: 110 instances
- **Sizes**: Small (7–51 activities)
- **Usage**: Classic small instances, useful for exact method validation

### KSD Instances (Kolisch, Sprecher & Drexl, 1995)
- **Same as PSPLIB j30/j60/j90/j120** — the original generation paper

---

## Instance Format (PSPLIB)

```
************************************************************************
file with calculation calculation calculation ...
************************************************************************
                                                    - renewable
                                                    - nonrenewable
                                                    - doubly constrained
************************************************************************
RESOURCES
  - renewable                 :  4   R 1  R 2  R 3  R 4
  - nonrenewable              :  0
  ...
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1        3           2  3  4
   2        1        2           5  6
   ...
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  R 3  R 4
   1    1     0       0    0    0    0
   2    1     8       4    0    0    0
   ...
RESOURCEAVAILABILITIES:
  R 1  R 2  R 3  R 4
   12    13   4   12
```

---

## Best Known Solutions

- **j30**: All 480 instances solved to optimality
- **j60**: Best known upper and lower bounds maintained at PSPLIB
- **j120**: Many instances still open — active research area
- Updated results: https://www.om-db.wi.tum.de/psplib/

---

## Benchmark Generators

| Generator | Reference | Notes |
|-----------|-----------|-------|
| ProGen | Kolisch et al. (1995) | Original PSPLIB generator |
| RanGen1 | Demeulemeester et al. (2003) | Better control over network topology |
| RanGen2 | Vanhoucke et al. (2008) | Extended parameter control |

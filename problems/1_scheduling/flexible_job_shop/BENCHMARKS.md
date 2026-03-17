# Flexible Job Shop Scheduling — Benchmarks

## Standard Test Instance Libraries

### Brandimarte Instances (1993) — MK01–MK10
- **Instances**: 10 instances (`MK01`–`MK10`)
- **Sizes**: 10×6 to 20×15 (jobs × machines)
- **Flexibility**: Varies from partial to total
- **Usage**: Most widely used FJSP benchmark
- **Reference**: Brandimarte, P. (1993). "Routing and Scheduling in a Flexible Job Shop by Tabu Search" — *Annals of Operations Research*

### Kacem Instances (2002)
- **Instances**: 4 instances
- **Sizes**: 4×5, 8×8, 10×7, 15×10
- **Flexibility**: Total (all machines eligible)
- **Usage**: Common for multi-objective FJSP

### Hurink Instances (1994) — edata, rdata, vdata
- **Instances**: 3 sets based on flexibility level
  - `edata`: Few eligible machines per operation
  - `rdata`: Moderate flexibility
  - `vdata`: High flexibility
- **Base**: Derived from classical JSP instances (la01–la40, ft06, ft10, ft20, etc.)
- **Usage**: Systematic study of flexibility impact

### Dauzère-Pérès & Paulli (1997)
- **Instances**: 18 instances
- **Sizes**: 10×5 to 20×10
- **Usage**: Medium difficulty, well-studied

### Fattahi Instances (2007)
- **Instances**: 20 small instances (SFJS01–SFJS10, MFJS01–MFJS10)
- **Sizes**: 2×2 to 10×7
- **Usage**: Validation of exact methods (small enough to solve optimally)

### Barnes & Chambers (1996)
- **Instances**: Setb4 set
- **Sizes**: Various
- **Usage**: Additional benchmarking

---

## Instance Format

Common format (Brandimarte-style):
```
n  m
n_ops  machine_1 time_1 machine_2 time_2 ...  |  machine_1 time_1 ...  |  ...
```

Each job line lists its operations separated by `|`. Each operation lists eligible (machine, processing_time) pairs.

---

## Best Known Solutions

- Brandimarte instances: well-studied, most solved to optimality
- Hurink instances: many best-known solutions available
- Comprehensive results: see Mastrolilli & Gambardella (2000) and subsequent papers

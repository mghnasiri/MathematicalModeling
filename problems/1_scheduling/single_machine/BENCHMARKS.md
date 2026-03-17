# Single Machine Scheduling — Benchmarks

## Standard Test Instance Libraries

### OR-Library (Beasley)
- **URL**: http://people.brunel.ac.uk/~mastjjb/jeb/info.html
- **Instances**: Weighted tardiness instances (40, 50, 100 jobs)
- **Format**: Text files with processing times, weights, due dates
- **Usage**: Standard benchmark for $1 \mid\mid \sum w_j T_j$

### Potts & Van Wassenhove (1985) Instances
- **URL**: Available through OR-Library
- **Instances**: Single machine total weighted tardiness
- **Sizes**: 10–100 jobs
- **Usage**: Classic benchmark, widely cited

### Crauwels, Potts & Van Wassenhove (1998)
- **Instances**: Single machine total weighted tardiness
- **Sizes**: 40, 50, 100 jobs with varying due date tightness and range
- **Note**: Generates instances using parameters (TF, RDD) controlling tardiness factor and due date range

### Random Instance Generation (Pinedo convention)
Standard random generation for single machine:
- $p_j \sim U[1, 100]$
- $w_j \sim U[1, 10]$
- $d_j \sim U[P(1 - TF - RDD/2),\ P(1 - TF + RDD/2)]$ where $P = \sum p_j$
- TF (tardiness factor): 0.2, 0.4, 0.6, 0.8
- RDD (range of due dates): 0.2, 0.4, 0.6, 0.8

---

## Instance Format

Typical single machine instance file:
```
n
p_1  w_1  d_1
p_2  w_2  d_2
...
p_n  w_n  d_n
```

---

## Best Known Solutions

- For weighted tardiness: see Congram, Potts & van de Velde (2002) — dynasearch results
- For total tardiness: see Du & Leung (1990) DP results

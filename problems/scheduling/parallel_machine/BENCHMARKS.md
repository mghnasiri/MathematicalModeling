# Parallel Machine Scheduling — Benchmarks

## Standard Test Instances

### Identical Parallel Machines ($P_m \mid\mid C_{\max}$)

#### Dell'Amico & Martello (1995)
- **Instances**: Generated using parameters $n$ and $m$
- **Sizes**: $n \in \{10, 20, 50, 100, 200, 500, 1000\}$, $m \in \{2, 3, 5, 10, 20\}$
- **Processing times**: $p_j \sim U[1, 100]$
- **Reference**: "Optimal Scheduling of Tasks on Identical Parallel Processors" — *ORSA J. on Computing*

#### França et al. (1994)
- **Instances**: Large-scale parallel machine makespan
- **Sizes**: Up to 1000 jobs
- **Processing times**: Various distributions

### Unrelated Parallel Machines ($R_m$)

#### Glass, Potts & Shade (1994)
- **Instances**: Unrelated parallel machine total weighted completion
- **Sizes**: $n \in \{10, 20, 30, 40, 50\}$, $m \in \{2, 5, 10\}$
- **URL**: Available through OR-Library

#### Bank & Werner (2001)
- **Instances**: Unrelated machines with sequence-dependent setup times
- **Sizes**: Various $n$ and $m$ combinations

### Weighted Tardiness on Parallel Machines

#### Azizoglu & Kirca (1998)
- **Instances**: $P_m \mid\mid \sum w_j T_j$
- **Sizes**: $n \in \{15, 20, 25\}$, $m \in \{2, 3, 4\}$

---

## Random Instance Generation

Standard generation (following Dell'Amico & Martello):
```
n = number of jobs
m = number of machines
p_j ~ U[1, 100]                        (identical)
p_ij ~ U[1, 100]                       (unrelated)
w_j ~ U[1, 10]                         (for weighted objectives)
d_j ~ U[P/m * (1-TF-RDD/2), P/m * (1-TF+RDD/2)]  (due dates)
```

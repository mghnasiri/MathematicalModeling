# Parallel Machine with Sequence-Dependent Setup Times (Rm | Ssd | Cmax)

## Problem Definition

Assign n jobs to m unrelated parallel machines, where setup time depends on the machine, the current job, and the preceding job. Minimize makespan.

```
min  Cmax
s.t. Cmax ≥ C_last(k)              ∀ k  (makespan definition)
     s_ijk + p_jk on machine k      (setup + processing)
     each job assigned to exactly one machine
```

## Complexity

NP-hard (generalizes Rm||Cmax which is already NP-hard).

## Applications

- **Semiconductor fabrication**: machine-specific tool changeovers
- **Printing**: color and plate changeovers vary by press
- **Injection molding**: mold changes depend on previous and next mold
- **Chemical processing**: cleaning between product batches varies by reactor

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy ECT-SDST | Heuristic | Rabadi et al. (2006) |
| LPT-SDST | Heuristic | Rabadi et al. (2006) |
| Simulated Annealing | Metaheuristic | Kirkpatrick et al. (1983) |

# Stochastic Flow Shop (PFSP Variant)

## What Changes

The **stochastic flow shop** (Fm | prmu, stoch | E[Cmax]) extends the
permutation flow shop by modeling processing times as random variables rather
than deterministic constants. Processing times typically follow truncated normal
or log-normal distributions with known means and standard deviations. The
objective changes from minimizing makespan to minimizing expected makespan.
This models manufacturing environments where processing durations vary due to
machine wear, operator skill differences, or material variability -- situations
common in job shops, semiconductor fabs, and custom manufacturing.

## Mathematical Formulation

The base PFSP formulation (see parent README) is modified as follows:

**Modified parameters:** Processing times are random variables:
```
P[i][j] ~ Distribution(mu[i][j], sigma[i][j])
```
where mu[i][j] is the mean and sigma[i][j] the standard deviation.

**Completion time recursion (same structure, random inputs):**
```
C[i][k] = max(C[i-1][k], C[i][k-1]) + P[i][pi(k)]
```

**Modified objective:** Minimize E[Cmax] = E[C[m-1][pi(n-1)]].

The max operator in the recursion makes analytical computation of E[Cmax]
intractable in general. Monte Carlo simulation is typically used: sample
N realizations of processing times, evaluate makespan for each, and average.

## Complexity

NP-hard (generalizes the deterministic PFSP, which is NP-hard for m >= 3).
The stochastic version adds the computational burden of evaluating expected
makespan, which requires simulation or analytical approximation.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| NEH (deterministic proxy) | Partially | Use mean processing times; ignores variance |
| NEH (Monte Carlo) | Yes | Evaluate insertions via simulation; slower but variance-aware |
| SA (base meta, adapted) | Possible | Replace deterministic eval with Monte Carlo; increase cost |
| GA (base meta, adapted) | Possible | Fitness = simulated E[Cmax]; noisy evaluation |
| IG (base meta, adapted) | Possible | Destroy-repair with stochastic evaluation |
| Robust approaches | Possible | Minimize worst-case or CVaR of makespan distribution |

**No implementation in this directory.** Parent heuristics and metaheuristics
can be adapted by replacing the deterministic makespan evaluation with
Monte Carlo simulation. The key design choice is the number of simulation
replications per evaluation (trading accuracy for computation time).

## Applications

- Manufacturing with variable processing durations
- Semiconductor fabrication (yield-dependent processing)
- Custom job shops with operator-dependent cycle times
- Service operations with uncertain task durations

## Key References

- Gourgand, M., Grangeon, N. & Norre, S. (2000). "Metaheuristics for the Stochastic Flow Shop Scheduling Problem" [TODO: verify DOI]
- Framinan, J.M. & Perez-Gonzalez, P. (2015). "On Heuristic Solutions for the Stochastic Flowshop Scheduling Problem" -- [DOI](https://doi.org/10.1016/j.ejor.2014.09.062)
- Makino, T. (1965). "On a Scheduling Problem" [TODO: verify] -- early work on stochastic flow shop

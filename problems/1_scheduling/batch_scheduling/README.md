# Batch Scheduling Problem

## 1. Problem Definition

- **Input:** $n$ jobs with processing times $p_j$, weights $w_j$, family memberships; setup time $s$ between families; single machine
- **Decision:** Partition jobs into batches (same family) and sequence batches
- **Objective:** Minimize total weighted completion time $\sum w_j C_j$
- **Classification:** NP-hard

### Scheduling Notation

$1 \mid \text{batch}, s_{\text{fam}} \mid \sum w_j C_j$

---

## 2. Mathematical Formulation

$$\min \sum_{j=1}^{n} w_j C_j \tag{1}$$

$$C_j = \text{completion time of the batch containing job } j \tag{2}$$

$$\text{batch}_b = \{j : j \in \text{family } f_b\}, \quad p_b = \sum_{j \in \text{batch}_b} p_j \tag{3}$$

$$C_b = C_{b-1} + s \cdot \mathbf{1}[f_b \neq f_{b-1}] + p_b \tag{4}$$

where $s$ is the setup time between different families and batches are processed sequentially.

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy batch | Heuristic | $O(n \log n)$ | WSPT within families, sequence families by aggregate ratio |
| BATC dispatch | Heuristic | $O(n^2)$ | Batch Apparent Tardiness Cost — composite urgency index |
| MIP formulation | Exact | exponential | Assignment + sequencing variables with setup constraints |

### Batch Apparent Tardiness Cost (BATC) Pseudocode

```
BATC(jobs, families, setup_time):
    time = 0
    schedule = []
    remaining = set(all jobs)
    while remaining is not empty:
        for each family f with unscheduled jobs:
            batch_f = jobs in f that are still in remaining
            p_batch = sum of processing times in batch_f
            w_batch = sum of weights in batch_f
            d_min   = min due date in batch_f
            slack   = max(0, d_min - time - p_batch - setup_time)
            I_f     = (w_batch / p_batch) * exp(-slack / (K * p_avg))
                      where K is a lookahead parameter, p_avg = avg processing time
        f* = family with highest index I_f
        sequence jobs in f* by WSPT (ascending p_j / w_j)
        time += setup_time
        for each job j in batch_f* (WSPT order):
            time += p_j
            C_j = time
            schedule.append(j)
        remaining -= batch_f*
    return schedule
```

---

## 4. Illustrative Instance

Single machine, setup time $s = 2$ between families:

| Job | Family | $p_j$ | $w_j$ |
|-----|--------|--------|--------|
| J1 | A | 3 | 4 |
| J2 | A | 2 | 3 |
| J3 | B | 5 | 2 |
| J4 | B | 1 | 6 |

Greedy WSPT-batch: Family A ratio = 7/5 = 1.4, Family B ratio = 8/6 = 1.33. Process A first: [J2, J1] then setup, then B: [J4, J3]. Completion times: J2=2, J1=5, J4=8, J3=13. Total weighted completion = 3(2) + 4(5) + 6(8) + 2(13) = 6 + 20 + 48 + 26 = 100.

---

## 5. Applications

- **Semiconductor fabrication:** Wafer lots from the same product family share oven setups; sequencing families minimizes furnace changeovers.
- **Pharmaceutical manufacturing:** Tablet compression runs grouped by active ingredient to reduce cleaning validation time.
- **Chemical processing:** Batch reactors require family-dependent purging; grouping similar compounds reduces setup overhead.
- **Printing:** Print jobs grouped by ink color or paper stock to minimize press changeover.

---

## 6. Implementations in This Repository

```
batch_scheduling/
├── instance.py                    # BatchSchedulingInstance, BatchSchedulingSolution
├── heuristics/
│   └── greedy_batch.py            # Greedy batch scheduling
└── tests/
    └── test_batch.py              # Batch scheduling test suite
```

---

## 7. Key References

- Potts, C.N. & Kovalyov, M.Y. (2000). Scheduling with batching: A review. *European J. Oper. Res.*, 120(2), 228-249.
- Mason, S.J. & Chen, J. (2010). Scheduling batch processing machines in complex job shops. *IIE Transactions*, 42(10), 700-713.
- Mehta, S.V. & Uzsoy, R. (1998). Predictable scheduling of a job shop subject to breakdowns. *IEEE Trans. Robot. Autom.*, 14(3), 365-378.

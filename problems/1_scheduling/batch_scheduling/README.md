# Batch Scheduling Problem

## 1. Problem Definition

- **Input:** $n$ jobs with processing times $p_j$, weights $w_j$, family memberships; setup time $s$ between families; single machine
- **Decision:** Partition jobs into batches (same family) and sequence batches
- **Objective:** Minimize total weighted completion time $\sum w_j C_j$
- **Classification:** NP-hard

### Scheduling Notation

$1 \mid \text{batch}, s_{\text{fam}} \mid \sum w_j C_j$

---

## 2. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy batch | Heuristic | $O(n \log n)$ | WSPT within families, sequence families by aggregate ratio |

---

## 3. Implementations in This Repository

```
batch_scheduling/
├── instance.py                    # BatchSchedulingInstance, BatchSchedulingSolution
├── heuristics/
│   └── greedy_batch.py            # Greedy batch scheduling
└── tests/
    └── test_batch.py              # Batch scheduling test suite
```

---

## 4. Key References

- Potts, C.N. & Kovalyov, M.Y. (2000). Scheduling with batching: A review. *European J. Oper. Res.*, 120(2), 228-249.

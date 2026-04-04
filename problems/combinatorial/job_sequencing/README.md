# Job Sequencing with Deadlines

## 1. Problem Definition

- **Input:** $n$ jobs with processing times $p_j$, deadlines $d_j$, and profits $w_j$; single machine
- **Decision:** Which jobs to accept and in what order to process them
- **Objective:** Maximize total profit of jobs completed by their deadlines
- **Constraints:** Each accepted job must finish by its deadline; unit-time variant or general processing times
- **Classification:** NP-hard (weighted). $O(n \log n)$ for unit-time jobs via greedy.

### Scheduling Notation

$1 \mid d_j \mid \sum w_j U_j$ (minimize weighted number of late jobs, equivalent to maximizing on-time profit)

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of jobs |
| $p_j$ | Processing time of job $j$ |
| $d_j$ | Deadline of job $j$ |
| $w_j$ | Profit (weight) of job $j$ |
| $U_j$ | Late indicator: 1 if job $j$ is late, 0 if on time |

### ILP Formulation

$$\max \sum_{j=1}^{n} w_j (1 - U_j) \tag{1}$$

subject to scheduling feasibility: accepted jobs ($U_j = 0$) can be ordered so each finishes by its deadline.

### Small Illustrative Instance

```
n = 4, unit processing times (p_j = 1)
Jobs: (profit, deadline) = [(20, 2), (15, 1), (10, 2), (5, 3)]
Greedy by profit: accept job 1 (slot 1), job 2 can't fit slot 1 → slot 1 taken
  Accept jobs {1, 3, 4}: profit = 20 + 10 + 5 = 35
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Greedy (unit time) | Heuristic | $O(n \log n)$ | Sort by profit, assign to latest available slot |
| Greedy (general) | Heuristic | $O(n \log n)$ | EDD-based with profit-weighted selection |

### Greedy (Unit Processing Time)

Sort jobs by profit descending. For each job, assign it to the latest available time slot before its deadline. Uses a union-find or simple array for slot assignment.

#### Greedy Job Sequencing Pseudocode

```
GREEDY_JOB_SEQUENCING(jobs):
    sort jobs by profit descending
    max_deadline = max(d_j for all jobs)
    slot[1..max_deadline] = FREE

    accepted = []
    for each job j in sorted order:
        for t = min(d_j, max_deadline) downto 1:
            if slot[t] == FREE:
                slot[t] = j
                accepted.append(j)
                break
    return accepted, sum of profits of accepted jobs
```

**Complexity:** $O(n \log n)$ with union-find for slot lookup; $O(n \cdot d_{\max})$ with simple array scan.

**Optimality:** For unit-time jobs, this greedy approach produces an optimal solution (matroid structure -- the set of feasible job subsets forms a matroid).

### Applications

- **Batch manufacturing** with product-specific deadlines and varying profitability
- **Project selection** under deadline constraints with limited capacity
- **Advertisement scheduling** (maximize revenue from time-slot ads with expiry dates)

---

## 4. Implementations in This Repository

```
job_sequencing/
├── instance.py                    # JobSequencingInstance, JobSequencingSolution
│                                  #   - Fields: n, processing_times, deadlines, profits
├── heuristics/
│   └── greedy_js.py               # Greedy profit-based sequencing
└── tests/
    └── test_job_sequencing.py     # Job sequencing test suite
```

---

## 5. Key References

- Moore, J.M. (1968). An $n$ job, one machine sequencing algorithm for minimizing the number of late jobs. *Management Science*, 15(1), 102-109.
- Lawler, E.L. & Moore, J.M. (1969). A functional equation and its application to resource allocation and sequencing problems. *Management Science*, 16(1), 77-84.
- Sahni, S.K. (1976). Algorithms for scheduling independent tasks. *JACM*, 23(1), 116-127.
- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems*. 5th ed. Springer.

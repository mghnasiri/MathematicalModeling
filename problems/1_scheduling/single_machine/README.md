# Single Machine Scheduling (1 | beta | gamma)

## 1. Problem Definition

- **Input:**
  - A set $N = \{1, 2, \ldots, n\}$ of jobs
  - Processing times $p_j > 0$, weights $w_j \geq 0$, due dates $d_j$, release dates $r_j \geq 0$
- **Decision:** Find a permutation $\pi = (\pi(1), \pi(2), \ldots, \pi(n))$ of jobs
- **Objective:** Minimize a scheduling objective (see Section 2)
- **Constraints:** One machine processes one job at a time. No preemption (unless specified). Completion times: $C_{\pi(1)} = r_{\pi(1)} + p_{\pi(1)}$, $C_{\pi(k)} = \max(C_{\pi(k-1)}, r_{\pi(k)}) + p_{\pi(k)}$
- **Classification:** The single machine is the foundational scheduling environment. Complexity ranges from trivial to strongly NP-hard depending on the objective.

### Complexity Landscape

| Problem | Complexity | Optimal Rule / Method | Reference |
|---------|-----------|----------------------|-----------|
| $1 \mid\mid C_{\max}$ | Trivial | Any order ($C_{\max} = \sum p_j$) | — |
| $1 \mid\mid \sum C_j$ | $O(n \log n)$ | SPT (Shortest Processing Time) | Conway et al. (1967) |
| $1 \mid\mid \sum w_j C_j$ | $O(n \log n)$ | WSPT (Smith's Rule: sort by $p_j/w_j$) | Smith (1956) |
| $1 \mid\mid L_{\max}$ | $O(n \log n)$ | EDD (Earliest Due Date) | Jackson (1955) |
| $1 \mid\mid \sum U_j$ | $O(n \log n)$ | Moore's Algorithm | Moore (1968) |
| $1 \mid\mid \sum T_j$ | NP-hard | Bitmask DP $O(2^n \cdot n)$ | Du & Leung (1990) |
| $1 \mid\mid \sum w_j T_j$ | Strongly NP-hard | B&B, ATC heuristic | Potts & Van Wassenhove (1985) |
| $1 \mid r_j \mid \sum C_j$ | NP-hard | B&B | Lenstra et al. (1977) |
| $1 \mid r_j \mid L_{\max}$ | NP-hard (preemptive: poly) | SRPT for preemptive | Schrage (1968) |
| $1 \mid prec \mid \sum w_j C_j$ | NP-hard | B&B | Lawler (1978) |
| $1 \mid s_{jk} \mid C_{\max}$ | NP-hard (reduces to TSP) | TSP methods | — |

The single machine is the **foundation** of scheduling theory — every dispatching rule here (SPT, EDD, WSPT) becomes a subroutine for multi-machine algorithms.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $p_j$ | Processing time of job $j$ | $\mathbb{R}_{>0}$ |
| $w_j$ | Weight (priority) of job $j$ | $\mathbb{R}_{\geq 0}$ |
| $d_j$ | Due date of job $j$ | $\mathbb{R}$ |
| $r_j$ | Release date of job $j$ | $\mathbb{R}_{\geq 0}$ |
| $C_j$ | Completion time of job $j$ | $\mathbb{R}_{>0}$ |
| $T_j$ | Tardiness: $T_j = \max(0, C_j - d_j)$ | $\mathbb{R}_{\geq 0}$ |
| $U_j$ | Unit penalty: $U_j = 1$ if $C_j > d_j$, else 0 | $\{0, 1\}$ |
| $L_j$ | Lateness: $L_j = C_j - d_j$ | $\mathbb{R}$ |

### Weighted Completion Time ($1 \mid\mid \sum w_j C_j$)

Optimality of WSPT: For any two adjacent jobs $j, k$ in the schedule, swapping them improves the objective if and only if $p_j / w_j > p_k / w_k$. Therefore, sorting by increasing $p_j / w_j$ (Smith's ratio rule) is optimal.

### Total Weighted Tardiness ($1 \mid\mid \sum w_j T_j$) — MILP

$$\min \sum_{j=1}^{n} w_j T_j \tag{1}$$

$$\text{s.t.} \quad C_j = \sum_{k=1}^{n} p_j \cdot x_{jk} \cdot k' \quad \text{(linearized via position variables)} \tag{2}$$

$$T_j \geq C_j - d_j \quad \forall j \tag{3}$$

$$T_j \geq 0 \quad \forall j \tag{4}$$

In practice, the position-based MILP is weak; B&B with the ATC heuristic as warm-start is more effective.

### Total Tardiness ($1 \mid\mid \sum T_j$) — MILP with Precedence Variables

An alternative MILP uses binary precedence variables $\delta_{jk} \in \{0,1\}$ where $\delta_{jk} = 1$ if job $j$ precedes job $k$:

$$\min \sum_{j=1}^{n} T_j \tag{5}$$

$$\text{s.t.} \quad C_j \geq p_j \quad \forall j \tag{6}$$

$$C_j \geq C_k + p_j - M(1 - \delta_{jk}) \quad \forall j \neq k \tag{7}$$

$$\delta_{jk} + \delta_{kj} = 1 \quad \forall j < k \tag{8}$$

$$T_j \geq C_j - d_j \quad \forall j \tag{9}$$

$$T_j \geq 0, \quad \delta_{jk} \in \{0, 1\} \quad \forall j, k \tag{10}$$

Here $M$ is a sufficiently large constant (e.g., $M = \sum_{j} p_j$). Constraint (7) is a disjunctive precedence constraint: if $\delta_{jk} = 1$ (job $k$ precedes $j$), then $C_j \geq C_k + p_j$. Constraint (8) ensures exactly one ordering per pair. This formulation has $O(n^2)$ binary variables and $O(n^2)$ constraints; the LP relaxation is tighter than the position-based MILP but still weak for large $n$.

### Bitmask DP Recurrence ($1 \mid\mid \sum T_j$)

The bitmask DP encodes the set of already-scheduled jobs as a binary mask $S \subseteq N$. The recurrence builds the schedule forward:

$$f(S) = \min_{j \in S} \left\{ f(S \setminus \{j\}) + \max\!\left(0,\; t(S) - d_j\right) \right\}$$

where $t(S) = \sum_{j \in S} p_j$ is the completion time after scheduling all jobs in $S$.

**Base case:** $f(\emptyset) = 0$.

**Optimal objective:** $f(N)$ where $N = \{1, \ldots, n\}$.

The key insight is that $t(S)$ depends only on the *set* $S$, not the ordering, because all jobs are available from time 0 (no release dates). Each job $j \in S$ is tried as the *last* job scheduled; job $j$ then completes at time $t(S)$ and incurs tardiness $\max(0, t(S) - d_j)$. There are $2^n$ states and each requires $O(n)$ transitions, yielding $O(2^n \cdot n)$ total.

### LP Relaxation and Preemptive Bounds

For NP-hard single machine problems, LP relaxation bounds are important for exact methods:

- **Preemptive relaxation for $1 \mid\mid \sum T_j$:** Allowing preemption makes some problems easier. The preemptive variant $1 \mid pmtn \mid \sum T_j$ remains NP-hard (Du & Leung, 1990), so the preemptive bound does not help directly for total tardiness. However, $1 \mid pmtn \mid L_{\max}$ is solvable by EDD and gives a valid lower bound on the non-preemptive $L_{\max}$.

- **Linear relaxation of the precedence MILP:** Relaxing $\delta_{jk} \in \{0,1\}$ to $\delta_{jk} \in [0,1]$ gives a weak LP bound. Tighter LP bounds can be obtained from the time-indexed formulation, where binary variable $x_{jt} = 1$ if job $j$ starts at time $t$. The LP relaxation of the time-indexed model provides stronger bounds but has $O(n \cdot T)$ variables where $T = \sum p_j$.

- **EDD bound for weighted tardiness:** The B&B implementation in this repository uses an EDD-based lower bound: sort remaining unscheduled jobs by EDD, schedule them after the current partial sequence, and compute the resulting weighted tardiness as a lower bound (since EDD minimizes $L_{\max}$, it tends to underestimate tardiness contributions).

---

## 3. Variants

| Variant | Directory | Notation | Key Difference |
|---------|-----------|----------|---------------|
| Preemptive | `variants/preemptive/` | $1 \mid pmtn, r_j \mid \sum C_j$ | Jobs can be interrupted and resumed |
| Batch | `variants/batch/` | $1 \mid batch, s_j \mid \sum w_j C_j$ | Jobs grouped into batches with setup times |

### 3.1 Preemptive Scheduling

Jobs can be interrupted and resumed later without penalty. With release dates, the preemptive case $1 \mid pmtn, r_j \mid L_{\max}$ is solvable in $O(n \log n)$ via Shortest Remaining Processing Time (SRPT), while the non-preemptive case is NP-hard.

### 3.2 Batch Scheduling

Jobs are grouped into batches. A setup time occurs between consecutive batches of different types. Objective typically involves minimizing total weighted completion time across all batches.

---

## 4. Benchmark Instances

Single machine scheduling benchmarks are typically generated randomly:

- **OR-Library** (Beasley, 1990): Weighted tardiness instances with $n = 40, 50, 100$.
- **Random generation:** Processing times $p_j \sim U[1, 100]$, due dates $d_j \sim U[P(1-\tau-R/2), P(1-\tau+R/2)]$ where $P = \sum p_j$, $\tau$ = tardiness factor, $R$ = due date range.

### OR-Library Weighted Tardiness Instances

The standard benchmark set for $1 \mid\mid \sum w_j T_j$ was introduced by Potts and Van Wassenhove (1985) and later extended. Key instance families:

- **Potts & Van Wassenhove (1985):** Instances with $n = 10, 20, 30, 40, 50$. Processing times $p_j \sim U[1, 100]$, weights $w_j \sim U[1, 10]$, due dates generated with tardiness factor $\tau$ and due date range $R$. These form the basis for evaluating B&B and heuristic methods.
- **OR-Library (Beasley, 1990):** Hosts standard weighted tardiness instances at [http://people.brunel.ac.uk/~mastjjb/jeb/info.html](http://people.brunel.ac.uk/~mastjjb/jeb/info.html). Includes instances with $n = 40, 50, 100$ across multiple $(\tau, R)$ combinations.
- **Crauwels et al. (1998) instances:** Extended benchmark set with $n$ up to 100 for $1 \mid\mid \sum w_j T_j$, used to evaluate tabu search and genetic algorithm approaches. [TODO: verify exact URL for these instances]

Best-known solutions (BKS) for the OR-Library instances are maintained by various authors. For $n \leq 50$, optimal solutions are known from B&B; for $n = 100$, the best-known values come from metaheuristic competitions.

### Instance Generation Parameters

The standard random generation scheme (Pinedo convention) uses two parameters:
- **Tardiness factor** $\tau \in \{0.2, 0.4, 0.6, 0.8\}$: higher $\tau$ means tighter due dates (more tardiness).
- **Due date range** $R \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$: higher $R$ means more spread in due dates.

Due dates are drawn as $d_j \sim U\left[P(1 - \tau - R/2),\; P(1 - \tau + R/2)\right]$ where $P = \sum p_j$. The implementation in `instance.py` uses `tardiness_factor` and `due_date_range` parameters matching this convention.

### Small Illustrative Instance

```
5 jobs: p = [4, 3, 7, 2, 6], w = [5, 2, 1, 4, 3], d = [8, 12, 10, 5, 15]
SPT order: [4, 2, 1, 5, 3] (sort by p_j)
WSPT order: [4, 1, 5, 2, 3] (sort by p_j/w_j = [0.8, 1.5, 7.0, 0.5, 2.0])
EDD order: [4, 1, 3, 2, 5] (sort by d_j)
```

---

## 5. Solution Methods

### 5.1 Exact Methods

#### Polynomial-Time Dispatching Rules

| Rule | Sorts By | Optimizes | Complexity |
|------|----------|-----------|-----------|
| SPT | Ascending $p_j$ | $\sum C_j$ | $O(n \log n)$ |
| WSPT | Ascending $p_j/w_j$ | $\sum w_j C_j$ | $O(n \log n)$ |
| EDD | Ascending $d_j$ | $L_{\max}$ | $O(n \log n)$ |
| LPT | Descending $p_j$ | Load balancing (used in parallel machine) | $O(n \log n)$ |

#### Moore's Algorithm ($1 \mid\mid \sum U_j$)

**Idea:** Process jobs in EDD order. Whenever a job is tardy, remove the longest job processed so far (it goes to the "late" set). The remaining jobs are on time.

**Complexity:** $O(n \log n)$ using a max-heap for the removal step.

```
ALGORITHM Moore(p[1..n], d[1..n])
  Sort jobs by ascending d_j → sequence S
  on_time ← [], late ← [], t ← 0
  FOR j in S:
    on_time.append(j), t ← t + p[j]
    IF t > d[j]:
      longest ← argmax p[k] for k in on_time
      Remove longest from on_time, add to late
      t ← t - p[longest]
  RETURN on_time + late   (on_time in EDD, late in any order)
```

**Detailed pseudocode with max-heap operations** (as implemented in `moores_algorithm.py`):

```
ALGORITHM Moore_Heap(p[1..n], d[1..n])
  Sort jobs by ascending d[j] → EDD order S = (s_1, s_2, ..., s_n)
  H ← empty max-heap (keyed on processing time)
  on_time_set ← ∅
  t ← 0

  FOR i = 1 TO n:
    j ← S[i]
    t ← t + p[j]
    HEAP-INSERT(H, (p[j], j))          // O(log n)
    on_time_set ← on_time_set ∪ {j}

    IF t > d[j]:                        // job j causes tardiness
      (p_max, longest) ← HEAP-EXTRACT-MAX(H)   // O(log n)
      on_time_set ← on_time_set \ {longest}
      t ← t − p_max                    // remove longest job's contribution

  on_time_seq ← [j for j in S if j ∈ on_time_set]   // preserve EDD order
  late_seq ← [j for j in S if j ∉ on_time_set]
  RETURN on_time_seq + late_seq
```

**Correctness sketch:** At each step, the algorithm maintains the invariant that all jobs in `on_time_set` can complete on time. When adding a new EDD job causes a violation, removing the longest job (regardless of which job it is) maximally reduces the total processing time while only adding one job to the late set. This greedy exchange is provably optimal by a matroid argument: the set of on-time job collections forms a matroid, and Moore's algorithm is the greedy algorithm on this matroid (Moore, 1968).

**Why the removed job is the longest, not the current job:** The current job $j$ might have a small $p_j$ but a tight $d_j$, while some earlier job $k$ has a large $p_k$. Removing $k$ frees more time for future jobs, leading to fewer total tardy jobs.

#### Apparent Tardiness Cost (ATC) — Heuristic for $1 \mid\mid \sum w_j T_j$

**Idea:** Composite dispatching rule combining WSPT ratio with due-date urgency. At each step, schedule the job maximizing:

$$I_j(t) = \frac{w_j}{p_j} \exp\left(-\frac{\max(d_j - p_j - t, 0)}{K \bar{p}}\right)$$

where $K$ is a look-ahead parameter (typically $K \in [1, 3]$) and $\bar{p}$ is the average processing time of remaining jobs.

**Complexity:** $O(n^2)$. Quality: typically within 1-5% of optimal for $n \leq 100$.

**ATC Priority Index Derivation.**
The ATC rule is a composite of two simpler rules:
1. **WSPT component** ($w_j / p_j$): optimal for $1 \mid\mid \sum w_j C_j$, captures the priority-to-cost ratio.
2. **Urgency component** ($\exp(-\text{slack}_j / (K \bar{p}))$): an exponential decay penalizing jobs with large positive slack $\text{slack}_j = \max(d_j - p_j - t, 0)$.

The slack $d_j - p_j - t$ represents the time available before job $j$ *must* start to avoid tardiness. When slack is zero or negative (job is already late or about to be), the exponential term equals 1 and ATC reduces to WSPT. When slack is large, the exponential term approaches 0, deprioritizing the job regardless of its WSPT ratio.

The full priority index is:

$$I_j(t) = \frac{w_j}{p_j} \cdot \exp\!\left(-\frac{\max(d_j - p_j - t,\; 0)}{K \cdot \bar{p}}\right)$$

where $\bar{p} = \frac{1}{|U|}\sum_{k \in U} p_k$ is the mean processing time of unscheduled jobs $U$.

**The parameter $K$ (look-ahead scaling).**
$K$ controls how aggressively ATC weights due-date urgency versus the WSPT ratio:

| $K$ value | Behavior | Best suited for |
|-----------|----------|-----------------|
| $K < 1$ | Strong urgency bias; nearly EDD-like | Very tight due dates (high tardiness factor) |
| $K \approx 1.5{-}2.0$ | Balanced; good general-purpose setting | Moderate due date tightness |
| $K \approx 2.0{-}3.0$ | WSPT-dominant; urgency is secondary | Loose due dates (low tardiness factor) |
| $K \to \infty$ | Pure WSPT (urgency term $\to 1$) | No due date urgency relevant |

Vepsalainen and Morton (1987) recommend $K \approx 2$ as a robust default. The implementation in this repository uses $K = 2.0$ by default. For automated tuning, grid search over $K \in \{0.5, 1.0, 1.5, 2.0, 3.0, 5.0\}$ and selecting the $K$ yielding the lowest $\sum w_j T_j$ is a practical approach (see `__main__` block in `apparent_tardiness_cost.py`).

#### Dynamic Programming ($1 \mid\mid \sum T_j$)

**Idea:** Bitmask DP. State: $(S, t)$ where $S$ is the set of scheduled jobs and $t$ is the current time. Transition: add any unscheduled job.

**Complexity:** $O(2^n \cdot n)$. **Practical limit:** $n \leq 20$.

#### Branch and Bound ($1 \mid\mid \sum w_j T_j$)

DFS with ATC warm-start upper bound. Lower bound: relaxation of due date constraints or WSPT on remaining jobs. Effective up to $n \approx 50{-}100$ depending on due date tightness.

```
ALGORITHM BranchAndBound_wTj(p[1..n], w[1..n], d[1..n])
  UB ← ATC(p, w, d).objective                // warm-start upper bound
  best_seq ← ATC solution
  scheduled ← ∅, t ← 0, cost ← 0

  FUNCTION DFS(scheduled, t, cost):
    IF |scheduled| = n:
      IF cost < UB: UB ← cost, best_seq ← scheduled
      RETURN
    remaining ← {1..n} \ scheduled
    Sort remaining by ascending p_j/w_j       // WSPT branching order
    FOR j in remaining:
      new_t ← t + p[j]
      new_cost ← cost + w[j] × max(0, new_t − d[j])
      // Lower bound: schedule remaining\{j} by EDD after new_t
      LB ← new_cost + EDD_tardiness(remaining\{j}, new_t, w, d)
      IF LB < UB:
        DFS(scheduled ∪ {j}, new_t, new_cost)

  DFS(∅, 0, 0)
  RETURN (best_seq, UB)
```

The EDD-based lower bound sorts remaining jobs by due date and computes the resulting weighted tardiness, which underestimates the true optimum because EDD is optimal for $L_{\max}$ (and thus tends to reduce tardiness).

#### Bitmask DP Pseudocode ($1 \mid\mid \sum T_j$)

```
ALGORITHM BitmaskDP_Tj(p[1..n], d[1..n])
  total_p ← Σ p[j]
  dp[0] ← 0                                  // empty set: zero cost
  last[0] ← −1

  FOR mask = 1 TO 2^n − 1:
    t ← Σ { p[j] : bit j is set in mask }    // completion time of set
    dp[mask] ← ∞
    FOR EACH j such that bit j is set in mask:
      prev ← mask XOR (1 << j)               // remove j from set
      IF dp[prev] = ∞: CONTINUE
      tardiness_j ← max(0, t − d[j])
      candidate ← dp[prev] + tardiness_j
      IF candidate < dp[mask]:
        dp[mask] ← candidate
        last[mask] ← j

  // Backtrack: follow last[] pointers from full_mask to 0
  seq ← [], mask ← 2^n − 1
  WHILE mask > 0:
    seq.append(last[mask])
    mask ← mask XOR (1 << last[mask])
  REVERSE(seq)
  RETURN (seq, dp[2^n − 1])
```

**Space:** $O(2^n)$ for the `dp` and `last` arrays. **Time:** $O(2^n \cdot n)$. Practical for $n \leq 20$; at $n = 20$ the state space is roughly $10^6$ entries.

### 5.2 Metaheuristics (for NP-hard objectives)

This repository implements **6 metaheuristics** for single machine:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Swap, insertion neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | Swap/insertion with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Swap neighborhood with recency tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove + reinsert jobs via ATC |
| 5 | Genetic Algorithm (GA) | Population | Permutation encoding, OX crossover |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Swap → insertion → block-move |

#### SA Parameter Table (as implemented)

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Cooling rate | $\alpha$ | 0.995 | Geometric: $T_{k+1} = \alpha \cdot T_k$ |
| Initial temperature factor | — | 0.5 | $T_0 = 0.5 \cdot \bar{P}/n$ where $\bar{P} = \sum p_j$ |
| Epoch length | $L$ | $n^2$ | Iterations per temperature level |
| Neighborhood | — | 50/50 | 50% swap, 50% insertion moves |
| Acceptance | — | Boltzmann | $P(\text{accept}) = \exp(-\Delta / T)$ for $\Delta > 0$ |
| Stopping | — | 50 000 iter | Or time limit, or $T < 10^{-10}$ |
| Initial solution | — | ATC ($\sum w_j T_j$), EDD ($\sum T_j$) | Warm-start from constructive heuristic |

#### Dispatching Rule Optimality Proofs

**SPT optimality for $1 \mid\mid \sum C_j$ (exchange argument).**
Consider two adjacent jobs $j$ and $k$ in a schedule. If $j$ is in position $i$ and $k$ in position $i+1$, their contribution to $\sum C_j$ is: $C_j + C_k = (t + p_j) + (t + p_j + p_k)$ where $t$ is the completion time before position $i$. Swapping gives $(t + p_k) + (t + p_k + p_j)$. The swap is beneficial iff $(t + p_k) + (t + p_k + p_j) < (t + p_j) + (t + p_j + p_k)$, which simplifies to $p_k < p_j$. Therefore, the optimal schedule sorts by ascending $p_j$ (SPT). By transitivity, any non-SPT schedule can be improved by swapping an adjacent pair, proving SPT is globally optimal.

**WSPT optimality for $1 \mid\mid \sum w_j C_j$ (Smith's exchange argument).**
For adjacent jobs $j$ (position $i$) and $k$ (position $i+1$), their weighted contribution is $w_j(t + p_j) + w_k(t + p_j + p_k)$. Swapping gives $w_k(t + p_k) + w_j(t + p_k + p_j)$. The difference is $w_j p_k - w_k p_j$. The original order is preferred iff $w_j p_k \geq w_k p_j$, i.e., $p_j/w_j \leq p_k/w_k$. Therefore, sorting by ascending Smith's ratio $p_j/w_j$ is optimal. This is Smith's Rule (1956).

**EDD optimality for $1 \mid\mid L_{\max}$ (exchange argument).**
For adjacent jobs $j, k$ with $d_j \leq d_k$: scheduling $j$ before $k$ gives $L_{\max} \geq \max(t + p_j - d_j, t + p_j + p_k - d_k)$, while the reverse gives $L_{\max} \geq \max(t + p_k - d_k, t + p_k + p_j - d_j)$. Since $d_j \leq d_k$, we have $t + p_j + p_k - d_j \geq t + p_j + p_k - d_k$, so the second arrangement cannot improve $L_{\max}$. By induction, EDD order minimizes $L_{\max}$ (Jackson, 1955).

---

## 6. Implementation Guide

### Modeling Tips

- **Completion time computation:** Given a permutation $\pi$, completion times are computed in $O(n)$: $C_{\pi(1)} = p_{\pi(1)}$, $C_{\pi(k)} = C_{\pi(k-1)} + p_{\pi(k)}$.
- **Tardiness:** $T_j = \max(0, C_j - d_j)$. Pre-compute completion times first, then tardiness.
- **ATC parameter $K$:** Start with $K = 2$. Lower values increase greediness (good for tight due dates); higher values make ATC approach WSPT.

### Common Pitfalls

- **WSPT ties:** When $p_j/w_j = p_k/w_k$, either order is optimal for $\sum w_j C_j$. But for $\sum w_j T_j$, tie-breaking matters — use EDD among tied jobs.
- **Moore's Algorithm correctness:** The removed job must be the longest *among all on-time jobs so far*, not the current job. Using a max-heap is essential.

---

## 7. Computational Results Summary

| Method | Objective | Gap (n=20) | Gap (n=100) |
|--------|-----------|-----------|-------------|
| SPT/WSPT/EDD | Polynomial objectives | 0% | 0% |
| Moore's | $\sum U_j$ | 0% | 0% |
| ATC | $\sum w_j T_j$ | 1-5% | 2-8% |
| DP (bitmask) | $\sum T_j$ | 0% | Infeasible |
| B&B + ATC | $\sum w_j T_j$ | 0% | 0% (minutes) |
| SA | $\sum w_j T_j$ | <1% | 1-3% |
| GA | $\sum w_j T_j$ | <1% | 1-3% |

---

## 8. Implementations in This Repository

```
single_machine/
├── instance.py                        # SingleMachineInstance, objective functions
│
├── exact/
│   ├── dynamic_programming.py         # Bitmask DP for 1||ΣTj — O(2^n × n)
│   └── branch_and_bound.py            # B&B for 1||ΣwjTj, ATC warm-start
│
├── heuristics/
│   ├── dispatching_rules.py           # SPT, WSPT, EDD, LPT
│   ├── moores_algorithm.py            # Moore's for 1||ΣUj — O(n log n)
│   └── apparent_tardiness_cost.py     # ATC for 1||ΣwjTj — O(n²)
│
├── metaheuristics/
│   ├── local_search.py                # Swap, insertion neighborhoods
│   ├── simulated_annealing.py         # SA for ΣwjTj and ΣTj
│   ├── tabu_search.py                 # TS with recency tabu
│   ├── iterated_greedy.py             # IG: remove + ATC reinsert
│   ├── genetic_algorithm.py           # GA: permutation encoding, OX
│   └── vns.py                         # VNS: swap → insertion → block-move
│
├── variants/
│   ├── preemptive/                    # 1 | pmtn, rj | ΣCj
│   └── batch/                         # 1 | batch, sj | ΣwjCj
│
└── tests/                             # 6 test files
    ├── test_single_machine.py         # Core algorithms
    ├── test_sm_tabu_search.py         # Tabu Search
    ├── test_sm_ga.py                  # Genetic Algorithm
    ├── test_sm_ig.py                  # Iterated Greedy
    ├── test_sm_ls.py                  # Local Search
    └── test_sm_vns.py                 # VNS
```

**Total:** 2 exact methods, 4 dispatching rules + Moore's + ATC, 6 metaheuristics/LS, 2 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Smith, W.E. (1956). Various optimizers for single-stage production. *Naval Research Logistics Quarterly*, 3(1-2), 59-66. DOI: [10.1002/nav.3800030106](https://doi.org/10.1002/nav.3800030106). -- Introduces the WSPT rule (Smith's ratio) and proves its optimality for $1 \mid\mid \sum w_j C_j$ via the pairwise exchange argument.
- Jackson, J.R. (1955). Scheduling a production line to minimize maximum tardiness. *Management Science Research Project, Research Report 43, UCLA*. -- Proves EDD is optimal for $1 \mid\mid L_{\max}$.
- Moore, J.M. (1968). An $n$ job, one machine sequencing algorithm for minimizing the number of late jobs. *Management Science*, 15(1), 102-109. DOI: [10.1287/mnsc.15.1.102](https://doi.org/10.1287/mnsc.15.1.102). -- Optimal $O(n \log n)$ algorithm for $1 \mid\mid \sum U_j$ using greedy EDD with longest-job removal.
- Conway, R.W., Maxwell, W.L. & Miller, L.W. (1967). *Theory of Scheduling*. Addison-Wesley. -- Foundational textbook; establishes SPT optimality for $\sum C_j$ and provides the formal framework for dispatching rules.
- Lawler, E.L. (1977). A pseudopolynomial algorithm for sequencing jobs to minimize total tardiness. *Annals of Discrete Mathematics*, 1, 331-342. DOI: [10.1016/S0167-5060(08)70742-8](https://doi.org/10.1016/S0167-5060(08)70742-8). -- Introduces the decomposition-based DP and proves $1 \mid\mid \sum T_j$ admits a pseudo-polynomial solution.
- Du, J. & Leung, J.Y.T. (1990). Minimizing total tardiness on one machine is NP-hard. *Mathematics of Operations Research*, 15(3), 483-495. DOI: [10.1287/moor.15.3.483](https://doi.org/10.1287/moor.15.3.483). -- Settles the complexity of $1 \mid\mid \sum T_j$ as (ordinary) NP-hard, complementing Lawler's pseudo-polynomial algorithm.
- Lenstra, J.K., Rinnooy Kan, A.H.G. & Brucker, P. (1977). Complexity of machine scheduling problems. *Annals of Discrete Mathematics*, 1, 343-362. DOI: [10.1016/S0167-5060(08)70743-X](https://doi.org/10.1016/S0167-5060(08)70743-X). -- Comprehensive complexity classification of scheduling problems; proves $1 \mid r_j \mid \sum C_j$ is NP-hard.

### Key Methods

- Potts, C.N. & Van Wassenhove, L.N. (1985). A branch and bound algorithm for the total weighted tardiness problem. *Operations Research*, 33(2), 363-377. DOI: [10.1287/opre.33.2.363](https://doi.org/10.1287/opre.33.2.363). -- Seminal B&B for $1 \mid\mid \sum w_j T_j$ with dominance rules and lower bounds; also provides the standard benchmark instances.
- Vepsalainen, A.P.J. & Morton, T.E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035-1047. DOI: [10.1287/mnsc.33.8.1035](https://doi.org/10.1287/mnsc.33.8.1035). -- Introduces the ATC (Apparent Tardiness Cost) dispatching rule combining WSPT ratio with exponential urgency.
- Potts, C.N. & Van Wassenhove, L.N. (1991). Single machine tardiness sequencing heuristics. *IIE Transactions*, 23(4), 346-354. DOI: [10.1080/07408179108963868](https://doi.org/10.1080/07408179108963868). -- Evaluates neighborhood search heuristics for single machine tardiness; reference for SA implementation.
- Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680. DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671). -- Foundational SA paper; introduces the Boltzmann acceptance criterion and cooling schedule framework.
- Beasley, J.E. (1990). OR-Library: distributing test problems by electronic mail. *Journal of the Operational Research Society*, 41(11), 1069-1072. DOI: [10.1057/jors.1990.166](https://doi.org/10.1057/jors.1990.166). -- Describes the OR-Library, which hosts standard benchmark instances for weighted tardiness and many other OR problems.

### Textbook

- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer. DOI: [10.1007/978-3-319-26580-3](https://doi.org/10.1007/978-3-319-26580-3). -- Comprehensive reference covering all single machine objectives, complexity proofs, algorithms, and the $\alpha \mid \beta \mid \gamma$ notation. Chapters 3-5 cover single machine scheduling in detail.

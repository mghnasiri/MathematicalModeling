# Job Shop Scheduling (JSP)

## 1. Problem Definition

- **Input:**
  - A set of $n$ jobs $J = \{1, \ldots, n\}$ and $m$ machines $M = \{1, \ldots, m\}$
  - Each job $j$ has an ordered sequence of operations $O_{j1}, O_{j2}, \ldots, O_{j,n_j}$
  - Each operation $O_{jk}$ requires machine $\mu_{jk} \in M$ for $p_{jk}$ time units
  - Different jobs may visit machines in different orders (job-specific routing)
- **Decision:** Determine the start time $s_{jk}$ of each operation
- **Objective:** Minimize makespan $C_{\max} = \max_{j,k} (s_{jk} + p_{jk})$
- **Constraints:** (1) Operations within a job are processed in order (precedence). (2) Each machine processes at most one operation at a time (disjunctive). (3) No preemption.
- **Classification:** Strongly NP-hard combinatorial optimization
- **Scheduling notation:** $J_m \mid\mid C_{\max}$

### Complexity

| Problem | Complexity | Reference |
|---------|-----------|-----------|
| $J_2 \mid\mid C_{\max}$ | NP-hard | Lenstra & Rinnooy Kan (1979) |
| $J_m \mid\mid C_{\max}$ | Strongly NP-hard | Garey & Johnson (1979) |
| $J_2 \mid n{=}2 \mid C_{\max}$ | $O(n \log n)$ | Jackson's Rule |
| $F_m \mid\mid C_{\max}$ | Special case (same route) | Garey, Johnson & Sethi (1976) |

The 10x10 instance **ft10** (Fisher & Thompson, 1963) remained unsolved for **26 years** — the optimal makespan of 930 was proven in 1989 by Carlier & Pinson using constraint propagation.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of jobs | $\mathbb{Z}^+$ |
| $m$ | Number of machines | $\mathbb{Z}^+$ |
| $O_{jk}$ | The $k$-th operation of job $j$ | — |
| $\mu_{jk}$ | Machine required by $O_{jk}$ | $M$ |
| $p_{jk}$ | Processing time of $O_{jk}$ | $\mathbb{R}_{>0}$ |
| $s_{jk}$ | Start time of $O_{jk}$ | $\mathbb{R}_{\geq 0}$ |

### Disjunctive Formulation

$$\min \quad C_{\max} \tag{1}$$

**Precedence constraints** (operations within a job must be sequential):

$$s_{j,k+1} \geq s_{jk} + p_{jk} \quad \forall\, j,\; k = 1, \ldots, n_j{-}1 \tag{2}$$

**Disjunctive constraints** (no overlap on the same machine):

$$s_{jk} + p_{jk} \leq s_{j'k'} \quad \text{OR} \quad s_{j'k'} + p_{j'k'} \leq s_{jk} \tag{3}$$

for all pairs of operations $(O_{jk}, O_{j'k'})$ sharing machine $\mu_{jk} = \mu_{j'k'}$.

**Makespan definition:**

$$C_{\max} \geq s_{jk} + p_{jk} \quad \forall\, j, k \tag{4}$$

### Disjunctive Graph Model (Roy & Sussmann, 1964)

The central data structure for JSP. Formally:

**Definition.** The disjunctive graph $G = (V, C, D)$ consists of:

- **Node set** $V = \{0\} \cup \{O_{jk} : j = 1,\ldots,n;\; k = 1,\ldots,n_j\} \cup \{*\}$, where $0$ is the source (dummy start) and $*$ is the sink (dummy end).
- **Conjunctive arc set** $C$: directed arcs encoding job precedence.
  - $(0, O_{j1})$ with weight $0$ for all jobs $j$
  - $(O_{jk}, O_{j,k+1})$ with weight $p_{jk}$ for $k = 1,\ldots,n_j{-}1$
  - $(O_{j,n_j}, *)$ with weight $p_{j,n_j}$ for all jobs $j$
- **Disjunctive arc set** $D$: undirected arcs connecting every pair of operations sharing the same machine. For operations $O_{jk}$ and $O_{j'k'}$ with $\mu_{jk} = \mu_{j'k'}$, the pair $\{(O_{jk}, O_{j'k'}), (O_{j'k'}, O_{jk})\}$ with weights $p_{jk}$ and $p_{j'k'}$ respectively belongs to $D$.

**Schedule as orientation.** A feasible schedule corresponds to selecting exactly one arc from each disjunctive pair (orienting all disjunctive arcs) such that the resulting directed graph $G' = (V, C \cup D')$ is **acyclic** (a DAG).

**Makespan = longest path.** Given a feasible orientation $D'$, the makespan equals the length of the longest (weighted) path from $0$ to $*$ in $G'$. This longest path is the **critical path**, and can be computed in $O(|V| + |E|)$ via topological sort followed by longest-path dynamic programming.

**Critical block:** A maximal sequence of consecutive operations on the same machine that all lie on the critical path. Effective neighborhoods modify operations at the boundaries of critical blocks. If $B = (O_{a_1}, O_{a_2}, \ldots, O_{a_l})$ is a critical block on machine $i$, then only moves involving $O_{a_1}$ or $O_{a_l}$ (the block endpoints) can potentially reduce the makespan.

### Time-Indexed MIP Formulation

An alternative to the disjunctive formulation uses binary variables indexed by time. Let $T$ be a valid time horizon (e.g., sum of all processing times).

**Variables:**

$$x_{jkt} = \begin{cases} 1 & \text{if operation } O_{jk} \text{ starts at time } t \\ 0 & \text{otherwise} \end{cases}$$

**Objective:**

$$\min \quad C_{\max} \tag{5}$$

**Assignment** (each operation starts exactly once):

$$\sum_{t=0}^{T - p_{jk}} x_{jkt} = 1 \quad \forall\, j, k \tag{6}$$

**Precedence** (operations within a job are sequential):

$$\sum_{t=0}^{T - p_{jk}} t \cdot x_{jkt} + p_{jk} \leq \sum_{t=0}^{T - p_{j,k+1}} t \cdot x_{j,k+1,t} \quad \forall\, j,\; k = 1, \ldots, n_j{-}1 \tag{7}$$

**Machine capacity** (at most one operation per machine at each time):

$$\sum_{\substack{(j,k):\\ \mu_{jk} = i}} \sum_{\tau = \max(0, t - p_{jk} + 1)}^{t} x_{jk\tau} \leq 1 \quad \forall\, i \in M,\; t = 0, \ldots, T \tag{8}$$

**Makespan definition:**

$$C_{\max} \geq \sum_{t=0}^{T - p_{j,n_j}} t \cdot x_{j,n_j,t} + p_{j,n_j} \quad \forall\, j \tag{9}$$

### Formulation Comparison

| Formulation | Variables | Constraints | Strength | Practical Use |
|-------------|-----------|-------------|----------|---------------|
| Disjunctive (Big-M) | $O(n^2 m)$ binary | $O(n^2 m)$ | Weak LP relaxation | Conceptual, small instances |
| Time-indexed MIP | $O(n \cdot m \cdot T)$ binary | $O(m \cdot T + n \cdot m)$ | Tight LP relaxation | Moderate instances, large $T$ |
| CP (interval variables) | $n \cdot m$ intervals | $m$ no-overlap | Very strong propagation | State-of-the-art exact solver |
| Disjunctive graph | N/A (combinatorial) | N/A | Exact critical path | Heuristic/metaheuristic backbone |

---

## 3. Variants

| Variant | Directory | Key Difference |
|---------|-----------|---------------|
| No-Wait JSP | `variants/no_wait/` | Operations of a job must be processed back-to-back |
| Weighted Tardiness | `variants/weighted_tardiness/` | Minimize $\sum w_j T_j$ instead of $C_{\max}$ |
| Flexible Tardiness | `variants/flexible_tardiness/` | Flexible machine assignment + weighted tardiness |

### 3.1 No-Wait Job Shop

No idle time between consecutive operations of the same job. Much more constrained than standard JSP. Often modeled by modifying the disjunctive graph with fixed time lags.

### 3.2 Weighted Tardiness JSP

Replace the makespan objective with $\min \sum w_j T_j$. This is a regular measure, meaning it cannot decrease if any operation starts later. Dispatching rules like ATC (from single machine) can be adapted.

---

## 4. Benchmark Instances

### Standard Instances

| Instance | Size ($n \times m$) | Optimal | Status |
|----------|-------------------|---------|--------|
| ft06 | 6 × 6 | 55 | Solved |
| ft10 | 10 × 10 | 930 | Solved (1989) |
| ft20 | 20 × 5 | 1165 | Solved |
| la01–la40 | 10×5 to 15×15 | Known | Lawrence (1984) |
| ta01–ta80 | 15×15 to 100×20 | Many open | Taillard (1993) |
| orb01–orb10 | 10 × 10 | Known | Applegate & Cook (1991) |

**URL:** http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir/

### Small Illustrative Instance (ft06)

6 jobs, 6 machines. Each job has 6 operations. Optimal makespan = 55. This instance is small enough to solve by hand or with any method.

### Best Known Solutions (BKS)

| Instance | Size ($n \times m$) | BKS | Optimal? | Source |
|----------|-------------------|-----|----------|--------|
| ft06 | 6 x 6 | 55 | Yes | Fisher & Thompson (1963) |
| ft10 | 10 x 10 | 930 | Yes | Carlier & Pinson (1989) |
| ft20 | 20 x 5 | 1165 | Yes | Fisher & Thompson (1963) |
| la01 | 10 x 5 | 666 | Yes | Lawrence (1984) |
| la02 | 10 x 5 | 655 | Yes | Lawrence (1984) |
| la03 | 10 x 5 | 597 | Yes | Lawrence (1984) |
| la04 | 10 x 5 | 590 | Yes | Lawrence (1984) |
| la05 | 10 x 5 | 593 | Yes | Lawrence (1984) |
| la06 | 15 x 5 | 926 | Yes | Lawrence (1984) |
| la07 | 15 x 5 | 890 | Yes | Lawrence (1984) |
| la08 | 15 x 5 | 863 | Yes | Lawrence (1984) |
| la09 | 15 x 5 | 951 | Yes | Lawrence (1984) |
| la10 | 15 x 5 | 958 | Yes | Lawrence (1984) |
| la16 | 10 x 10 | 945 | Yes | Lawrence (1984) |
| la17 | 10 x 10 | 784 | Yes | Lawrence (1984) |
| la18 | 10 x 10 | 848 | Yes | Lawrence (1984) |
| la19 | 10 x 10 | 842 | Yes | Lawrence (1984) |
| la20 | 10 x 10 | 902 | Yes | Lawrence (1984) |
| abz5 | 10 x 10 | 1234 | Yes | Adams, Balas & Zawack (1988) |
| abz6 | 10 x 10 | 943 | Yes | Adams, Balas & Zawack (1988) |
| abz7 | 20 x 15 | 656 | [TODO: verify BKS] | Adams, Balas & Zawack (1988) |
| abz8 | 20 x 15 | 665 | [TODO: verify BKS] | Adams, Balas & Zawack (1988) |
| abz9 | 20 x 15 | 679 | [TODO: verify BKS] | Adams, Balas & Zawack (1988) |
| orb01 | 10 x 10 | 1059 | Yes | Applegate & Cook (1991) |
| orb02 | 10 x 10 | 888 | Yes | Applegate & Cook (1991) |
| orb03 | 10 x 10 | 1005 | Yes | Applegate & Cook (1991) |
| orb04 | 10 x 10 | 1005 | Yes | Applegate & Cook (1991) |
| orb05 | 10 x 10 | 887 | Yes | Applegate & Cook (1991) |
| ta01 | 15 x 15 | 1231 | Yes | Taillard (1993) |
| ta11 | 20 x 15 | [TODO: verify BKS] | [TODO: verify BKS] | Taillard (1993) |
| ta21 | 20 x 20 | [TODO: verify BKS] | [TODO: verify BKS] | Taillard (1993) |
| ta31 | 30 x 15 | [TODO: verify BKS] | [TODO: verify BKS] | Taillard (1993) |
| ta41 | 30 x 20 | [TODO: verify BKS] | No | Taillard (1993) |
| ta51 | 50 x 15 | [TODO: verify BKS] | No | Taillard (1993) |
| ta61 | 50 x 20 | [TODO: verify BKS] | No | Taillard (1993) |
| ta71 | 100 x 20 | [TODO: verify BKS] | No | Taillard (1993) |

**Notes:**
- BKS values for ft, la (small), abz5-6, and orb instances are proven optimal via constraint propagation or branch-and-bound.
- The abz7-9 (20x15) instances have tight bounds but optimality proofs remain uncertain for some; verify against current literature.
- Large Taillard instances (ta41+) remain open; best known upper bounds are continuously improved by new metaheuristics.
- BKS values sourced from OR-Library and http://mistic.heig-vd.ch/taillard/. Verify against the latest tables before benchmarking.

---

## 5. Solution Methods

### 5.1 Exact Methods

- **Branch & Bound:** Carlier & Pinson (1989) — constraint propagation on the disjunctive graph. Solved ft10 after 26 years. Practical for $n \times m \leq 15 \times 15$.
- **Constraint Programming:** CP-SAT (OR-Tools) models JSP with interval variables and no-overlap constraints. Very effective for medium instances.
- **MIP:** Big-M disjunctive constraints or indicator variables. Generally weaker than CP.

*Note: This repository does not currently include exact method implementations for JSP. The focus is on heuristic and metaheuristic methods.*

### 5.2 Constructive Heuristics

#### Dispatching Rules (Giffler & Thompson, 1960)

**Idea:** Build a schedule operation by operation. At each step, identify the set of schedulable operations and select one using a priority rule:

| Rule | Priority | Description |
|------|----------|-------------|
| SPT | Ascending $p_{jk}$ | Shortest processing time first |
| LPT | Descending $p_{jk}$ | Longest processing time first |
| MWR | Descending remaining work | Most Work Remaining |
| LWR | Ascending remaining work | Least Work Remaining |
| FIFO | Ascending release time | First In First Out |

The Giffler-Thompson (G&T) procedure generates **active schedules** (no operation can be started earlier without delaying another). Active schedules contain an optimal schedule.

**Pseudocode: Giffler-Thompson Active Schedule Generation**

```
GIFFLER-THOMPSON(instance, priority_rule)
 1  Input: n jobs, m machines, processing times p[j][k], machine assignments mu[j][k]
 2  Output: start_times S[j][k] for all operations
 3
 4  next_op[j] <- 0            for j = 1, ..., n       // next unscheduled op per job
 5  machine_avail[i] <- 0      for i = 1, ..., m       // machine ready time
 6  job_avail[j] <- 0          for j = 1, ..., n       // job ready time
 7  scheduled <- 0
 8
 9  while scheduled < total_operations do
10      // Step 1: Collect schedulable operations
11      C <- {}
12      for j = 1, ..., n do
13          k <- next_op[j]
14          if k < num_ops(j) then
15              i <- mu[j][k]
16              r <- max(machine_avail[i], job_avail[j])    // earliest start
17              e <- r + p[j][k]                             // earliest completion
18              C <- C U {(j, k, i, r, e)}
19
20      // Step 2: Find operation with minimum earliest completion
21      e* <- min{e : (j, k, i, r, e) in C}
22      i* <- machine of the operation achieving e*
23
24      // Step 3: Build conflict set (ops on machine i* that could start before e*)
25      Q <- {(j, k, i, r, e) in C : i = i* and r < e*}
26
27      // Step 4: Select one operation from Q using priority_rule
28      (j*, k*, ...) <- priority_rule(Q)
29
30      // Step 5: Schedule the selected operation
31      S[j*][k*] <- max(machine_avail[i*], job_avail[j*])
32      machine_avail[i*] <- S[j*][k*] + p[j*][k*]
33      job_avail[j*] <- S[j*][k*] + p[j*][k*]
34      next_op[j*] <- k* + 1
35      scheduled <- scheduled + 1
36
37  return S
```

#### Shifting Bottleneck (Adams, Balas & Zawack, 1988)

**Idea:** Iteratively identify the "bottleneck" machine — the one most constraining the makespan. Solve a single-machine sub-problem on each unscheduled machine, select the one with maximum optimal makespan, fix its sequence, then re-optimize previously sequenced machines.

**Quality:** Produces very good initial solutions, often within 5-10% of optimal. Combines single-machine exact algorithms with iterative refinement.

**Pseudocode: Shifting Bottleneck Procedure**

```
SHIFTING-BOTTLENECK(instance)
 1  Input: n jobs, m machines, processing times p[j][k], routings mu[j][k]
 2  Output: machine_order[i] for each machine i (complete schedule)
 3
 4  M0 <- {}                                // set of sequenced machines
 5  machine_order[i] <- empty   for i = 1, ..., m
 6
 7  while |M0| < m do
 8      // Step 1: Compute heads (release times) and tails for all operations
 9      //         using current partial schedule (job precedence + sequenced machines)
10      (head, tail) <- COMPUTE-HEADS-TAILS(instance, machine_order)
11
12      // Step 2: For each unsequenced machine, solve 1|r_j|Lmax sub-problem
13      best_machine <- nil
14      best_Lmax <- -infinity
15      best_seq <- nil
16
17      for each machine i not in M0 do
18          ops_i <- {(j, k) : mu[j][k] = i}     // operations assigned to machine i
19
20          // Build single-machine sub-problem:
21          //   release time r_op = head[(j,k)]
22          //   processing time p_op = p[j][k]
23          //   due date d_op = C_LB - tail[(j,k)]
24          //     where C_LB = max over ops_i of { head[op] + p[op] + tail[op] }
25          C_LB <- max{ head[(j,k)] + p[j][k] + tail[(j,k)] : (j,k) in ops_i }
26
27          for each (j, k) in ops_i do
28              r_op <- head[(j, k)]
29              d_op <- C_LB - tail[(j, k)]
30
31          // Solve 1|r_j|Lmax using modified EDD (preemptive relaxation)
32          (sigma_i, Lmax_i) <- SOLVE-SINGLE-MACHINE(ops_i, r, p, d)
33
34          if Lmax_i > best_Lmax then
35              best_Lmax <- Lmax_i
36              best_machine <- i
37              best_seq <- sigma_i
38
39      // Step 3: Fix the bottleneck machine sequence
40      machine_order[best_machine] <- best_seq
41      M0 <- M0 U {best_machine}
42
43      // Step 4 (optional): Re-optimize previously sequenced machines
44      //   For each i in M0 \ {best_machine}, re-solve 1|r_j|Lmax
45      //   with updated heads/tails and replace if improved.
46
47  // Step 5: Build start times by forward pass through the complete graph
48  start_times <- BUILD-START-TIMES(instance, machine_order)
49  return start_times, machine_order
```

**Complexity:** $O(m^2 \cdot n^2)$ without re-optimization; $O(m^3 \cdot n^2)$ with re-optimization. The bottleneck identification step dominates.

### 5.3 Metaheuristics

This repository implements **6 metaheuristic/LS methods** for JSP:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Critical-path swap/insertion neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | N1 neighborhood with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | N1/N5 neighborhood with aspiration criterion |
| 4 | Iterated Greedy (IG) | Trajectory | Remove operations + reconstruct via dispatching |
| 5 | Genetic Algorithm (GA) | Population | Permutation-with-repetition encoding |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Systematic N1 → swap → block-move |

### Critical-Path Neighborhoods

The most effective JSP neighborhoods operate on **critical blocks** of the disjunctive graph:

- **N1 (Van Laarhoven et al., 1992):** Swap any two adjacent operations in a critical block. Size: $O(m)$ moves per critical path.
- **N5 (Nowicki & Smutnicki, 1996):** Swap only the first/last operation of each critical block with its neighbor. Smaller than N1, but empirically as effective.
- **N7 (Balas & Vazacopoulos, 1998):** Insert an operation to a different position within its critical block. More disruptive than swap, can escape deeper local optima. Size: $O(\sum_b |B_b|^2)$ where $B_b$ is the $b$-th critical block.

**Detailed neighborhood definitions.** Let $B = (u_1, u_2, \ldots, u_l)$ be a critical block on some machine (consecutive critical-path operations on the same machine):

| Neighborhood | Move Set | Size per Block | Description |
|-------------|----------|----------------|-------------|
| N1 | $\{(u_i, u_{i+1}) : i = 1,\ldots,l{-}1\}$ | $l - 1$ | Swap adjacent pair in block |
| N5 | $\{(u_1, u_2), (u_{l-1}, u_l)\}$ | $\leq 2$ | Swap only at block endpoints |
| N7 | $\{(u_i, \text{pos}) : i \neq \text{pos}\}$ | $O(l^2)$ | Insert $u_i$ at any position in block |

**Key property (Nowicki & Smutnicki, 1996):** For a critical block $B$ with $|B| \geq 3$, only moves involving the first or last operation of the block can reduce the makespan. This insight reduces N1 to N5 without loss of solution quality in practice.

#### Nowicki-Smutnicki Tabu Search (i-TSAB)

The Nowicki & Smutnicki (1996) approach is one of the most effective JSP methods:

1. **Initialization:** Best dispatching rule solution (try SPT, MWR, LPT, LWR).
2. **Neighborhood:** N5 (endpoint swaps on critical blocks). Evaluate all moves; select the best non-tabu move, or a tabu move if it improves the global best (aspiration criterion).
3. **Tabu list:** Store the reversed pair $(u, v)$ for each swap. Tenure $= n + m/2$ (adaptive).
4. **Termination:** Fixed iteration count or no improvement for $\text{max\_iter}/5$ iterations.
5. **Key insight:** The compact N5 neighborhood avoids evaluating many inferior moves, making each iteration very fast ($O(n \cdot m)$ for critical path recomputation).

#### Metaheuristic Parameter Tables

**Simulated Annealing (Van Laarhoven et al., 1992)**

| Parameter | Symbol | Default | Tuning Range | Notes |
|-----------|--------|---------|-------------|-------|
| Initial temperature | $T_0$ | $0.05 \cdot C_{\max}^{\text{init}}$ | $[0.01, 0.10] \cdot C_{\max}^{\text{init}}$ | Auto-calibrated from initial makespan |
| Cooling rate | $\alpha$ | 0.995 | $[0.990, 0.999]$ | Geometric: $T_{k+1} = \alpha \cdot T_k$ |
| Neighborhood | N1 | — | N1, N5 | Critical-path adjacent swap |
| Max iterations | — | 10,000 | $[5000, 50000]$ | Scale with $n \times m$ |
| Acceptance | Boltzmann | — | — | $P(\text{accept}) = \exp(-\Delta / T)$ |

**Tabu Search (Nowicki & Smutnicki, 1996)**

| Parameter | Symbol | Default | Tuning Range | Notes |
|-----------|--------|---------|-------------|-------|
| Tabu tenure | $\theta$ | $n + m/2$ | $[n, n + m]$ | Length of tabu list |
| Aspiration | — | Best-so-far | — | Accept tabu move if it improves global best |
| Neighborhood | N1/N5 | — | N1, N5 | N5 preferred for speed |
| Max iterations | — | 5,000 | $[1000, 50000]$ | Scale with problem size |
| Restart threshold | — | $\text{max\_iter}/5$ | — | Stagnation-based restart |

**Genetic Algorithm (Bierwirth, 1995)**

| Parameter | Symbol | Default | Tuning Range | Notes |
|-----------|--------|---------|-------------|-------|
| Population size | $N_{\text{pop}}$ | 30 | $[20, 100]$ | Seeded from dispatching rules |
| Generations | $G$ | 200 | $[100, 1000]$ | Scale with $n \times m$ |
| Crossover rate | $p_c$ | 0.8 | $[0.6, 0.95]$ | JOX (Job-Order Crossover) |
| Mutation rate | $p_m$ | 0.2 | $[0.05, 0.3]$ | Swap two random genes |
| Selection | — | Binary tournament | — | Fitness-proportional selection |
| Replacement | — | Replace worst | — | Steady-state with elitism |
| Encoding | — | Operation-based | — | Bierwirth (1995) permutation with repetition |

### 5.4 Lagrangian Relaxation for JSP

**Idea:** Relax the machine capacity constraints (disjunctive constraints) using Lagrangian multipliers. The relaxed problem decomposes into $n$ independent single-job longest-path sub-problems.

**Formulation.** Introduce multipliers $\lambda_{it} \geq 0$ for the machine capacity constraints (8). The Lagrangian relaxation is:

$$L(\lambda) = \min_{x} \; C_{\max} + \sum_{i=1}^{m} \sum_{t=0}^{T} \lambda_{it} \left( \sum_{\substack{(j,k):\\ \mu_{jk}=i}} \sum_{\tau=\max(0,t-p_{jk}+1)}^{t} x_{jk\tau} - 1 \right)$$

subject to the precedence and assignment constraints only.

**Decomposition.** For fixed $\lambda$, each job's operations can be scheduled independently by solving a shortest-path problem on a time-expanded graph (the precedence constraints define a simple chain per job). Each sub-problem has complexity $O(n_j \cdot T)$.

**Lower bound.** The Lagrangian dual $\max_{\lambda \geq 0} L(\lambda)$ provides a lower bound on $C_{\max}^*$. This bound is at least as strong as the LP relaxation of the time-indexed formulation. The dual is solved via subgradient optimization:

$$\lambda_{it}^{(k+1)} = \max\left(0, \; \lambda_{it}^{(k)} + \theta_k \left( \sum_{\substack{(j,k'):\\ \mu_{jk'}=i}} \sum_{\tau=\max(0,t-p_{jk'}+1)}^{t} x_{jk'\tau}^{(k)} - 1 \right)\right)$$

where $\theta_k$ is a step size (e.g., $\theta_k = \alpha_k (UB - L(\lambda^{(k)})) / \|\text{subgradient}\|^2$).

**Practical use:** Lagrangian relaxation provides strong dual bounds for branch-and-bound and enables Lagrangian heuristics that round fractional solutions to feasible schedules.

---

## 6. Implementation Guide

### Modeling Tips

- **Disjunctive graph:** Store as adjacency lists with operation nodes. Compute the critical path via topological sort + longest-path DP in $O(V + E)$.
- **Makespan after a swap:** After swapping two adjacent operations on a machine, only the affected subgraph needs re-evaluation. Full critical path recomputation is $O(V + E)$, but incremental evaluation can be $O(m + n)$.
- **Feasibility check:** A swap is feasible (no cycle) if and only if the resulting graph remains a DAG. For adjacent critical operations, this is guaranteed by the N1 structure.

### Common Pitfalls

- **Conjunctive vs. disjunctive arcs:** Conjunctive arcs have fixed orientation (job precedence). Only disjunctive arcs (machine sequencing) are decision variables.
- **Active vs. semi-active schedules:** A semi-active schedule has no unnecessary idle time; an active schedule additionally ensures no operation can start earlier. Active schedules are preferable.

---

## 7. Computational Results Summary

| Method | Gap on ft06 (6×6) | Gap on ft10 (10×10) | Gap on ta (15-100) |
|--------|-------------------|---------------------|---------------------|
| Dispatching (MWR) | 5-15% | 10-20% | 15-30% |
| Shifting Bottleneck | 0-5% | 3-8% | 5-15% |
| SA (N1) | 0% | 0-2% | 2-5% |
| TS (N5) | 0% | 0% | 1-3% |
| GA | 0% | 0-1% | 2-5% |

**State-of-the-art:** Tabu Search with N5 neighborhood (Nowicki & Smutnicki) remains competitive. Recent hybrid methods combining TS with path relinking achieve <1% gap on most Taillard instances.

---

## 8. Implementations in This Repository

```
job_shop/
├── instance.py                        # JobShopInstance, disjunctive graph, ft06/ft10
│
├── heuristics/
│   ├── dispatching_rules.py           # SPT, LPT, MWR, LWR, FIFO (G&T active schedules)
│   └── shifting_bottleneck.py         # Adams-Balas-Zawack (1988)
│
├── metaheuristics/
│   ├── local_search.py                # Critical-path swap/insertion
│   ├── simulated_annealing.py         # SA with N1 neighborhood
│   ├── tabu_search.py                 # TS with N1/N5 + aspiration
│   ├── iterated_greedy.py             # IG: remove + dispatching reconstruct
│   ├── genetic_algorithm.py           # GA: permutation-with-repetition
│   └── vns.py                         # VNS: N1 → swap → block-move
│
├── variants/
│   ├── no_wait/                       # No-wait JSP
│   ├── weighted_tardiness/            # Jm || ΣwjTj
│   └── flexible_tardiness/            # FJSP + ΣwjTj
│
└── tests/                             # 6 test files
    ├── conftest.py                    # Shared fixtures
    ├── test_job_shop.py               # Core algorithms
    ├── test_job_shop_ga.py            # Genetic Algorithm
    ├── test_job_shop_ig.py            # Iterated Greedy
    ├── test_job_shop_vns.py           # VNS
    └── test_jsp_ls.py                 # Local Search
```

**Total:** 2 heuristic methods, 6 metaheuristics/LS, 3 variants, 6 test files.

---

## 9. Key References

### Seminal Papers

- Roy, B. & Sussmann, B. (1964). Les problemes d'ordonnancement avec contraintes disjonctives. *Note DS no. 9 bis, SEMA*, Paris. --- *Introduced the disjunctive graph model, the foundational data structure for JSP.*
- Fisher, H. & Thompson, G.L. (1963). Probabilistic learning combinations of local job-shop scheduling rules. *Industrial Scheduling*, 225-251. --- *Created the ft06 and ft10 benchmark instances; ft10 remained unsolved for 26 years.*
- Giffler, B. & Thompson, G.L. (1960). Algorithms for solving production-scheduling problems. *Operations Research*, 8(4), 487-503. [DOI:10.1287/opre.8.4.487](https://doi.org/10.1287/opre.8.4.487) --- *Introduced the active schedule generation procedure using dispatching rules.*
- Adams, J., Balas, E. & Zawack, D. (1988). The shifting bottleneck procedure for job shop scheduling. *Management Science*, 34(3), 391-401. [DOI:10.1287/mnsc.34.3.391](https://doi.org/10.1287/mnsc.34.3.391) --- *Proposed the shifting bottleneck heuristic; iterative single-machine decomposition.*
- Carlier, J. & Pinson, E. (1989). An algorithm for solving the job-shop problem. *Management Science*, 35(2), 164-176. [DOI:10.1287/mnsc.35.2.164](https://doi.org/10.1287/mnsc.35.2.164) --- *Proved ft10 optimal at 930 using constraint propagation-based B&B.*

### Key Metaheuristic References

- Van Laarhoven, P.J.M., Aarts, E.H.L. & Lenstra, J.K. (1992). Job shop scheduling by simulated annealing. *Operations Research*, 40(1), 113-125. [DOI:10.1287/opre.40.1.113](https://doi.org/10.1287/opre.40.1.113) --- *Introduced the N1 neighborhood (adjacent swap on critical blocks) for SA.*
- Nowicki, E. & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. *Management Science*, 42(6), 797-813. [DOI:10.1287/mnsc.42.6.797](https://doi.org/10.1287/mnsc.42.6.797) --- *Proposed the N5 neighborhood (endpoint-only swaps); state-of-the-art for two decades.*
- Bierwirth, C. (1995). A generalized permutation approach to job shop scheduling with genetic algorithms. *OR Spektrum*, 17(2), 87-92. [DOI:10.1007/BF01719250](https://doi.org/10.1007/BF01719250) --- *Introduced operation-based (permutation with repetition) encoding for GA.*
- Balas, E. & Vazacopoulos, A. (1998). Guided local search with shifting bottleneck for job shop scheduling. *Management Science*, 44(2), 262-275. [DOI:10.1287/mnsc.44.2.262](https://doi.org/10.1287/mnsc.44.2.262) --- *Combined shifting bottleneck with local search; introduced the N7 (insertion) neighborhood.*
- Nowicki, E. & Smutnicki, C. (2005). An advanced tabu search algorithm for the job shop problem. *Journal of Scheduling*, 8(2), 145-159. [DOI:10.1007/s10951-005-6364-5](https://doi.org/10.1007/s10951-005-6364-5) --- *i-TSAB: improved TS with big-valley structure exploitation and path relinking.*

### Textbooks

- Pinedo, M.L. (2016). *Scheduling: Theory, Algorithms, and Systems*. 5th Edition, Springer. --- *Standard textbook covering JSP theory, algorithms, and complexity. Chapters 7 (job shop) and 15 (constraint-based approaches).*
- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman. --- *Proved strong NP-hardness of JSP; foundational complexity reference.*
- Blazewicz, J., Ecker, K.H., Pesch, E., Schmidt, G. & Weglarz, J. (2007). *Handbook on Scheduling*. Springer. --- *Comprehensive scheduling reference with detailed JSP coverage.*

### Benchmark and Complexity References

- Lawrence, S. (1984). Resource constrained project scheduling: An experimental investigation of heuristic scheduling techniques (Supplement). *Graduate School of Industrial Administration, Carnegie-Mellon University*. --- *Created the la01-la40 benchmark instances.*
- Taillard, E. (1993). Benchmarks for basic scheduling problems. *European Journal of Operational Research*, 64(2), 278-285. [DOI:10.1016/0377-2217(93)90182-M](https://doi.org/10.1016/0377-2217(93)90182-M) --- *Created the ta01-ta80 benchmark instances.*

### Surveys

- Jain, A.S. & Meeran, S. (1999). Deterministic job-shop scheduling: Past, present and future. *European Journal of Operational Research*, 113(2), 390-434. [DOI:10.1016/S0377-2217(98)00113-1](https://doi.org/10.1016/S0377-2217(98)00113-1) --- *Comprehensive survey covering exact, heuristic, and metaheuristic approaches up to 1999.*
- Zhang, J., Ding, G., Zou, Y., Qin, S. & Fu, J. (2019). Review of job shop scheduling research and its new perspectives under Industry 4.0. *Journal of Intelligent Manufacturing*, 30(4), 1809-1830. [DOI:10.1007/s10845-017-1350-2](https://doi.org/10.1007/s10845-017-1350-2) --- *Modern survey connecting classical JSP to Industry 4.0 themes.*

### Key Insight

> The **disjunctive graph** is the central data structure for JSP. Understanding it unlocks everything: the critical path determines the makespan, critical blocks define effective neighborhoods, and the graph structure enables efficient move evaluation. Nearly all state-of-the-art methods operate directly on this graph.

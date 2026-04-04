# Flexible Job Shop Scheduling (FJSP)

## 1. Problem Definition

- **Input:**
  - $n$ jobs, each with an ordered sequence of operations; $m$ machines
  - Each operation $O_{jk}$ has a set of eligible machines $M_{jk} \subseteq M$
  - Processing time $p_{jki}$ for operation $O_{jk}$ on machine $i \in M_{jk}$
- **Decision:** (1) **Routing** — assign each operation to one eligible machine. (2) **Sequencing** — determine the processing order on each machine.
- **Objective:** Minimize makespan $C_{\max}$
- **Constraints:** Operations within a job are sequential (precedence). Each machine processes one operation at a time (disjunctive). No preemption.
- **Classification:** NP-hard (generalizes JSP)

### Variants

| Variant | Description |
|---------|-------------|
| **Total FJSP (T-FJSP)** | Every operation eligible on all machines |
| **Partial FJSP (P-FJSP)** | Each operation eligible on a subset |
| **Multi-objective FJSP** | Optimize makespan + workload balance + total workload |

**Complexity:** NP-hard even for $FJ_2 \mid\mid C_{\max}$. The routing sub-problem alone is equivalent to unrelated parallel machine scheduling.

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $O_{jk}$ | $k$-th operation of job $j$ | — |
| $M_{jk}$ | Eligible machines for $O_{jk}$ | $\subseteq M$ |
| $p_{jki}$ | Processing time of $O_{jk}$ on machine $i$ | $\mathbb{R}_{>0}$ |
| $x_{jki}$ | 1 if $O_{jk}$ assigned to machine $i$ | $\{0, 1\}$ |
| $s_{jk}$ | Start time of $O_{jk}$ | $\mathbb{R}_{\geq 0}$ |

### MILP Formulation

$$\min \quad C_{\max} \tag{1}$$

$$\sum_{i \in M_{jk}} x_{jki} = 1 \quad \forall j, k \quad \text{(one machine per operation)} \tag{2}$$

$$s_{j,k+1} \geq s_{jk} + \sum_{i \in M_{jk}} p_{jki} \cdot x_{jki} \quad \forall j, k \quad \text{(precedence)} \tag{3}$$

$$s_{jk} + p_{jki} \leq s_{j'k'} + B(1 - y_{jk,j'k'}) \quad \text{(disjunctive, if both on machine } i\text{)} \tag{4}$$

$$C_{\max} \geq s_{jk} + \sum_{i \in M_{jk}} p_{jki} \cdot x_{jki} \quad \forall j, k \tag{5}$$

The coupling between routing (2) and sequencing (4) makes FJSP harder than JSP.

---

## 3. Solution Methods

### 3.1 Constructive Heuristics

#### Dispatching Rules (adapted from JSP)

**Priority rules** (SPT, LPT, MWR, LWR) select which operation to schedule. **Machine selection rules** (ECT — Earliest Completion Time, SPT — shortest on that machine) assign the operation to a machine.

#### Hierarchical Decomposition (Brandimarte, 1993)

**Step 1 — Routing:** Assign each operation to a machine using local rules or LP relaxation.
**Step 2 — Sequencing:** Solve the resulting JSP on the assigned machines.

Quality is limited by the routing-sequencing interaction: a good routing may produce bad sequencing and vice versa.

### 3.2 Metaheuristics

This repository implements **6 metaheuristic/LS methods** for FJSP:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search | Improvement | Assignment + sequencing neighborhoods |
| 2 | Simulated Annealing (SA) | Trajectory | Combined routing/sequencing moves |
| 3 | Tabu Search (TS) | Trajectory | Dual neighborhoods (assignment + critical-path) |
| 4 | Iterated Greedy (IG) | Trajectory | Remove operations + reinsert via dispatching |
| 5 | Genetic Algorithm (GA) | Population | Pezzella-style integrated encoding |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Assignment → sequencing → combined |

### Key Neighborhoods

- **Assignment move:** Reassign an operation to a different eligible machine
- **Sequencing move:** Swap/insert operations on the same machine (critical-path based)
- **Combined:** Apply both simultaneously

The interaction between routing and sequencing is why integrated approaches (GA, TS with dual neighborhoods) outperform decomposition.

---

## 4. Benchmark Instances

| Set | Author | Jobs × Machines | Instances |
|-----|--------|----------------|-----------|
| BRdata | Brandimarte (1993) | 10×4 to 20×15 | 10 (Mk01–Mk10) |
| BCdata | Barnes & Chambers (1996) | 10×11 to 15×11 | 21 |
| HUdata | Hurink et al. (1994) | Various | ~130 (edata/rdata/vdata) |

**URL:** http://people.brunel.ac.uk/~mastjjb/jeb/orlib/fjspinfo.html

---

## 5. Implementations in This Repository

```
flexible_job_shop/
├── instance.py                    # FlexibleJobShopInstance (total/partial FJSP)
├── heuristics/
│   ├── dispatching_rules.py       # SPT/LPT/MWR/LWR + ECT/SPT machine selection
│   └── hierarchical.py            # Route-then-sequence decomposition
├── metaheuristics/
│   ├── local_search.py            # Assignment + sequencing neighborhoods
│   ├── simulated_annealing.py     # SA with combined moves
│   ├── tabu_search.py             # TS with dual neighborhoods
│   ├── iterated_greedy.py         # IG: remove + dispatching reconstruct
│   ├── genetic_algorithm.py       # GA: Pezzella-style encoding
│   └── vns.py                     # VNS: assignment → sequencing → combined
└── tests/                         # 7 test files
    ├── conftest.py                # Shared fixtures
    ├── test_fjsp.py               # Core algorithms
    ├── test_fjsp_sa.py            # SA
    ├── test_fjsp_ts.py            # TS
    ├── test_fjsp_ig.py            # IG
    ├── test_fjsp_ls.py            # LS
    └── test_fjsp_vns.py           # VNS
```

*Note: No exact method implementations currently exist. The `exact/` directory is empty.*

**Total:** 2 heuristic methods, 6 metaheuristics/LS, 7 test files.

---

## 6. Key References

- Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search. *Annals of Operations Research*, 41(3), 157-183.
- Pezzella, F., Morganti, G. & Ciaschetti, G. (2008). A genetic algorithm for the flexible job-shop scheduling problem. *Computers & Operations Research*, 35(10), 3202-3212.
- Mastrolilli, M. & Gambardella, L.M. (2000). Effective neighbourhood functions for the flexible job shop problem. *Journal of Scheduling*, 3(1), 3-20.
- Brucker, P. & Schlie, R. (1990). Job-shop scheduling with multi-purpose machines. *Computing*, 45(4), 369-375.

### Key Insight

> The FJSP's difficulty comes from the **interaction between routing and sequencing**: assigning an operation to a faster machine may create a bottleneck for other operations. Integrated methods that simultaneously address both decisions outperform decomposition approaches.

# Resource-Constrained Project Scheduling (RCPSP)

## 1. Problem Definition

- **Input:**
  - $n$ activities (plus dummy source 0 and sink $n{+}1$) with durations $p_j$
  - $K$ renewable resource types with capacities $R_k$
  - Resource requirements $r_{jk}$ (activity $j$ needs $r_{jk}$ units of resource $k$)
  - Precedence relations $E$: set of pairs $(i, j)$ meaning $i$ must finish before $j$ starts
- **Decision:** Determine start times $s_j$ for all activities
- **Objective:** Minimize project makespan $C_{\max} = s_{n+1}$
- **Constraints:** (1) Precedence: $s_j \geq s_i + p_i$ for all $(i,j) \in E$. (2) Resource: at any time $t$, $\sum_{j: s_j \leq t < s_j + p_j} r_{jk} \leq R_k$ for all resources $k$. (3) No preemption.
- **Classification:** Strongly NP-hard, even with 2 resource types
- **Scheduling notation:** $PS \mid prec \mid C_{\max}$

### Complexity

- Strongly NP-hard even with $K = 2$ resource types
- With no resource constraints: reduces to longest path (polynomial)
- With unit processing times: still NP-hard
- The decision version (is makespan $\leq T$?) is NP-complete

### Variants

| Variant | Description |
|---------|-------------|
| Multi-Mode (MRCPSP) | Multiple execution modes per activity (time/resource trade-off) |
| Generalized Precedence (RCPSP/GPR) | Min/max time lags between activities |
| Multi-Skill (MS-RCPSP) | Workers have different skills |
| Stochastic (SRCPSP) | Uncertain activity durations |
| Preemptive (PRCPSP) | Activities can be interrupted |

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of non-dummy activities | $\mathbb{Z}^+$ |
| $K$ | Number of resource types | $\mathbb{Z}^+$ |
| $p_j$ | Duration of activity $j$ | $\mathbb{R}_{>0}$ |
| $r_{jk}$ | Resource $k$ requirement of activity $j$ | $\mathbb{Z}_{\geq 0}$ |
| $R_k$ | Capacity of resource $k$ | $\mathbb{Z}_{>0}$ |
| $s_j$ | Start time of activity $j$ | $\mathbb{R}_{\geq 0}$ |
| $E$ | Precedence pairs $(i,j)$ | — |

### Time-Indexed Formulation

Binary variables $x_{jt} = 1$ if activity $j$ starts at time $t$:

$$\min \sum_t t \cdot x_{(n+1),t} \tag{1}$$

$$\sum_{t \in \mathcal{T}_j} x_{jt} = 1 \quad \forall j \quad \text{(each activity starts once)} \tag{2}$$

$$\sum_t t \cdot x_{jt} + p_j \leq \sum_t t \cdot x_{j't} \quad \forall (j, j') \in E \quad \text{(precedence)} \tag{3}$$

$$\sum_{j=1}^{n} r_{jk} \sum_{\tau=\max(0,t-p_j+1)}^{t} x_{j\tau} \leq R_k \quad \forall k, t \quad \text{(resource)} \tag{4}$$

Good LP relaxation but grows linearly with the time horizon.

---

## 3. Solution Methods

### 3.1 Schedule Generation Schemes (SGS)

The **SGS** is the core building block — it decodes a priority list into a feasible schedule:

| Scheme | Property | Description |
|--------|----------|-------------|
| **Serial SGS** | Active schedules | Schedule one activity at a time, earliest feasible start |
| **Parallel SGS** | Non-delay schedules | At each time point, schedule all feasible activities |

**Serial SGS** is generally preferred: the set of active schedules contains an optimal schedule, while non-delay schedules may not.

```
ALGORITHM SerialSGS(priority_list, precedence, resources)
  scheduled ← {0}    (dummy start)
  FOR j in priority_list (precedence-feasible order):
    t ← earliest time when:
      (1) all predecessors of j are complete
      (2) resources available for j's entire duration
    Schedule j at time t
  RETURN schedule
```

### 3.2 Priority Rules

| Rule | Abbreviation | Description |
|------|-------------|-------------|
| Latest Finish Time | LFT | Schedule most urgent activity first |
| Earliest Start Time | EST | Schedule earliest available first |
| Most Total Successors | MTS | Schedule most constrained first |
| Greatest Rank Positional Weight | GRPW | Weighted path length to sink |

### 3.3 Metaheuristics

This repository implements **6 metaheuristic/LS methods** for RCPSP:

| # | Method | Category | Encoding |
|---|--------|----------|---------|
| 1 | Local Search | Improvement | Activity-list + Serial SGS |
| 2 | Simulated Annealing (SA) | Trajectory | Swap/insert in activity list |
| 3 | Tabu Search (TS) | Trajectory | Priority-list swap with tabu |
| 4 | Iterated Greedy (IG) | Trajectory | Remove activities + reinsert via priority rules |
| 5 | Genetic Algorithm (GA) | Population | Activity-list encoding, one-point crossover (Hartmann, 1998) |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Swap → insert → block-move in activity list |

**Activity-list encoding:** A precedence-feasible permutation of activities. The GA's crossover preserves precedence feasibility, and the Serial SGS decoder produces the schedule.

### 3.4 Lower Bounds

| Method | Quality |
|--------|---------|
| Critical Path (CPM) | Ignores resources — weakest |
| LP relaxation (time-indexed) | Strong but slow for large horizons |
| Destructive improvement | Binary search with feasibility check |
| Energy-based reasoning | From constraint propagation |

---

## 4. Benchmark Instances

### PSPLIB (Project Scheduling Problem Library)

| Set | Activities | Resources | Instances |
|-----|-----------|-----------|-----------|
| j30 | 30 | 4 | 480 |
| j60 | 60 | 4 | 480 |
| j90 | 90 | 4 | 480 |
| j120 | 120 | 4 | 600 |

**URL:** http://www.om-db.wi.tum.de/psplib/

Parameters: network complexity (NC), resource factor (RF), resource strength (RS).

---

## 5. Implementations in This Repository

```
rcpsp/
├── instance.py                    # RCPSPInstance, precedence DAG, resources
├── heuristics/
│   ├── serial_sgs.py              # Serial SGS (LFT/EST/MTS/GRPW rules)
│   └── parallel_sgs.py            # Parallel SGS (non-delay schedules)
├── metaheuristics/
│   ├── local_search.py            # Swap/insert in activity list
│   ├── simulated_annealing.py     # SA with activity-list moves
│   ├── tabu_search.py             # TS with priority-list tabu
│   ├── iterated_greedy.py         # IG: remove + priority-rule reinsert
│   ├── genetic_algorithm.py       # GA: activity-list encoding (Hartmann 1998)
│   └── vns.py                     # VNS: swap → insert → block-move
├── variants/
│   └── multi_mode/                # MRCPSP (multiple execution modes)
└── tests/                         # 7 test files
    ├── conftest.py                # Shared fixtures
    ├── test_rcpsp.py              # Core algorithms
    ├── test_rcpsp_sa.py           # SA
    ├── test_rcpsp_ts.py           # TS
    ├── test_rcpsp_ig.py           # IG
    ├── test_rcpsp_ls.py           # LS
    └── test_rcpsp_vns.py          # VNS
```

*Note: No exact method implementations currently exist. The `exact/` directory is empty.*

**Total:** 2 SGS methods + 4 priority rules, 6 metaheuristics/LS, 1 variant, 7 test files.

---

## 6. Key References

- Kolisch, R. & Sprecher, A. (1997). PSPLIB — a project scheduling problem library. *European Journal of Operational Research*, 96(1), 205-216.
- Hartmann, S. (1998). A competitive genetic algorithm for resource-constrained project scheduling. *Naval Research Logistics*, 45(7), 733-750.
- Kolisch, R. & Hartmann, S. (2006). Experimental investigation of heuristics for resource-constrained project scheduling: An update. *European Journal of Operational Research*, 174(1), 23-37.
- Demeulemeester, E. & Herroelen, W. (2002). *Project Scheduling: A Research Handbook*. Springer.
- Blazewicz, J., Lenstra, J.K. & Rinnooy Kan, A.H.G. (1983). Scheduling subject to resource constraints. *Discrete Applied Mathematics*, 5(1), 11-24.

### Key Insight

> The **Schedule Generation Scheme** is RCPSP's secret weapon. Unlike JSP where solutions are machine sequences, RCPSP solutions must respect both precedence and resource constraints simultaneously. The SGS decodes a priority list into a feasible schedule — making it possible to apply any permutation-based metaheuristic to RCPSP by searching over activity orderings and feeding them through SGS.

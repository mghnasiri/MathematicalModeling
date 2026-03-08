# Problem Taxonomy — Full Classification

## Scheduling Problems (Phase 1)

### 1. Single Machine Scheduling
**Path**: `problems/scheduling/single_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total completion | $1 \mid\mid \sum C_j$ | P | SPT |
| Weighted completion | $1 \mid\mid \sum w_j C_j$ | P | WSPT |
| Maximum lateness | $1 \mid\mid L_{\max}$ | P | EDD |
| Number of tardy jobs | $1 \mid\mid \sum U_j$ | P | Moore's |
| Total tardiness | $1 \mid\mid \sum T_j$ | NP-hard | DP |
| Weighted tardiness | $1 \mid\mid \sum w_j T_j$ | NP-hard (strongly) | B&B, SA |
| With release dates | $1 \mid r_j \mid \sum C_j$ | NP-hard | B&B |
| With setups (= TSP) | $1 \mid s_{jk} \mid C_{\max}$ | NP-hard | TSP methods |

### 2. Parallel Machine Scheduling
**Path**: `problems/scheduling/parallel_machine/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Identical, makespan | $P_m \mid\mid C_{\max}$ | NP-hard | LPT (4/3) |
| Preemptive makespan | $P_m \mid pmtn \mid C_{\max}$ | P | McNaughton |
| Total completion | $P_m \mid\mid \sum C_j$ | P | SPT |
| Uniform machines | $Q_m \mid\mid C_{\max}$ | NP-hard | |
| Unrelated machines | $R_m \mid\mid C_{\max}$ | NP-hard (strongly) | LP rounding |

### 3. Flow Shop Scheduling
**Path**: `problems/scheduling/flow_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 2-machine makespan | $F_2 \mid\mid C_{\max}$ | P | Johnson's rule |
| m-machine makespan | $F_m \mid\mid C_{\max}$ | NP-hard ($m \geq 3$) | NEH, IG |
| Permutation flow shop | $F_m \mid prmu \mid C_{\max}$ | NP-hard ($m \geq 3$) | NEH, IG |
| Blocking flow shop | $F_m \mid block \mid C_{\max}$ | NP-hard ($m \geq 3$) | |
| No-wait flow shop | $F_m \mid no\text{-}wait \mid C_{\max}$ | NP-hard ($m \geq 3$) | TSP reduction |

### 4. Job Shop Scheduling
**Path**: `problems/scheduling/job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Makespan | $J_m \mid\mid C_{\max}$ | NP-hard ($m \geq 2$) | Tabu search, CP |
| 2-job case | $J_m \mid n=2 \mid C_{\max}$ | P | Jackson's rule |
| With recirculation | $J_m \mid rcrc \mid C_{\max}$ | NP-hard | |

### 5. Flexible Job Shop Scheduling
**Path**: `problems/scheduling/flexible_job_shop/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total flexibility | T-FJSP | NP-hard | Integrated TS, GA |
| Partial flexibility | P-FJSP | NP-hard | Hierarchical TS |
| Multi-objective | MO-FJSP | NP-hard | NSGA-II |

### 6. Resource-Constrained Project Scheduling
**Path**: `problems/scheduling/rcpsp/`

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Basic | RCPSP | NP-hard (strongly) | GA + SGS |
| Multi-mode | MRCPSP | NP-hard | |
| Gen. precedence | RCPSP/max | NP-hard | |
| Multi-skill | MS-RCPSP | NP-hard | |

---

## Future Phases (Planned)

### Phase 2: Routing
- TSP, ATSP, CVRP, VRPTW, PDVRP, Arc Routing

### Phase 3: Packing & Cutting
- 0-1 Knapsack, Bin Packing (1D, 2D, 3D), Cutting Stock

### Phase 4: Location & Network
- Facility Location, p-Median, Hub Location, QAP, Shortest Path, Max Flow

### Phase 5: Stochastic & Robust
- Two-Stage SP, Robust Optimization, Chance-Constrained

# Uncapacitated Facility Location Problem (UFLP)

## 1. Problem Definition

- **Input:**
  - $m$ potential facility sites with opening costs $f_i$ ($i = 1, \ldots, m$)
  - $n$ customers with assignment costs $c_{ij}$ (cost to serve customer $j$ from facility $i$)
- **Decision:** Select facilities to open ($y_i \in \{0,1\}$) and assign each customer to an open facility ($x_{ij} \in \{0,1\}$)
- **Objective:** Minimize total fixed + assignment cost: $\sum_i f_i y_i + \sum_{i,j} c_{ij} x_{ij}$
- **Constraints:** Each customer assigned to exactly one open facility
- **Classification:** NP-hard. Best known approximation: 1.488 (Li, 2013)

The UFLP is also known as the Simple Plant Location Problem (SPLP) in the European
literature and the Warehouse Location Problem in supply chain contexts. It is a
fundamental model for discrete location theory, appearing in network design,
service facility siting, server placement, and distribution logistics.

### Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| UFLP | NP-hard | Cornuejols et al. (1977) |
| Capacitated FL (CFLP) | Strongly NP-hard | Cornuejols, Sridharan & Thizy (1991) |
| Metric UFLP | NP-hard, 1.488-approx | Li (2013) |
| Best approximation | 1.488 | Li (2013) |
| Primal-dual | 3-approximation | Jain & Vazirani (2001) |
| Greedy augmentation | 1.61-approximation | Jain, Mahdian & Saberi (2003) |
| LP-rounding | 4-approximation | Shmoys, Tardos & Aardal (1997) |
| Inapproximability | 1.463 (unless NP in DTIME(n^O(log log n))) | Guha & Khuller (1998) |

---

## 2. Mathematical Formulation

### 2.1 UFLP Primal (MILP)

$$\min \sum_{i=1}^{m} f_i y_i + \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \tag{1}$$

$$\sum_{i=1}^{m} x_{ij} = 1 \quad \forall j = 1, \ldots, n \quad \text{(each customer assigned)} \tag{2}$$

$$x_{ij} \leq y_i \quad \forall i = 1, \ldots, m,\; j = 1, \ldots, n \quad \text{(assign only to open facilities)} \tag{3}$$

$$y_i \in \{0,1\},\; x_{ij} \in \{0,1\} \tag{4}$$

The formulation has $m + m \cdot n$ binary variables and $m \cdot n + n$ constraints
(excluding variable bounds). In practice, the $x_{ij}$ integrality can be relaxed
to $x_{ij} \geq 0$ without loss of optimality: given integral $y$, each customer is
optimally assigned to its cheapest open facility, yielding integral $x$ automatically.

### 2.2 LP Relaxation and Integrality Gap

LP relaxation: replace (4) with $0 \leq y_i \leq 1$, $x_{ij} \geq 0$. The LP relaxation
of UFLP exhibits **half-integrality** — every extreme point has variables in
$\{0, 1/2, 1\}$ (Nemhauser & Wolsey, 1988). This structural property is exploited
by rounding-based approximation algorithms.

The integrality gap of the LP relaxation is at most $O(\log n)$ in the worst case,
and for metric instances (where $c_{ij}$ satisfies the triangle inequality), the gap
is bounded by a constant. Shmoys, Tardos & Aardal (1997) gave the first constant-factor
LP-rounding algorithm with ratio 4. Subsequent work tightened the ratio: Chudak &
Shmoys (2003) achieved $1 + 2/e \approx 1.736$ via randomized rounding, and Byrka &
Aardal (2010) obtained 1.5 via dependent rounding.

### 2.3 Dual Formulation and Complementary Slackness

The LP dual of the UFLP relaxation assigns a dual variable $v_j$ to each customer
assignment constraint (2) and $w_{ij} \geq 0$ to each linking constraint (3):

$$\max \sum_{j=1}^{n} v_j \tag{D1}$$

$$v_j - w_{ij} \leq c_{ij} \quad \forall i, j \tag{D2}$$

$$\sum_{j=1}^{n} w_{ij} \leq f_i \quad \forall i \tag{D3}$$

$$w_{ij} \geq 0 \tag{D4}$$

Interpretation: $v_j$ is the maximum amount customer $j$ is willing to "pay,"
and $w_{ij} = \max(0, v_j - c_{ij})$ is the contribution of customer $j$ toward
opening facility $i$. Facility $i$ is opened when the total contributions
$\sum_j w_{ij}$ cover its fixed cost $f_i$. The Jain-Vazirani primal-dual algorithm
is built directly on this dual structure.

Complementary slackness conditions for optimality:
- **Primal CS:** $x_{ij} > 0 \Rightarrow v_j - w_{ij} = c_{ij}$ (customer pays exactly its assignment cost)
- **Dual CS:** $y_i < 1 \Rightarrow \sum_j w_{ij} = f_i$ (partial opening absorbs full dual contribution)

### 2.4 Capacitated Variant (CFLP)

The Capacitated Facility Location Problem adds demand $d_j$ per customer and
capacity $u_i$ per facility. The additional constraint is:

$$\sum_{j=1}^{n} d_j x_{ij} \leq u_i y_i \quad \forall i = 1, \ldots, m \tag{5}$$

This couples $x$ and $y$ multiplicatively, making the LP relaxation weaker and
the problem strongly NP-hard. Unlike UFLP, the $x_{ij}$ integrality matters:
fractional assignments may be needed to respect capacities. The CFLP variant
is implemented in `variants/capacitated/` with its own instance, heuristics,
and metaheuristics.

---

## 3. Variants

| Variant | Directory | Key Difference | Complexity |
|---------|-----------|---------------|------------|
| Capacitated FL (CFLP) | `variants/capacitated/` | Each facility $i$ has capacity $u_i$; demand $d_j$ per customer | Strongly NP-hard |
| p-Median | `../../p_median/` | Exactly $p$ facilities opened; no fixed costs | NP-hard for general $p$ |

### 3.1 Capacitated Facility Location (CFLP)

The CFLP extends UFLP by assigning each customer a demand $d_j$ and each facility a
capacity $u_i$, adding constraint (5) from Section 2.4. Applications include
warehouse location with throughput limits, distribution center planning with truck
capacity, and server placement with bandwidth constraints.

The `variants/capacitated/` directory implements:
- `CFLPInstance` with capacities and demands
- Greedy heuristics adapted for capacity feasibility
- SA metaheuristic with capacity-violation penalty

---

## 4. Solution Methods

### 4.1 Approximation Algorithms

Several polynomial-time approximation algorithms exist for the metric UFLP:

**Jain-Vazirani Primal-Dual (3-approximation).** Simultaneously raises all customer
dual variables $v_j$ at uniform speed. When $v_j - c_{ij}$ exceeds zero for some
facility $i$, customer $j$ starts contributing to opening $i$. When contributions
$\sum_j \max(0, v_j - c_{ij})$ reach $f_i$, facility $i$ is "tentatively opened."
After all facilities are tentatively opened or all customers are served, a cleanup
phase removes redundancies. The algorithm achieves a 3-approximation guarantee and
runs in $O(m \cdot n \log(m \cdot n))$ time.

**Shmoys-Tardos-Aardal LP-Rounding (4-approximation).** Solves the LP relaxation
and rounds fractional facility openings using a filtering and rounding technique
based on grouping customers by their fractional assignment cost.

**Li's 1.488-Approximation.** Combines LP-rounding with a primal-dual approach
using dependent rounding on a carefully constructed bipartite graph. This is the
current best approximation ratio, close to the inapproximability threshold of 1.463
established by Guha & Khuller (1998).

### 4.2 Constructive Heuristics

- **Greedy Add:** Start with no facilities open. Iteratively open the facility giving the largest cost reduction. $O(m \cdot n)$ per iteration, $O(m^2 \cdot n)$ total.
- **Greedy Drop:** Start with all facilities open. Iteratively close the facility whose removal increases cost the least. $O(m \cdot n)$ per iteration, $O(m^2 \cdot n)$ total.

**Greedy Add pseudocode:**

```
GREEDY-ADD(f, c, m, n):
  S = {}
  open facility i* = argmin_i { f_i + sum_j c_ij }     // best single facility
  S = S + {i*}
  repeat:
    for each closed facility k not in S:
      delta_k = f_k - sum_{j: c_kj < c_{sigma(j),j}} (c_{sigma(j),j} - c_kj)
                // fixed cost minus reassignment savings
    k* = argmin_k delta_k
    if delta_{k*} < 0:
      S = S + {k*}
    else:
      break
  assign each customer j to argmin_{i in S} c_ij
  return S, assignments
```

Here $\sigma(j)$ denotes the facility currently serving customer $j$, and
$\delta_k$ is the marginal cost of opening facility $k$: its fixed cost minus the
total assignment savings from customers that would switch to it.

**Greedy Drop pseudocode:**

```
GREEDY-DROP(f, c, m, n):
  S = {1, 2, ..., m}                                   // all facilities open
  repeat:
    for each open facility k in S (|S| > 1):
      delta_k = -f_k + sum_{j: sigma(j)=k} (c_{sigma'(j),j} - c_kj)
                // saved fixed cost minus reassignment penalty
                // where sigma'(j) = nearest facility in S - {k}
    k* = argmin_k delta_k
    if delta_{k*} < 0:
      S = S - {k*}
    else:
      break
  assign each customer j to argmin_{i in S} c_ij
  return S, assignments
```

**Swap-based local search.** An important neighborhood for UFLP improvement:
simultaneously open a closed facility $i$ and close an open facility $j$. The swap
move evaluates $\delta(i, j)$: the cost change from reassigning all customers
served by $j$ to their best alternative (possibly $i$). There are $O(|S| \cdot (m - |S|))$
candidate swaps per iteration. Korupolu, Plaxton & Rajaraman (2000) showed
that single-swap local search achieves a 5-approximation for metric UFLP, and
Arya et al. (2004) improved this to 3 + 2/p for p-swap.

### 4.3 Metaheuristics

This repository implements **6 metaheuristics**:

| # | Method | Category | Key Feature |
|---|--------|----------|-------------|
| 1 | Local Search (LS) | Improvement | Add/drop/swap with best-improvement + random restarts |
| 2 | Simulated Annealing (SA) | Trajectory | Toggle/swap with Boltzmann acceptance |
| 3 | Tabu Search (TS) | Trajectory | Facility-level tabu with aspiration criterion |
| 4 | Iterated Greedy (IG) | Trajectory | Destroy (close d facilities) + greedy repair |
| 5 | Genetic Algorithm (GA) | Population | Binary encoding, uniform crossover, bit-flip mutation |
| 6 | Variable Neighborhood Search (VNS) | Trajectory | Toggle, swap, multi-toggle neighborhoods |

#### Default Parameters

**Simulated Annealing:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 30,000 | Total SA iterations |
| `initial_temp` | Auto ($0.05 \times$ greedy cost) | Starting temperature |
| `cooling_rate` | 0.9995 | Geometric cooling factor per iteration |
| Warm-start | Greedy Add | Initial solution |
| Neighborhoods | Toggle + Swap | Equally weighted random selection |

**Tabu Search:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 2,000 | Total TS iterations |
| `tabu_tenure` | $\lfloor\sqrt{m}\rfloor$ (min 3) | Iterations a facility remains tabu |
| Aspiration | Global best | Tabu overridden when move yields new best |
| Warm-start | Greedy Add | Initial solution |

**Genetic Algorithm:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 30 | Number of individuals |
| `generations` | 200 | Maximum generations |
| `crossover_rate` | 0.8 | Probability of uniform crossover |
| `mutation_rate` | $1/m$ | Per-gene bit-flip probability |
| Encoding | Binary vector | $m$-bit chromosome, gene $i$ = 1 if facility $i$ open |
| Selection | Tournament | Standard tournament selection |
| Warm-start | Greedy Add + Greedy Drop | Two seed individuals |

**Iterated Greedy:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 3,000 | Total IG iterations |
| `d` | Auto | Number of facilities to close per destroy phase |
| `temperature_factor` | 0.05 | Boltzmann acceptance temperature as fraction of initial cost |
| Warm-start | Greedy Add | Initial solution |

**Variable Neighborhood Search:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 500 | Total VNS iterations |
| `k_max` | 3 | Number of neighborhood structures |
| Neighborhoods | N1: Toggle, N2: Swap, N3: Multi-toggle | Ordered by size |
| Local search | Best-improvement toggle | Applied after shaking |
| Warm-start | Greedy Add | Initial solution |

---

## 5. Implementations in This Repository

```
facility_location/
├── instance.py                    # FacilityLocationInstance, FacilityLocationSolution, validation
│                                  #   random() — Euclidean instances with configurable params
│                                  #   from_ors() — real-world instances via OpenRouteService API
│                                  #   small_uflp_3_5() — 3 facilities, 5 customers benchmark
│                                  #   medium_uflp_5_10() — 5 facilities, 10 customers (seed=42)
├── heuristics/
│   └── greedy_facility.py         # greedy_add() — iterative facility opening, O(m^2 n)
│                                  # greedy_drop() — iterative facility closing, O(m^2 n)
├── metaheuristics/
│   ├── local_search.py            # local_search() — add/drop/swap, best-improvement
│   ├── simulated_annealing.py     # simulated_annealing() — toggle/swap, Boltzmann
│   ├── tabu_search.py             # tabu_search() — facility-level tabu, aspiration
│   ├── iterated_greedy.py         # iterated_greedy() — destroy-repair with acceptance
│   ├── genetic_algorithm.py       # genetic_algorithm() — binary encoding, uniform crossover
│   └── vns.py                     # vns() — N1/N2/N3 neighborhoods, shaking + LS
├── variants/
│   └── capacitated/               # CFLP variant
│       ├── instance.py            # CFLPInstance with capacities and demands
│       ├── heuristics.py          # Capacity-aware greedy heuristics
│       ├── metaheuristics.py      # SA with capacity-violation penalty
│       └── tests/test_cflp.py     # CFLP test suite
└── tests/                         # 6 test files
    ├── test_facility_location.py  # Core heuristic tests (greedy add/drop)
    ├── test_fl_ls.py              # Local search tests
    ├── test_fl_sa.py              # Simulated annealing tests (removed from main test file)
    ├── test_fl_ts.py              # Tabu search tests
    ├── test_fl_ig.py              # Iterated greedy tests
    ├── test_fl_ga.py              # Genetic algorithm tests
    └── test_fl_vns.py             # Variable neighborhood search tests
```

### Data Structures

```python
@dataclass
class FacilityLocationInstance:
    m: int                         # number of potential facility sites
    n: int                         # number of customers
    fixed_costs: np.ndarray        # shape (m,), opening cost f_i
    assignment_costs: np.ndarray   # shape (m, n), serving cost c_ij
    coords_facilities: np.ndarray | None  # optional (m, 2) coordinates
    coords_customers: np.ndarray | None   # optional (n, 2) coordinates

@dataclass
class FacilityLocationSolution:
    open_facilities: list[int]     # indices of opened facilities
    assignments: list[int]         # assignments[j] = facility for customer j
    cost: float                    # total cost (fixed + assignment)
```

---

## 6. Benchmark Instances

The OR-Library (Beasley, 1990) provides standard UFLP/CFLP benchmark instances
widely used in the literature:

| Instance Set | $m$ | $n$ | Type | Source |
|-------------|-----|-----|------|--------|
| cap41-cap44 | 16 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| cap51-cap54 | 16 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| cap61-cap64 | 16 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| cap71-cap74 | 16 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| cap101-cap104 | 25 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| cap131-cap134 | 50 | 50 | CFLP | Cornuejols, Sridharan & Thizy (1991) |
| capa-capb-capc | 100 | 1000 | CFLP | Large-scale variants |

These instances are available from the OR-Library at
`https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html`.
The instances encode fixed costs, capacities, demands, and assignment costs
in a standard format.

For UFLP specifically, removing the capacity constraints from cap instances
creates uncapacitated benchmarks. Additional UFLP instances include:

| Instance Set | Description | Source |
|-------------|-------------|--------|
| Krarup & Pruzan | Systematic test instances for SPLP | Krarup & Pruzan (1983) |
| Koerkel | Large-scale UFLP (up to 1000 facilities) | Koerkel (1989) |
| Galvao & Raggi | Euclidean plane instances | Galvao & Raggi (1989) |

---

## 7. Key References

1. Cornuejols, G., Fisher, M.L. & Nemhauser, G.L. (1977). Location of bank accounts to optimize float: An analytic study of exact and approximate algorithms. *Management Science*, 23(8), 789-810. https://doi.org/10.1287/mnsc.23.8.789

2. Cornuejols, G., Nemhauser, G.L. & Wolsey, L.A. (1990). The uncapacitated facility location problem. In: Mirchandani, P.B. & Francis, R.L. (eds) *Discrete Location Theory*, Wiley, 119-171.

3. Cornuejols, G., Sridharan, R. & Thizy, J.M. (1991). A comparison of heuristics and relaxations for the capacitated plant location problem. *European Journal of Operational Research*, 50(3), 280-297. https://doi.org/10.1016/0377-2217(91)90261-S

4. Shmoys, D.B., Tardos, E. & Aardal, K. (1997). Approximation algorithms for facility location problems. *Proceedings of the 29th ACM STOC*, 265-274. https://doi.org/10.1145/258533.258600

5. Guha, S. & Khuller, S. (1998). Greedy strikes back: Improved facility location algorithms. *Proceedings of the 9th ACM-SIAM SODA*, 649-657.

6. Jain, K. & Vazirani, V.V. (2001). Approximation algorithms for metric facility location and k-median problems using the primal-dual schema and Lagrangian relaxation. *Journal of the ACM*, 48(2), 274-296. https://doi.org/10.1145/375827.375845

7. Jain, K., Mahdian, M. & Saberi, A. (2003). A new greedy approach for facility location problems. *Proceedings of the 35th ACM STOC*, 731-740. https://doi.org/10.1145/780542.780645

8. Ghosh, D. (2003). Neighborhood search heuristics for the uncapacitated facility location problem. *European Journal of Operational Research*, 150(1), 150-162. https://doi.org/10.1016/S0377-2217(02)00504-6

9. Sun, M. (2006). Solving the uncapacitated facility location problem using tabu search. *Computers & Operations Research*, 33(9), 2563-2589. https://doi.org/10.1016/j.cor.2005.07.014

10. Li, S. (2013). A 1.488 approximation algorithm for the uncapacitated facility location problem. *Information and Computation*, 222, 45-58. https://doi.org/10.1016/j.ic.2012.01.007

11. Daskin, M.S. (2013). *Network and Discrete Location: Models, Algorithms, and Applications*, 2nd ed. Wiley. https://doi.org/10.1002/9781118537015

12. Krarup, J. & Pruzan, P.M. (1983). The simple plant location problem: Survey and synthesis. *European Journal of Operational Research*, 12(1), 36-81. https://doi.org/10.1016/0377-2217(83)90181-9

13. Nemhauser, G.L. & Wolsey, L.A. (1988). *Integer and Combinatorial Optimization*. Wiley.

14. Beasley, J.E. (1990). OR-Library: Distributing test problems by electronic mail. *Journal of the Operational Research Society*, 41(11), 1069-1072. https://doi.org/10.1057/jors.1990.166

15. Kratica, J., Tosic, D., Filipovic, V. & Ljubic, I. (2001). Solving the simple plant location problem by genetic algorithm. *RAIRO - Operations Research*, 35(1), 127-142. https://doi.org/10.1051/ro:2001107

16. Mladenovic, N. & Hansen, P. (1997). Variable neighborhood search. *Computers & Operations Research*, 24(11), 1097-1100. https://doi.org/10.1016/S0305-0548(97)00031-2

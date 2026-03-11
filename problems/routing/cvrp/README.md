# Capacitated Vehicle Routing Problem (CVRP)

## Problem Definition

Given $n$ customers with demands $q_i$, a depot (node 0), a fleet of identical vehicles each with capacity $Q$, and pairwise distances $d_{ij}$, find a set of vehicle routes starting and ending at the depot that:

- Visits each customer exactly once
- Does not exceed vehicle capacity on any route
- Minimizes total travel distance

## Mathematical Formulation

**Parameters:**
- $n$: number of customers
- $Q$: vehicle capacity
- $q_i$: demand of customer $i$, $i = 1, \ldots, n$
- $d_{ij}$: distance from node $i$ to node $j$ (node 0 = depot)
- $K$: number of available vehicles

**Decision variables:**
- $x_{ijk} \in \{0, 1\}$: 1 if vehicle $k$ travels from $i$ to $j$

**Objective:**

$$\min \sum_{k=1}^{K} \sum_{i=0}^{n} \sum_{j=0}^{n} d_{ij} x_{ijk}$$

**Subject to:**

$$\sum_{k=1}^{K} \sum_{j=0}^{n} x_{ijk} = 1 \quad \forall i \in \{1,\ldots,n\} \quad \text{(each customer visited once)}$$

$$\sum_{i=0}^{n} x_{i0k} = \sum_{j=0}^{n} x_{0jk} = 1 \quad \forall k \quad \text{(each vehicle starts and ends at depot)}$$

$$\sum_{i=1}^{n} q_i \sum_{j=0}^{n} x_{ijk} \leq Q \quad \forall k \quad \text{(capacity)}$$

## Complexity

NP-hard — generalizes both the Traveling Salesman Problem (single vehicle, no capacity) and the Bin Packing Problem (assignment only, no routing).

## Solution Approaches

### Constructive Heuristics

| Method | Complexity | Description |
|--------|-----------|-------------|
| Clarke-Wright Savings | $O(n^2 \log n)$ | Merge route pairs by largest savings $s(i,j) = d(0,i) + d(0,j) - d(i,j)$ |
| Sweep Algorithm | $O(n \log n)$ | Angular sweep from depot, start new route at capacity |

### Metaheuristics

| Method | Neighborhoods | Description |
|--------|--------------|-------------|
| Simulated Annealing | Relocate, swap, 2-opt* | Inter-route neighborhoods with Boltzmann acceptance |
| Genetic Algorithm | Giant-tour OX | Route-first cluster-second encoding (Prins, 2004) |

## Implementations in This Repository

```
cvrp/
├── instance.py              # CVRPInstance, CVRPSolution, validation
├── heuristics/
│   ├── clarke_wright.py     # Clarke-Wright savings algorithm
│   └── sweep.py             # Angular sweep + multi-start
├── metaheuristics/
│   ├── simulated_annealing.py # Relocate/swap/2-opt* neighborhoods
│   └── genetic_algorithm.py # Giant-tour encoding with split decoder
└── tests/
    └── test_cvrp.py         # 41 tests, 8 test classes
```

## Benchmark Instances

| Instance | Customers | Capacity | Source |
|----------|-----------|----------|--------|
| small6 | 6 | 15 | Handcrafted (two clusters) |
| christofides1 | 5 | 6 | Clarke-Wright example |
| medium12 | 12 | 40 | Random with mixed demands |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Multi-Depot VRP (MDVRP)](variants/multi_depot/) | `variants/multi_depot/` | Multiple depots, each vehicle assigned to a home depot |
| [Split Delivery VRP (SDVRP)](variants/split_delivery/) | `variants/split_delivery/` | Customer demand can be split across multiple vehicles |
| [Open VRP (OVRP)](variants/open_vrp/) | `variants/open_vrp/` | Vehicles need not return to the depot |
| [Periodic VRP (PVRP)](variants/periodic/) | `variants/periodic/` | Customers require service on specific day combinations over a planning horizon |
| [Cumulative VRP (CumVRP)](variants/cumulative/) | `variants/cumulative/` | Minimize sum of arrival times (latency) rather than total distance |
| [VRP with Backhauls (VRPB)](variants/backhaul/) | `variants/backhaul/` | Linehaul deliveries before backhaul pickups |
| [Electric VRP (EVRP)](variants/electric/) | `variants/electric/` | Battery constraints with recharging stations |
| [Multi-Compartment VRP (MCVRP)](variants/multi_compartment/) | `variants/multi_compartment/` | Vehicles have separate compartments for incompatible goods |
| [Multi-Trip VRP (MTVRP)](variants/multi_trip/) | `variants/multi_trip/` | Vehicles can make multiple trips to reduce fleet size |

## Key References

- Dantzig, G.B. & Ramser, J.H. (1959). The truck dispatching problem. *Management Science*, 6(1), 80-91. https://doi.org/10.1287/mnsc.6.1.80
- Clarke, G. & Wright, J.W. (1964). Scheduling of vehicles from a central depot. *Oper. Res.*, 12(4), 568-581. https://doi.org/10.1287/opre.12.4.568
- Gillett, B.E. & Miller, L.R. (1974). A heuristic algorithm for the vehicle-dispatch problem. *Oper. Res.*, 22(2), 340-349. https://doi.org/10.1287/opre.22.2.340
- Prins, C. (2004). A simple and effective evolutionary algorithm for the VRP. *Comput. Oper. Res.*, 31(12), 1985-2002. https://doi.org/10.1016/S0305-0548(03)00158-8
- Toth, P. & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). SIAM. https://doi.org/10.1137/1.9781611973594

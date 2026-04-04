# Inventory-Routing Problem (IRP)

## Family 8 -- Integrated Structural

The Inventory-Routing Problem jointly optimizes inventory replenishment decisions
(when and how much to deliver) and vehicle routing decisions (which customers to
visit on each route) over a multi-period planning horizon.

---

## 1. Problem Definition

**Input:**
- A depot (node 0) with unlimited supply
- n customers, each with:
  - Deterministic per-period demand rate d_i
  - Per-unit per-period holding cost h_i
  - Maximum storage capacity U_i
  - Initial inventory level I_i^0
- Euclidean distance matrix between all nodes (depot + customers)
- K homogeneous vehicles, each with capacity Q
- Planning horizon of T discrete periods

**Decision variables:**
- For each period t: which customers to visit
- Delivery quantities q_{it} for each visited customer i in period t
- Vehicle routes (sequence of customers per vehicle per period)

**Objective:** Minimize total cost = routing cost + holding cost

**Constraints:**
- No stockouts: inventory at each customer must remain non-negative after demand consumption
- Storage capacity: inventory after delivery must not exceed U_i
- Vehicle capacity: total delivery on each route must not exceed Q
- Vehicle availability: at most K routes per period
- Each customer visited at most once per period

**Complexity:** NP-hard (generalizes both VRP and multi-period lot-sizing).

---

## 2. Mathematical Formulation

### Sets and Parameters

| Symbol | Description |
|--------|-------------|
| N = {1,...,n} | Set of customers |
| V = {0} U N | All nodes (depot + customers) |
| T = {1,...,T} | Planning periods |
| K = {1,...,K} | Vehicle set |
| d_i | Per-period demand of customer i |
| h_i | Holding cost per unit per period for customer i |
| U_i | Storage capacity of customer i |
| I_i^0 | Initial inventory of customer i |
| Q | Vehicle capacity |
| c_{ij} | Travel distance from node i to node j |

### Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| x_{ijk}^t | {0,1} | 1 if vehicle k travels arc (i,j) in period t |
| y_{ik}^t | {0,1} | 1 if customer i is served by vehicle k in period t |
| q_{it} | R+ | Quantity delivered to customer i in period t |
| I_{it} | R+ | Inventory level of customer i at end of period t |

### MILP Formulation

```
min  sum_{t in T} sum_{k in K} sum_{(i,j) in A} c_{ij} * x_{ijk}^t
   + sum_{t in T} sum_{i in N} h_i * I_{it}

subject to:

  (1) I_{it} = I_{i,t-1} + q_{it} - d_i               for all i in N, t in T
  (2) 0 <= I_{it} <= U_i                                for all i in N, t in T
  (3) q_{it} <= U_i * sum_{k in K} y_{ik}^t             for all i in N, t in T
  (4) sum_{i in N} q_{it} * y_{ik}^t <= Q               for all k in K, t in T
  (5) sum_{k in K} y_{ik}^t <= 1                         for all i in N, t in T
  (6) sum_{j in V} x_{ijk}^t = y_{ik}^t                 for all i in N, k in K, t in T
  (7) sum_{i in V} x_{ijk}^t = sum_{i in V} x_{jik}^t  for all j in V, k in K, t in T
  (8) subtour elimination constraints
  (9) I_{i0} = I_i^0                                     for all i in N
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|------------|-------------|
| Greedy IRP | Constructive | O(T * n^2) | Period-by-period urgency-based insertion |
| Greedy Fill-Up | Constructive | O(T * n^2) | Visit all, fill to capacity every period |
| Simulated Annealing | Metaheuristic | O(iter * T * n) | Reassign/swap/2-opt neighborhoods |

---

## 4. Implementations

```
inventory_routing/
    instance.py                     # IRPInstance, IRPSolution, compute_cost
    heuristics/
        greedy_irp.py               # Urgency-based greedy + fill-up baseline
    metaheuristics/
        simulated_annealing.py      # SA with multi-move neighborhood
    tests/
        test_irp.py                 # 25+ tests across 4 test classes
    README.md                       # This file
```

---

## 5. Key References

1. Federgruen, A. & Zipkin, P. (1984). A combined vehicle routing and
   inventory allocation problem. *Operations Research*, 32(5), 1019-1037.
   https://doi.org/10.1287/opre.32.5.1019

2. Campbell, A., Clarke, L., Kleywegt, A. & Savelsbergh, M. (1998).
   The inventory routing problem. In: *Fleet Management and Logistics*,
   pp. 95-113. Springer.
   https://doi.org/10.1007/978-1-4615-5755-5_4

3. Bertazzi, L., Savelsbergh, M. & Speranza, M.G. (2008). Inventory
   routing. In: *The Vehicle Routing Problem: Latest Advances and New
   Challenges*, pp. 49-72. Springer.
   https://doi.org/10.1007/978-0-387-77778-8_3

4. Coelho, L.C., Cordeau, J.-F. & Laporte, G. (2014). Thirty years
   of inventory routing. *Transportation Science*, 48(1), 1-19.
   https://doi.org/10.1287/trsc.2013.0472

5. Archetti, C., Bertazzi, L., Laporte, G. & Speranza, M.G. (2007).
   A branch-and-cut algorithm for a vendor-managed inventory-routing
   problem. *Transportation Science*, 41(3), 382-391.
   https://doi.org/10.1287/trsc.1060.0188

6. Adulyasak, Y., Cordeau, J.-F. & Jans, R. (2015). The production
   routing problem: A review of formulations and solution algorithms.
   *Computers & Operations Research*, 55, 141-152.
   https://doi.org/10.1016/j.cor.2014.01.011

---

## See also

- [`../../7_inventory_lotsizing/`](../../7_inventory_lotsizing/) -- Inventory and lot sizing implementations
- [`../../2_routing/`](../../2_routing/) -- Vehicle routing implementations

# Open Vehicle Routing Problem (OVRP) (CVRP Variant)

## What Changes

In standard CVRP, every route forms a closed loop: depot to customers to depot. The OVRP removes the requirement that vehicles return to the depot after serving their last customer. Each route is an open path starting at the depot and ending at the last customer visited. This models scenarios where drivers go home after their final delivery rather than returning to a central warehouse, such as courier services, school bus routing (buses end at the school, not the garage), home healthcare visits (nurses go home after last patient), and newspaper delivery.

The removal of the return leg changes the cost structure. In CVRP, the last customer on a route should ideally be near the depot; in OVRP, the last customer's location relative to the depot is irrelevant.

Compared to the base CVRP:
- Routes do not include the return trip from last customer to depot.
- Route cost is the sum of edge costs from depot through the customer sequence (no closing arc).
- Capacity constraints remain unchanged.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Open routes:** For each route k, the cost is sum_{edges in path from depot to last customer} d(i,j). The edge from the last customer back to depot is not included.
- **Route structure:** Each route starts at depot (node 0) and ends at some customer node. There is no constraint forcing a return to depot.
- **No closing arc:** The decision variable for the arc (last_customer, depot) is removed or set to zero cost.

```
min  sum_k path_distance(route_k)
s.t. sum_{i in route_k} d_i <= Q        for all k  (capacity)
     each customer visited exactly once
     routes start at depot, end at last customer
```

All other constraints (capacity, visit-once) remain unchanged.

## Complexity

NP-hard (generalizes TSP, since a single-vehicle OVRP with infinite capacity is the shortest Hamiltonian path problem, which is NP-hard). The OVRP is not simply CVRP with modified costs -- the open-path structure affects the local search neighborhood topology.

## Applications

- **Courier delivery**: drivers go home after last delivery
- **School bus routing**: buses end at the school
- **Home healthcare**: nurses visit patients then go home
- **Newspaper delivery**: carriers end their route at home

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor OVRP | Heuristic | Yes | Greedy nearest with no return-to-depot cost |
| Modified Savings | Heuristic | Yes | Clarke-Wright with savings excluding return leg |
| Simulated Annealing | Metaheuristic | Yes | Relocate/swap/2-opt* with open-route cost evaluation |

Standard Clarke-Wright savings are modified: the savings formula s(i,j) = d(0,i) + d(0,j) - d(i,j) is adjusted because routes no longer include return trips. The SA neighborhoods (relocate, swap, 2-opt*) must evaluate route costs as open paths rather than closed tours.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `OVRPInstance` dataclass with open-path route cost computation and validation |
| `heuristics.py` | Nearest neighbor OVRP; modified Clarke-Wright savings for open routes |
| `metaheuristics.py` | Simulated Annealing with relocate, swap, and 2-opt* neighborhoods on open routes |

## Key References

- Sariklis, D. & Powell, S. (2000). A heuristic method for the open vehicle routing problem. *Journal of the Operational Research Society*, 51(5), 564-573. https://doi.org/10.1057/palgrave.jors.2600924
- Li, F., Golden, B. & Wasil, E. (2007). The open vehicle routing problem: Algorithms, large-scale test problems, and computational results. *Computers & Operations Research*, 34(10), 2918-2930. https://doi.org/10.1016/j.cor.2005.11.018
- Brandao, J. (2004). A tabu search algorithm for the open vehicle routing problem. *European Journal of Operational Research*, 157(3), 552-564. [TODO: verify DOI]

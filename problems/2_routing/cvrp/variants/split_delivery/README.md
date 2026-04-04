# Split Delivery Vehicle Routing Problem (SDVRP) (CVRP Variant)

## What Changes

Standard CVRP requires each customer to be served by exactly one vehicle in a single visit. The SDVRP relaxes this by allowing a customer's demand to be split across multiple vehicles and visits. If a customer's demand is large relative to vehicle capacity, or if splitting enables more efficient route combinations, a customer may receive partial deliveries from different routes. This models bulk delivery scenarios (e.g., delivering construction materials, fuel, or agricultural supplies) where a single truckload may not cover a customer's full order.

Dror and Trudeau (1989) showed that split deliveries can reduce total distance by up to 50% compared to non-split CVRP, and can also reduce the number of required vehicles.

Compared to the base CVRP:
- The visit-once constraint is relaxed: a customer may appear on multiple routes.
- Each visit delivers a partial quantity; the sum of partial deliveries must equal the customer's total demand.
- Customers with demand exceeding Q (which are infeasible in standard CVRP) become feasible.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Split delivery:** For each customer i, let q_{i,k} >= 0 be the quantity delivered to customer i by route k. The constraint sum_k q_{i,k} = d_i ensures total demand satisfaction.
- **Relaxed visit-once:** Customer i may appear on multiple routes. The binary visit variable is replaced by a continuous delivery fraction.
- **Per-route capacity:** sum_{i in route_k} q_{i,k} <= Q for each route k.
- **Non-trivial visit:** If route k visits customer i, then q_{i,k} > 0 (no empty visits).

The objective remains minimizing total travel distance. Note that a customer visited by multiple routes incurs travel cost for each visit.

## Complexity

NP-hard (generalizes CVRP, since when all demands d_i <= Q and no splits are beneficial, the optimal SDVRP solution is the same as the optimal CVRP solution). However, SDVRP can be easier in practice because the feasible region is larger -- more solutions are feasible due to the relaxed visit constraint.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor Split | Heuristic | Yes | NN that splits delivery when remaining capacity is insufficient |
| Savings Split | Heuristic | Yes | Adapted Clarke-Wright with split-aware route merging |
| Simulated Annealing | Metaheuristic | Yes | Relocate, swap, and resplit moves at the route level |

The key heuristic challenge is deciding when to split: splitting always satisfies demand but adds an extra visit cost. The NN heuristic delivers as much as possible to the current customer, then continues the route or starts a new one with the remaining demand. The SA resplit move re-optimizes how demand is divided between routes sharing a customer.

## Applications

- **Bulk material delivery**: construction materials, gravel, or sand where a single truck cannot carry the full order
- **Fuel distribution**: gas stations with large tank capacities that exceed a single tanker's load
- **Agricultural supply**: fertilizer or feed delivery to large farms requiring multiple truckloads
- **Industrial chemicals**: large-volume orders that must be split across multiple tanker trips

When customer demands are small relative to vehicle capacity (d_i << Q for all i), SDVRP solutions tend to coincide with CVRP solutions because splitting offers no benefit. The greatest savings occur when several customers have demands close to Q.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `SDVRPInstance` dataclass with demand splitting support, partial delivery tracking, and validation |
| `heuristics.py` | Route-first split-second; nearest neighbor with demand splitting when capacity reached |
| `metaheuristics.py` | Simulated Annealing with relocate, swap, and resplit neighborhoods |

## Key References

- Dror, M. & Trudeau, P. (1989). Savings by split delivery routing. *Transportation Science*, 23(2), 141-145. https://doi.org/10.1287/trsc.23.2.141
- Archetti, C., Savelsbergh, M.W.P. & Speranza, M.G. (2006). Worst-case analysis for split delivery vehicle routing problems. *Transportation Science*, 40(2), 226-234. https://doi.org/10.1287/trsc.1050.0117
- Archetti, C. & Speranza, M.G. (2012). Vehicle routing problems with split deliveries. *International Transactions in Operational Research*, 19(1-2), 3-22. https://doi.org/10.1111/j.1475-3995.2011.00811.x
